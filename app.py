"""
app.py
Giao diện người dùng (Streamlit Web App) cho bài toán Dự đoán thời gian giao đồ ăn.
Sử dụng mô hình Random Forest Regression.
"""

import json
import os
import sys
import streamlit as st

# Thêm đường dẫn src vào hệ thống
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import calculate_haversine_distance
from predict import load_model, predict_single

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Dự đoán thời gian giao hàng | DeliTime",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .main {
        background-color: #f0f4f8;
    }

    /* ---- Header ---- */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.25);
    }
    .app-header-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .app-header-sub {
        font-size: 14px;
        color: rgba(255,255,255,0.78);
        margin: 0;
        line-height: 1.6;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        color: #fff;
        margin-right: 8px;
        margin-top: 10px;
    }

    /* ---- Preset Buttons ---- */
    .preset-label {
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        letter-spacing: 0.4px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    /* ---- Section title ---- */
    .section-title {
        font-size: 13px;
        font-weight: 700;
        color: #1e40af;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #dbeafe;
    }

    /* ---- Form container ---- */
    .form-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }

    /* ---- Result card ---- */
    .result-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 18px;
        box-shadow: 0 6px 24px rgba(29, 78, 216, 0.3);
        color: #ffffff;
    }
    .result-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: rgba(255,255,255,0.7);
        margin-bottom: 6px;
    }
    .result-number {
        font-size: 56px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        display: inline-block;
    }
    .result-unit {
        font-size: 22px;
        font-weight: 600;
        color: rgba(255,255,255,0.8);
        margin-left: 6px;
        vertical-align: bottom;
        padding-bottom: 8px;
        display: inline-block;
    }
    .result-range {
        font-size: 13.5px;
        color: rgba(255,255,255,0.72);
        margin-top: 10px;
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 8px 12px;
        display: inline-block;
    }

    /* ---- Factor pills ---- */
    .factors-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .factor-pill {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 13px;
        color: #334155;
        font-weight: 500;
    }
    .factor-pill b {
        color: #1d4ed8;
    }

    /* ---- GPS info box ---- */
    .gps-dist-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 14px;
        color: #1e40af;
        font-weight: 500;
        margin-bottom: 14px;
    }

    /* ---- Sidebar tweaks ---- */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* ---- Submit button style ---- */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 14px !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 14px;
    }
</style>
""", unsafe_allow_html=True)


# ===================== LOAD MODEL =====================
@st.cache_resource
def get_cached_model():
    """Nạp mô hình và metadata với cơ chế cache."""
    model_path = os.path.join(CURRENT_DIR, "models", "delivery_pipeline.joblib")
    meta_path = os.path.join(CURRENT_DIR, "models", "model_metadata.json")
    pipeline = None
    metadata = {}
    if os.path.exists(model_path):
        pipeline = load_model(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return pipeline, metadata


pipeline, metadata = get_cached_model()


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### 🛵 DeliTime")
    st.caption("Hệ thống dự đoán thời gian giao đồ ăn bằng Machine Learning")
    st.divider()

    st.markdown("**📊 Độ chính xác mô hình**")
    if metadata and "metrics" in metadata:
        m = metadata["metrics"]
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("MAE", f"{m.get('validation_mae_minutes', 3.27)} p")
        col_m2.metric("R² Score", f"{m.get('validation_r2_score', 0.81)*100:.1f}%")
        st.caption(
            f"RMSE: {m.get('validation_rmse_minutes', 4.12)} phút  \n"
            f"Số cây: {metadata.get('n_estimators', 120)} | "
            f"Tập val: {metadata.get('validation_samples', 9119):,} mẫu"
        )
    else:
        st.info("Chưa tìm thấy metadata. Chạy `python src/train.py` trước.")

    st.divider()
    st.markdown("**📈 Biểu đồ huấn luyện**")
    show_plots = st.checkbox("Xem biểu đồ đánh giá", value=False)
    if show_plots:
        fi_path = os.path.join(CURRENT_DIR, "plots", "feature_importance.png")
        act_path = os.path.join(CURRENT_DIR, "plots", "actual_vs_predicted.png")
        res_path = os.path.join(CURRENT_DIR, "plots", "residuals_distribution.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Tầm quan trọng đặc trưng", use_container_width=True)
        if os.path.exists(act_path):
            st.image(act_path, caption="Thực tế vs Dự đoán", use_container_width=True)
        if os.path.exists(res_path):
            st.image(res_path, caption="Phân phối sai số", use_container_width=True)

    st.divider()
    st.markdown("**⚙️ Thông tin kỹ thuật**")
    st.caption(
        "Mô hình: Random Forest Regressor  \n"
        "Đặc trưng: 15 biến (9 số + 6 phân loại)  \n"
        "Pipeline: Imputer → Scaler → OHE → RF"
    )


# ===================== MAIN HEADER =====================
st.markdown("""
<div class="app-header">
    <div class="app-header-title">🛵 DeliTime — Dự đoán thời gian giao đồ ăn</div>
    <div class="app-header-sub">
        Ứng dụng học máy sử dụng <b>Random Forest Regression</b> để ước tính thời gian giao hàng
        dựa trên quãng đường, điều kiện giao thông, thời tiết và thông tin tài xế.
    </div>
    <div>
        <span class="header-badge">🎯 MAE: 3.27 phút</span>
        <span class="header-badge">📊 R²: 80.68%</span>
        <span class="header-badge">🌳 120 cây quyết định</span>
    </div>
</div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Chưa tìm thấy tệp mô hình! Hãy chạy `python src/train.py` trước.")
    st.stop()


# ===================== KỊCH BẢN MẪU =====================
st.markdown('<div class="preset-label">⚡ Thử nhanh kịch bản mẫu</div>', unsafe_allow_html=True)
col_pre1, col_pre2, col_pre3, col_pre_reset = st.columns(4)

if "preset_data" not in st.session_state:
    st.session_state.preset_data = {}

if col_pre1.button("🚗 Giờ cao điểm kẹt xe", use_container_width=True):
    st.session_state.preset_data = {
        "age": 25, "rating": 4.5, "dist": 8.0, "weather": "Cloudy",
        "traffic": "Jam", "vehicle_cond": 1, "order_type": "Meal",
        "vehicle_type": "motorcycle", "multiple": 1, "festival": "No",
        "city": "Metropolitian", "hour": 19, "prep": 15
    }

if col_pre2.button("☀️ Trưa nắng thông thoáng", use_container_width=True):
    st.session_state.preset_data = {
        "age": 30, "rating": 4.9, "dist": 3.5, "weather": "Sunny",
        "traffic": "Low", "vehicle_cond": 2, "order_type": "Snack",
        "vehicle_type": "scooter", "multiple": 0, "festival": "No",
        "city": "Urban", "hour": 14, "prep": 5
    }

if col_pre3.button("🌧️ Mưa bão dịp lễ hội", use_container_width=True):
    st.session_state.preset_data = {
        "age": 28, "rating": 4.2, "dist": 12.0, "weather": "Stormy",
        "traffic": "Jam", "vehicle_cond": 0, "order_type": "Buffet",
        "vehicle_type": "motorcycle", "multiple": 2, "festival": "Yes",
        "city": "Metropolitian", "hour": 20, "prep": 15
    }

if col_pre_reset.button("🔄 Đặt lại mặc định", use_container_width=True):
    st.session_state.preset_data = {}

preset = st.session_state.preset_data

st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)


# ===================== TABS =====================
tab_manual, tab_gps = st.tabs([
    "📍 Nhập thông tin & Khoảng cách (km)",
    "🗺️ Nhập theo Toạ độ GPS (Lat/Lon)"
])


# ─────────────── TAB 1: NHẬP THEO KM ───────────────
with tab_manual:
    with st.form("form_predict_distance"):
        col1, col2, col3 = st.columns(3)

        # Cột 1: Tài xế & Xe
        with col1:
            st.markdown('<div class="section-title">🧑‍✈️ Tài xế &amp; Phương tiện</div>', unsafe_allow_html=True)
            p_age = st.number_input(
                "Tuổi tài xế",
                min_value=18, max_value=60,
                value=preset.get("age", 28), step=1,
                help="Độ tuổi của tài xế giao hàng (18–60 tuổi)"
            )
            p_rating = st.slider(
                "Điểm đánh giá tài xế",
                min_value=1.0, max_value=5.0,
                value=float(preset.get("rating", 4.7)), step=0.1,
                help="Điểm đánh giá trung bình của tài xế (1.0 – 5.0 sao)"
            )
            p_vehicle = st.selectbox(
                "Loại phương tiện",
                options=["motorcycle", "scooter", "electric_scooter", "bicycle"],
                index=["motorcycle", "scooter", "electric_scooter", "bicycle"].index(
                    preset.get("vehicle_type", "motorcycle")
                ),
                format_func=lambda x: {
                    "motorcycle": "🏍️ Xe máy xăng",
                    "scooter": "🛵 Xe tay ga",
                    "electric_scooter": "⚡ Xe máy điện",
                    "bicycle": "🚲 Xe đạp"
                }.get(x, x)
            )
            p_vcond = st.selectbox(
                "Tình trạng phương tiện",
                options=[2, 1, 0, 3],
                index=[2, 1, 0, 3].index(preset.get("vehicle_cond", 2)),
                format_func=lambda x: {
                    3: "⭐ Rất tốt (Mức 3)",
                    2: "✅ Tốt (Mức 2)",
                    1: "🔶 Bình thường (Mức 1)",
                    0: "🔴 Kém (Mức 0)"
                }.get(x, str(x))
            )

        # Cột 2: Môi trường & Giao thông
        with col2:
            st.markdown('<div class="section-title">🌍 Môi trường &amp; Giao thông</div>', unsafe_allow_html=True)
            p_dist = st.number_input(
                "Khoảng cách giao hàng (km)",
                min_value=0.5, max_value=30.0,
                value=float(preset.get("dist", 6.5)), step=0.5,
                help="Khoảng cách thực tế từ nhà hàng đến điểm giao"
            )
            p_traffic = st.selectbox(
                "Mật độ giao thông",
                options=["Low", "Medium", "High", "Jam"],
                index=["Low", "Medium", "High", "Jam"].index(preset.get("traffic", "Medium")),
                format_func=lambda x: {
                    "Low": "🟢 Thông thoáng",
                    "Medium": "🟡 Trung bình",
                    "High": "🟠 Đông đúc",
                    "Jam": "🔴 Ùn tắc nặng"
                }.get(x, x)
            )
            p_weather = st.selectbox(
                "Điều kiện thời tiết",
                options=["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"],
                index=["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"].index(
                    preset.get("weather", "Sunny")
                ),
                format_func=lambda x: {
                    "Sunny": "☀️ Nắng ráo",
                    "Cloudy": "☁️ Nhiều mây",
                    "Windy": "💨 Gió lớn",
                    "Fog": "🌫️ Sương mù",
                    "Sandstorms": "🌪️ Bão cát",
                    "Stormy": "⛈️ Mưa bão"
                }.get(x, x)
            )
            p_city = st.selectbox(
                "Khu vực thành phố",
                options=["Metropolitian", "Urban", "Semi-Urban"],
                index=["Metropolitian", "Urban", "Semi-Urban"].index(preset.get("city", "Metropolitian")),
                format_func=lambda x: {
                    "Metropolitian": "🏙️ Đô thị lớn",
                    "Urban": "🏘️ Nội thành",
                    "Semi-Urban": "🌄 Ngoại ô"
                }.get(x, x)
            )

        # Cột 3: Đơn hàng & Thời gian
        with col3:
            st.markdown('<div class="section-title">📦 Đơn hàng &amp; Thời gian</div>', unsafe_allow_html=True)
            p_order_type = st.selectbox(
                "Loại đồ ăn",
                options=["Meal", "Snack", "Drinks", "Buffet"],
                index=["Meal", "Snack", "Drinks", "Buffet"].index(preset.get("order_type", "Meal")),
                format_func=lambda x: {
                    "Meal": "🍱 Bữa chính",
                    "Snack": "🍿 Đồ ăn nhẹ",
                    "Drinks": "🥤 Đồ uống",
                    "Buffet": "🍽️ Tiệc Buffet"
                }.get(x, x)
            )
            p_multi = st.selectbox(
                "Số đơn giao cùng lúc",
                options=[0, 1, 2, 3],
                index=[0, 1, 2, 3].index(preset.get("multiple", 1)),
                format_func=lambda x: "🎯 Chỉ 1 đơn duy nhất" if x == 0 else f"📦 Ghép {x} đơn khác",
                help="Số lượng đơn hàng tài xế giao cùng 1 chuyến"
            )
            p_hour = st.slider(
                "Giờ đặt hàng (0h – 23h)",
                min_value=0, max_value=23,
                value=int(preset.get("hour", 18)), step=1,
                help="Khung giờ đặt hàng trong ngày"
            )
            p_prep = st.slider(
                "Thời gian nhà hàng chuẩn bị (phút)",
                min_value=5, max_value=20,
                value=int(preset.get("prep", 10)), step=5,
                help="Thời gian nhà hàng chế biến và đóng gói món ăn"
            )
            p_festival = st.selectbox(
                "Dịp lễ hội",
                options=["No", "Yes"],
                index=["No", "Yes"].index(preset.get("festival", "No")),
                format_func=lambda x: "🎉 Có — Ngày lễ hội" if x == "Yes" else "📅 Không — Ngày thường"
            )

        submit_btn = st.form_submit_button("🛵 DỰ ĐOÁN THỜI GIAN GIAO HÀNG", use_container_width=True)

    # Xử lý kết quả Tab 1
    if submit_btn:
        order_dict = {
            "Delivery_person_Age": p_age,
            "Delivery_person_Ratings": p_rating,
            "distance_km": p_dist,
            "Weatherconditions": p_weather,
            "Road_traffic_density": p_traffic,
            "Vehicle_condition": p_vcond,
            "Type_of_order": p_order_type,
            "Type_of_vehicle": p_vehicle,
            "multiple_deliveries": p_multi,
            "Festival": p_festival,
            "City": p_city,
            "order_hour": p_hour,
            "day_of_week": 3,
            "is_weekend": 0,
            "prep_time_min": p_prep
        }
        try:
            res = predict_single(order_dict)
            minutes = res["predicted_minutes"]
            low = res["min_estimated_minutes"]
            high = res["max_estimated_minutes"]

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">⏱️ Kết quả dự đoán</div>
                <span class="result-number">{minutes}</span>
                <span class="result-unit">phút</span>
                <div class="result-range">
                    📊 Khoảng ước lượng: <b>{low} – {high} phút</b> &nbsp;|&nbsp; Độ lệch chuẩn ±3.27 phút
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Yếu tố ảnh hưởng
            traffic_vn = {"Low": "Thông thoáng", "Medium": "Trung bình", "High": "Đông đúc", "Jam": "Ùn tắc"}
            weather_vn = {
                "Sunny": "Nắng", "Cloudy": "Nhiều mây", "Windy": "Gió lớn",
                "Fog": "Sương mù", "Sandstorms": "Bão cát", "Stormy": "Mưa bão"
            }
            st.markdown(f"""
            <div style="margin-top: 14px;">
                <div style="font-size:13px; font-weight:600; color:#475569; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.4px;">
                    🔎 Các yếu tố chính ảnh hưởng đến kết quả:
                </div>
                <div class="factors-row">
                    <span class="factor-pill">📍 Quãng đường: <b>{p_dist} km</b></span>
                    <span class="factor-pill">🚦 Giao thông: <b>{traffic_vn.get(p_traffic, p_traffic)}</b></span>
                    <span class="factor-pill">⛅ Thời tiết: <b>{weather_vn.get(p_weather, p_weather)}</b></span>
                    <span class="factor-pill">📦 Đơn ghép: <b>{p_multi} đơn khác</b></span>
                    <span class="factor-pill">⭐ Tài xế: <b>{p_rating} sao / {p_age} tuổi</b></span>
                    <span class="factor-pill">🍳 Chuẩn bị: <b>{p_prep} phút</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")


# ─────────────── TAB 2: NHẬP THEO GPS ───────────────
with tab_gps:
    st.markdown(
        "Nhập toạ độ GPS của nhà hàng và điểm giao hàng. "
        "Hệ thống sẽ tự động tính khoảng cách theo **công thức Haversine** rồi thực hiện dự đoán."
    )
    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)

    with st.form("form_predict_gps"):
        cg1, cg2 = st.columns(2)
        with cg1:
            st.markdown('<div class="section-title">🍴 Toạ độ Nhà hàng</div>', unsafe_allow_html=True)
            r_lat = st.number_input("Vĩ độ (Latitude)", value=12.9716, format="%.6f",
                                    help="Vĩ độ của nhà hàng. Ví dụ: 12.9716")
            r_lon = st.number_input("Kinh độ (Longitude)", value=77.5946, format="%.6f",
                                    help="Kinh độ của nhà hàng. Ví dụ: 77.5946")
        with cg2:
            st.markdown('<div class="section-title">📬 Toạ độ Điểm giao hàng</div>', unsafe_allow_html=True)
            d_lat = st.number_input("Vĩ độ (Latitude) ", value=13.0116, format="%.6f",
                                    help="Vĩ độ điểm giao hàng. Ví dụ: 13.0116")
            d_lon = st.number_input("Kinh độ (Longitude) ", value=77.6346, format="%.6f",
                                    help="Kinh độ điểm giao hàng. Ví dụ: 77.6346")

        st.divider()

        cgps1, cgps2, cgps3 = st.columns(3)
        with cgps1:
            gps_traffic = st.selectbox(
                "Mật độ giao thông", ["Low", "Medium", "High", "Jam"], index=1,
                format_func=lambda x: {
                    "Low": "🟢 Thông thoáng", "Medium": "🟡 Trung bình",
                    "High": "🟠 Đông đúc", "Jam": "🔴 Ùn tắc"
                }.get(x, x)
            )
            gps_weather = st.selectbox(
                "Điều kiện thời tiết",
                ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"], index=0,
                format_func=lambda x: {
                    "Sunny": "☀️ Nắng", "Cloudy": "☁️ Nhiều mây",
                    "Windy": "💨 Gió lớn", "Fog": "🌫️ Sương mù",
                    "Sandstorms": "🌪️ Bão cát", "Stormy": "⛈️ Mưa bão"
                }.get(x, x)
            )
        with cgps2:
            gps_vehicle = st.selectbox(
                "Phương tiện",
                ["motorcycle", "scooter", "electric_scooter", "bicycle"], index=0,
                format_func=lambda x: {
                    "motorcycle": "🏍️ Xe máy xăng",
                    "scooter": "🛵 Xe tay ga",
                    "electric_scooter": "⚡ Xe máy điện",
                    "bicycle": "🚲 Xe đạp"
                }.get(x, x)
            )
            gps_multi = st.selectbox(
                "Số đơn ghép", [0, 1, 2, 3], index=1,
                format_func=lambda x: "🎯 1 đơn duy nhất" if x == 0 else f"📦 Ghép {x} đơn"
            )
        with cgps3:
            gps_rating = st.slider("Điểm tài xế", 1.0, 5.0, 4.6, 0.1)
            gps_hour = st.slider("Giờ đặt hàng", 0, 23, 19)

        gps_submit = st.form_submit_button("🗺️ TÍNH KHOẢNG CÁCH & DỰ ĐOÁN", use_container_width=True)

    if gps_submit:
        calc_dist = calculate_haversine_distance(r_lat, r_lon, d_lat, d_lon)
        st.markdown(
            f'<div class="gps-dist-box">📏 Khoảng cách Haversine tính được: <b>{calc_dist:.2f} km</b>'
            f'&nbsp; (từ [{r_lat:.4f}, {r_lon:.4f}] → [{d_lat:.4f}, {d_lon:.4f}])</div>',
            unsafe_allow_html=True
        )

        gps_order_dict = {
            "Delivery_person_Age": 29,
            "Delivery_person_Ratings": gps_rating,
            "distance_km": calc_dist,
            "Weatherconditions": gps_weather,
            "Road_traffic_density": gps_traffic,
            "Vehicle_condition": 1,
            "Type_of_order": "Meal",
            "Type_of_vehicle": gps_vehicle,
            "multiple_deliveries": gps_multi,
            "Festival": "No",
            "City": "Metropolitian",
            "order_hour": gps_hour,
            "day_of_week": 3,
            "is_weekend": 0,
            "prep_time_min": 10.0
        }

        try:
            res_gps = predict_single(gps_order_dict)
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">⏱️ Kết quả dự đoán</div>
                <span class="result-number">{res_gps['predicted_minutes']}</span>
                <span class="result-unit">phút</span>
                <div class="result-range">
                    📊 Khoảng ước lượng: <b>{res_gps['min_estimated_minutes']} – {res_gps['max_estimated_minutes']} phút</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            traffic_vn = {"Low": "Thông thoáng", "Medium": "Trung bình", "High": "Đông đúc", "Jam": "Ùn tắc"}
            weather_vn = {
                "Sunny": "Nắng", "Cloudy": "Nhiều mây", "Windy": "Gió lớn",
                "Fog": "Sương mù", "Sandstorms": "Bão cát", "Stormy": "Mưa bão"
            }
            st.markdown(f"""
            <div style="margin-top: 14px;">
                <div style="font-size:13px; font-weight:600; color:#475569; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.4px;">
                    🔎 Các yếu tố chính ảnh hưởng:
                </div>
                <div class="factors-row">
                    <span class="factor-pill">📍 Khoảng cách: <b>{calc_dist:.2f} km</b></span>
                    <span class="factor-pill">🚦 Giao thông: <b>{traffic_vn.get(gps_traffic, gps_traffic)}</b></span>
                    <span class="factor-pill">⛅ Thời tiết: <b>{weather_vn.get(gps_weather, gps_weather)}</b></span>
                    <span class="factor-pill">📦 Đơn ghép: <b>{gps_multi} đơn khác</b></span>
                    <span class="factor-pill">⭐ Tài xế: <b>{gps_rating} sao</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")
