"""
src/predict.py
Module và giao diện dòng lệnh (CLI) thực hiện dự đoán thời gian giao hàng
sử dụng mô hình Random Forest đã được huấn luyện.
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Đảm bảo đường dẫn module luôn tìm thấy thư mục src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    ALL_MODEL_FEATURES,
    calculate_haversine_distance,
    clean_dataframe
)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "delivery_pipeline.joblib")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Nạp pipeline Random Forest từ file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại '{model_path}'. "
            f"Vui lòng chạy 'python src/train.py' trước để huấn luyện và lưu mô hình!"
        )
    return joblib.load(model_path)


def predict_batch(df: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH) -> np.ndarray:
    """
    Dự đoán thời gian giao hàng cho một DataFrame (ví dụ test.csv).
    Tự động áp dụng làm sạch, loại bỏ ID, Delivery_person_ID và tính toán đặc trưng.
    """
    pipeline = load_model(model_path)
    X_features = clean_dataframe(df, is_training=False)
    predictions = pipeline.predict(X_features)
    return np.round(predictions, 2)


def predict_single(order_info: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Dự đoán thời gian giao hàng cho 1 đơn hàng cụ thể từ dictionary.
    order_info có thể chứa khoảng cách 'distance_km' trực tiếp hoặc chứa toạ độ GPS.
    """
    pipeline = load_model(model_path)
    df_single = pd.DataFrame([order_info])

    # Nếu người dùng truyền toạ độ GPS mà chưa có distance_km, tự động tính Haversine
    if 'distance_km' not in df_single.columns or pd.isna(df_single['distance_km'].iloc[0]):
        if all(k in df_single.columns for k in ['Restaurant_latitude', 'Restaurant_longitude', 
                                                'Delivery_location_latitude', 'Delivery_location_longitude']):
            df_single['distance_km'] = calculate_haversine_distance(
                df_single['Restaurant_latitude'].iloc[0],
                df_single['Restaurant_longitude'].iloc[0],
                df_single['Delivery_location_latitude'].iloc[0],
                df_single['Delivery_location_longitude'].iloc[0]
            )

    X_features = clean_dataframe(df_single, is_training=False)
    pred_time = float(pipeline.predict(X_features)[0])

    # Dự đoán khoảng tin cậy ước lượng (± 1 MAE ~ 3.2 phút)
    confidence_margin = 3.3
    lower_bound = max(5.0, round(pred_time - confidence_margin, 1))
    upper_bound = round(pred_time + confidence_margin, 1)

    return {
        "predicted_minutes": round(pred_time, 1),
        "min_estimated_minutes": lower_bound,
        "max_estimated_minutes": upper_bound,
        "features_used": X_features.iloc[0].to_dict()
    }


# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def demo_sample_prediction():
    """Chạy thử dự đoán trên một mẫu đơn hàng giả định."""
    print("\n[+] Dang chay du doan mau don hang...")
    sample_order = {
        "Delivery_person_Age": 28,
        "Delivery_person_Ratings": 4.8,
        "distance_km": 6.5,
        "Weatherconditions": "Sunny",
        "Road_traffic_density": "Jam",
        "Vehicle_condition": 1,
        "Type_of_order": "Meal",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 1,
        "Festival": "No",
        "City": "Metropolitian",
        "order_hour": 19,
        "day_of_week": 4, # Thứ 6
        "is_weekend": 0,
        "prep_time_min": 10.0
    }

    result = predict_single(sample_order)
    print("=" * 55)
    print("THONG TIN DON HANG:")
    for k, v in sample_order.items():
        print(f"  * {k:25}: {v}")
    print("-" * 55)
    print(f"THOI GIAN GIAO HANG DU DOAN: {result['predicted_minutes']} phut")
    print(f"Khoang uoc luong (95% tin cay): {result['min_estimated_minutes']} - {result['max_estimated_minutes']} phut")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán thời gian giao đồ ăn với Random Forest Regressor.")
    parser.add_argument("--input", type=str, help="Đường dẫn file CSV dữ liệu đầu vào (ví dụ: test.csv)")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Đường dẫn lưu kết quả (nếu dùng --input)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Đường dẫn file mô hình .joblib")
    parser.add_argument("--sample", action="store_true", help="Chạy dự đoán thử trên một mẫu đơn hàng ví dụ")

    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Lỗi: Không tìm thấy file '{args.input}'")
            sys.exit(1)

        print(f"📥 Đang đọc dữ liệu từ: {args.input}")
        df_in = pd.read_csv(args.input)
        preds = predict_batch(df_in, model_path=args.model)

        df_out = df_in.copy()
        df_out["Predicted_Delivery_Time(min)"] = preds
        df_out.to_csv(args.output, index=False)
        print(f"✅ Đã dự đoán cho {len(preds):,} đơn hàng và lưu kết quả tại: {args.output}")
        print(df_out[["Predicted_Delivery_Time(min)"]].describe())

    elif args.sample or len(sys.argv) == 1:
        demo_sample_prediction()
