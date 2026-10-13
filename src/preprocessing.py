"""
src/preprocessing.py
Module tiền xử lý dữ liệu và trích xuất đặc trưng cho bài toán Dự đoán thời gian giao đồ ăn.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Danh sách các đặc trưng được sử dụng trong mô hình
NUMERIC_FEATURES = [
    'Delivery_person_Age',
    'Delivery_person_Ratings',
    'Vehicle_condition',
    'multiple_deliveries',
    'distance_km',
    'order_hour',
    'prep_time_min',
    'day_of_week',
    'is_weekend'
]

CATEGORICAL_FEATURES = [
    'Weatherconditions',
    'Road_traffic_density',
    'Type_of_order',
    'Type_of_vehicle',
    'Festival',
    'City'
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách đường tròn lớn giữa 2 toạ độ (Haversine formula) tính bằng kilômét.
    Xử lý toạ độ âm bằng trị tuyệt đối. Toạ độ gần 0 (< 1) được coi là dữ liệu thiếu (trả về NaN).
    """
    lat1 = np.abs(np.asarray(lat1, dtype=float))
    lon1 = np.abs(np.asarray(lon1, dtype=float))
    lat2 = np.abs(np.asarray(lat2, dtype=float))
    lon2 = np.abs(np.asarray(lon2, dtype=float))

    valid = (lat1 > 1.0) & (lon1 > 1.0) & (lat2 > 1.0) & (lon2 > 1.0)
    
    # Khởi tạo kết quả NaN
    distances = np.full_like(lat1, fill_value=np.nan, dtype=float)

    if np.any(valid):
        r_lat1 = np.radians(lat1[valid])
        r_lon1 = np.radians(lon1[valid])
        r_lat2 = np.radians(lat2[valid])
        r_lon2 = np.radians(lon2[valid])

        dlat = r_lat2 - r_lat1
        dlon = r_lon2 - r_lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0) ** 2
        # Giới hạn a trong khoảng [0, 1] để tránh lỗi căn bậc hai do sai số float
        a = np.clip(a, 0.0, 1.0)
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371.0 * c
        distances[valid] = km

    if np.isscalar(lat1) or distances.ndim == 0:
        return float(distances)
    return distances


def clean_dataframe(df: pd.DataFrame, is_training: bool = True):
    """
    Làm sạch dữ liệu và trích xuất đặc trưng từ DataFrame thô.
    Yêu cầu:
    - Loại bỏ thuộc tính 'ID' và 'Delivery_person_ID' theo yêu cầu đề tài.
    - Tính khoảng cách giao hàng 'distance_km'.
    - Trích xuất thông tin thời gian: 'day_of_week', 'is_weekend', 'order_hour', 'prep_time_min'.
    - Chuẩn hóa chuỗi (loại bỏ tiền tố 'conditions ', khoảng trắng thừa, giá trị 'NaN').
    """
    df = df.copy()

    # 1. Trích xuất biến mục tiêu y (nếu có cột Time_taken(min))
    y = None
    if 'Time_taken(min)' in df.columns:
        # Giá trị có dạng '(min) 24', trích xuất phần số
        extracted = df['Time_taken(min)'].astype(str).str.extract(r'(\d+)')
        y = extracted[0].astype(float)

    # 2. Loại bỏ các thuộc tính không cần thiết: ID, Delivery_person_ID, và Time_taken(min) khỏi X
    drop_cols = [c for c in ['ID', 'id', 'Delivery_person_ID', 'Time_taken(min)'] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # 3. Tính khoảng cách Haversine (distance_km)
    if 'distance_km' not in X.columns:
        if all(col in X.columns for col in ['Restaurant_latitude', 'Restaurant_longitude', 
                                            'Delivery_location_latitude', 'Delivery_location_longitude']):
            X['distance_km'] = calculate_haversine_distance(
                X['Restaurant_latitude'],
                X['Restaurant_longitude'],
                X['Delivery_location_latitude'],
                X['Delivery_location_longitude']
            )
        else:
            X['distance_km'] = np.nan

    # 4. Làm sạch các trường dạng chuỗi / phân loại
    cat_columns = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 
                   'Type_of_vehicle', 'Festival', 'City']
    for col in cat_columns:
        if col in X.columns:
            cleaned = (
                X[col]
                .astype(str)
                .str.strip()
                .str.replace('conditions ', '', regex=False)
                .replace({'NaN': np.nan, 'nan': np.nan, '': np.nan})
            )
            X[col] = cleaned
        else:
            X[col] = np.nan

    # 5. Chuyển đổi và làm sạch các trường số
    numeric_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 
                       'multiple_deliveries', 'Vehicle_condition']
    for col in numeric_columns:
        if col in X.columns:
            cleaned = (
                X[col]
                .astype(str)
                .str.strip()
                .replace({'NaN': np.nan, 'nan': np.nan, '': np.nan})
            )
            X[col] = pd.to_numeric(cleaned, errors='coerce')
        else:
            X[col] = np.nan

    # 6. Trích xuất đặc trưng ngày từ Order_Date
    if 'Order_Date' in X.columns:
        order_dates = pd.to_datetime(
            X['Order_Date'].astype(str).str.strip(),
            format='%d-%m-%Y',
            errors='coerce'
        )
        X['day_of_week'] = order_dates.dt.dayofweek
        X['is_weekend'] = (order_dates.dt.dayofweek >= 5).astype(float)
    else:
        if 'day_of_week' not in X.columns:
            X['day_of_week'] = np.nan
        if 'is_weekend' not in X.columns:
            X['is_weekend'] = np.nan

    # 7. Trích xuất giờ đặt hàng (order_hour) từ Time_Orderd
    if 'Time_Orderd' in X.columns:
        order_times = pd.to_timedelta(
            X['Time_Orderd'].astype(str).str.strip().replace({'NaN': np.nan, 'nan': np.nan}),
            errors='coerce'
        )
        X['order_hour'] = order_times.dt.components.hours.astype(float)
    else:
        if 'order_hour' not in X.columns:
            X['order_hour'] = np.nan

    # 8. Tính thời gian chuẩn bị món (prep_time_min) = Time_Order_picked - Time_Orderd
    if 'prep_time_min' not in X.columns:
        if 'Time_Orderd' in X.columns and 'Time_Order_picked' in X.columns:
            to = pd.to_timedelta(
                X['Time_Orderd'].astype(str).str.strip().replace({'NaN': np.nan, 'nan': np.nan}),
                errors='coerce'
            )
            tp = pd.to_timedelta(
                X['Time_Order_picked'].astype(str).str.strip().replace({'NaN': np.nan, 'nan': np.nan}),
                errors='coerce'
            )
            diff_min = (tp - to).dt.total_seconds() / 60.0
            # Nếu qua đêm (thời gian âm), cộng thêm 24 giờ = 1440 phút
            diff_min = diff_min.apply(lambda m: m + 1440.0 if pd.notnull(m) and m < 0 else m)
            X['prep_time_min'] = diff_min
        else:
            X['prep_time_min'] = np.nan

    # 9. Chỉ giữ lại đúng các đặc trưng phục vụ huấn luyện / dự đoán
    # Loại bỏ các cột thô gốc (tọa độ thô, chuỗi ngày giờ thô)
    final_cols = [c for c in ALL_MODEL_FEATURES if c in X.columns]
    X_features = X[final_cols].copy()

    # Đảm bảo tất cả các cột cần thiết đều hiện diện
    for col in ALL_MODEL_FEATURES:
        if col not in X_features.columns:
            X_features[col] = np.nan

    X_features = X_features[ALL_MODEL_FEATURES]

    if is_training:
        return X_features, y
    return X_features


def create_preprocessor() -> ColumnTransformer:
    """
    Tạo ColumnTransformer tiền xử lý chuẩn hoá Scikit-Learn:
    - Biến số: SimpleImputer(median) + StandardScaler
    - Biến phân loại: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERIC_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor
