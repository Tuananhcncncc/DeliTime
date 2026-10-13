"""
src/train.py
Script huấn luyện mô hình Random Forest Regression cho bài toán Dự đoán thời gian giao đồ ăn.
Thực hiện huấn luyện, đánh giá mô hình (MAE, RMSE, R2), xuất biểu đồ và lưu trữ pipeline.
"""

import argparse
import json
import os
import sys
import time

# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_dataframe,
    create_preprocessor
)


def train_model(
    train_path: str = "train.csv",
    models_dir: str = "models",
    plots_dir: str = "plots",
    n_estimators: int = 120,
    max_depth: int = 18,
    min_samples_split: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
):
    print("=" * 65)
    print(">>> BAT DAU QUA TRINH HUAN LUYEN RANDOM FOREST REGRESSION <<<")
    print(f"File du lieu huan luyen: {train_path}")
    print(f"Tham so: n_estimators={n_estimators}, max_depth={max_depth}, test_size={test_size}")
    print("=" * 65)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Đọc dữ liệu
    start_time = time.time()
    print("\n[1/5] Dang nap du lieu...")
    raw_df = pd.read_csv(train_path)
    print(f" -> Kich thuoc du lieu goc: {raw_df.shape[0]:,} dong, {raw_df.shape[1]} cot")

    # 2. Tiền xử lý & Trích xuất đặc trưng
    print("\n[2/5] Dang tien xu ly du lieu va trich xuat dac trung...")
    X, y = clean_dataframe(raw_df, is_training=True)

    # Lọc bỏ các mẫu thiếu biến mục tiêu (nếu có)
    valid_idx = y.notnull()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    print(f" -> Da loai bo 'ID' va 'Delivery_person_ID' theo dung yeu cau de tai.")
    print(f" -> So luong dac trung dua vao mo hinh: {len(X.columns)}")
    print(f"    + Bien so ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"    + Bien phan loai ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f" -> So mau hop le sau tien xu ly: {len(X):,}")

    # 3. Phân chia tập Train / Validation
    print(f"\n[3/5] Phan chia tap du lieu (Train: {1-test_size:.0%}, Validation: {test_size:.0%})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f" -> Tap huan luyen (Train): {len(X_train):,} mau")
    print(f" -> Tap kiem dinh (Val):   {len(X_val):,} mau")

    # 4. Xây dựng Pipeline & Huấn luyện
    print("\n[4/5] Dang huan luyen Random Forest Regressor...")
    preprocessor = create_preprocessor()
    rf_regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_regressor)
    ])

    fit_start = time.time()
    full_pipeline.fit(X_train, y_train)
    fit_duration = time.time() - fit_start
    print(f" -> Huan luyen hoan tat trong {fit_duration:.2f} giay!")

    # 5. Đánh giá mô hình trên tập Validation
    print("\n[5/5] Dang danh gia ket qua mo hinh...")
    y_pred_val = full_pipeline.predict(X_val)
    y_pred_train = full_pipeline.predict(X_train)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_pred_val)

    print("\n" + "=" * 50)
    print("KET QUA DANH GIA MO HINH (EVALUATION METRICS):")
    print("=" * 50)
    print(f"  * Train MAE:               {train_mae:.4f} phut")
    print(f"  * Validation MAE:          {val_mae:.4f} phut")
    print(f"  * Validation RMSE:         {val_rmse:.4f} phut")
    print(f"  * Validation R2 Score:     {val_r2:.4f} ({val_r2 * 100:.2f}%)")
    print("=" * 50)

    # 6. Lưu Pipeline và Metadata
    model_path = os.path.join(models_dir, "delivery_pipeline.joblib")
    joblib.dump(full_pipeline, model_path)
    print(f"\n[+] Da luu model pipeline tai: {model_path}")

    # Lấy tên các đặc trưng sau One-Hot Encoding để trích xuất Feature Importance
    try:
        cat_encoder = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        ohe_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        feature_names = NUMERIC_FEATURES + ohe_cols
        importances = full_pipeline.named_steps['regressor'].feature_importances_
        feature_importance_dict = dict(zip(feature_names, [float(x) for x in importances]))
        # Sắp xếp top 15 đặc trưng quan trọng nhất
        sorted_fi = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    except Exception as e:
        sorted_fi = []
        print(f"Luu y: Khong the xuat feature importances chi tiet: {e}")

    metadata = {
        "model_type": "RandomForestRegressor",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": test_size,
        "random_state": random_state,
        "metrics": {
            "validation_mae_minutes": round(float(val_mae), 4),
            "validation_rmse_minutes": round(float(val_rmse), 4),
            "validation_r2_score": round(float(val_r2), 4),
            "train_mae_minutes": round(float(train_mae), 4),
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all": ALL_MODEL_FEATURES
        },
        "top_features": sorted_fi,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "training_time_seconds": round(float(fit_duration), 2)
    }

    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"[+] Da luu thong tin cau hinh va metrics tai: {meta_path}")

    # 7. Vẽ biểu đồ đánh giá
    print("\n[+] Dang tao cac bieu do truc quan hoa...")
    # Biểu đồ 1: Feature Importances
    if sorted_fi:
        plt.figure(figsize=(10, 6))
        fi_df = pd.DataFrame(sorted_fi, columns=['Feature', 'Importance']).iloc[::-1]
        plt.barh(fi_df['Feature'], fi_df['Importance'], color='#3B82F6')
        plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=11)
        plt.tight_layout()
        fi_plot_path = os.path.join(plots_dir, "feature_importance.png")
        plt.savefig(fi_plot_path, dpi=200)
        plt.close()
        print(f" -> Bieu do Feature Importance: {fi_plot_path}")

    # Biểu đồ 2: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    # Lấy mẫu 2,000 điểm ngẫu nhiên để biểu đồ không bị quá dày
    sample_indices = np.random.RandomState(42).choice(len(y_val), size=min(2000, len(y_val)), replace=False)
    y_val_samp = y_val.iloc[sample_indices]
    y_pred_samp = y_pred_val[sample_indices]

    plt.scatter(y_val_samp, y_pred_samp, alpha=0.35, color='#2563EB', edgecolors='none', s=25)
    min_val = min(y_val_samp.min(), y_pred_samp.min())
    max_val = max(y_val_samp.max(), y_pred_samp.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction (y = x)')
    plt.title(f'Actual vs Predicted (R2 = {val_r2:.3f}, MAE = {val_mae:.2f}m)', fontsize=13, fontweight='bold')
    plt.xlabel('Actual Time (min)', fontsize=11)
    plt.ylabel('Predicted Time (min)', fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    act_plot_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    plt.savefig(act_plot_path, dpi=200)
    plt.close()
    print(f" -> Bieu do Actual vs Predicted: {act_plot_path}")

    # Biểu đồ 3: Residuals Distribution
    residuals = y_val - y_pred_val
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, kde=True, bins=40, color='#10B981')
    plt.axvline(0, color='red', linestyle='--', lw=1.5)
    plt.title('Residuals Distribution (Actual - Predicted)', fontsize=13, fontweight='bold')
    plt.xlabel('Residuals (min)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    res_plot_path = os.path.join(plots_dir, "residuals_distribution.png")
    plt.savefig(res_plot_path, dpi=200)
    plt.close()
    print(f" -> Bieu do Residuals: {res_plot_path}")

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] HOAN TAT HUAN LUYEN TOAN BO TRONG {total_time:.2f} GIAY!")
    return full_pipeline, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Random Forest dự đoán thời gian giao đồ ăn.")
    parser.add_argument("--train_path", type=str, default="train.csv", help="Đường dẫn file train.csv")
    parser.add_argument("--n_estimators", type=int, default=120, help="Số lượng cây quyết định (mặc định: 120)")
    parser.add_argument("--max_depth", type=int, default=18, help="Độ sâu tối đa của cây (mặc định: 18)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Tỷ lệ tập kiểm định (mặc định: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Seed ngẫu nhiên (mặc định: 42)")

    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state
    )
