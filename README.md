# DeliTime – Dự đoán thời gian giao đồ ăn bằng Machine Learning

## 1. Giới thiệu

**DeliTime** là dự án ứng dụng Machine Learning để dự đoán thời gian giao đồ ăn dựa trên các thông tin của đơn hàng, người giao hàng, phương tiện, thời tiết, tình trạng giao thông và các đặc trưng liên quan.

Dự án sử dụng **Random Forest Regression** làm mô hình dự đoán chính và **Streamlit** để xây dựng giao diện web tương tác.

### Chức năng chính

- Nhập thông tin đơn hàng để dự đoán thời gian giao hàng.
- Hỗ trợ dự đoán trên dữ liệu đơn hàng từ file CSV.
- Tiền xử lý dữ liệu và trích xuất đặc trưng trước khi huấn luyện.
- Đánh giá mô hình bằng MAE, RMSE và R².
- Trực quan hóa mức độ quan trọng của các đặc trưng và kết quả dự đoán.
- Lưu mô hình đã huấn luyện để sử dụng cho các lần dự đoán tiếp theo.

## 2. Công nghệ sử dụng

- Python
- Anaconda / Conda
- Pandas, NumPy
- Scikit-learn
- Random Forest Regression
- Joblib
- Matplotlib, Seaborn
- Streamlit
- Git và GitHub
- Visual Studio Code

## 3. Cấu trúc dự án

```text
DeliTime/
├── app.py
├── requirements.txt
├── train.csv
├── test.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── model_metadata.json
├── plots/
├── README.md
├── CHANGELOG.md
├── HUONG_DAN_CHAY_PROJECT.md
├── HUONG_DAN_TRAIN.md
└── HUONG_DAN_SU_DUNG_WEB.md
```

**Lưu ý:** File `models/delivery_pipeline.joblib` được tạo sau khi huấn luyện mô hình và không được lưu trên GitHub. Thư mục `models/` và `plots/` sẽ được chương trình tạo khi cần thiết.

## 4. Yêu cầu cài đặt

Trước khi chạy dự án, cần chuẩn bị:

- Visual Studio Code.
- Git.
- Anaconda.
- Python 3.11 thông qua môi trường Conda.
- Kết nối Internet để tải mã nguồn và cài đặt các thư viện.

## 5. Hướng dẫn cài đặt và chạy dự án

### Bước 1: Tải mã nguồn từ GitHub

Mở Visual Studio Code, chọn **Terminal → New Terminal**.

Chạy lệnh:

```bash
git clone https://github.com/Tuananhcncncc/DeliTime.git
cd DeliTime
```

Sau khi tải xong, hãy bảo đảm Terminal đang ở thư mục gốc `DeliTime`, nơi chứa file `app.py` và `requirements.txt`.

Nếu bạn đã có mã nguồn trên máy, chỉ cần mở thư mục dự án trong VS Code và chuyển Terminal đến thư mục gốc; không cần clone lại.

### Bước 2: Tạo và kích hoạt môi trường Conda

Tạo môi trường Python riêng cho dự án:

```bash
conda create -n delitime python=3.11 -y
```

Kích hoạt môi trường:

```bash
conda activate delitime
```

Môi trường `delitime` giúp các thư viện của dự án không ảnh hưởng đến những project Python khác trên máy.

Nếu đã tạo môi trường `delitime` trước đó, bạn không cần tạo lại; chỉ cần kích hoạt nó.

### Bước 3: Cấu hình Python trong VS Code

Trong VS Code:

1. Cài extension **Python** của Microsoft nếu chưa có.
2. Mở Command Palette bằng `Ctrl + Shift + P` trên Windows hoặc `Cmd + Shift + P` trên macOS.
3. Chọn **Python: Select Interpreter**.
4. Chọn Python thuộc môi trường Conda `delitime`.

Nếu không thấy môi trường, hãy kiểm tra Conda đã được cài đặt và môi trường đã được tạo thành công. Sau khi chọn interpreter, bạn có thể tiếp tục sử dụng Terminal tích hợp của VS Code.

### Bước 4: Cài đặt các thư viện cần thiết

Trong Terminal, tại thư mục gốc dự án và sau khi kích hoạt môi trường `delitime`, chạy:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Bước 5: Huấn luyện mô hình Machine Learning

Dự án không đưa file `delivery_pipeline.joblib` lên GitHub vì file này được tạo từ quá trình huấn luyện và có dung lượng lớn.

Sau khi cài đặt thư viện, chạy lệnh sau trong Terminal của VS Code:

```bash
python src/train.py
```

Chương trình sẽ:

1. Đọc dữ liệu huấn luyện từ `train.csv`.
2. Tiền xử lý dữ liệu và trích xuất đặc trưng.
3. Chia dữ liệu thành tập huấn luyện và tập kiểm định.
4. Huấn luyện mô hình Random Forest Regression.
5. Đánh giá kết quả bằng MAE, RMSE và R².
6. Lưu mô hình tại `models/delivery_pipeline.joblib`.
7. Lưu thông tin mô hình tại `models/model_metadata.json`.
8. Xuất các biểu đồ đánh giá vào thư mục `plots/`.

Khi quá trình huấn luyện hoàn tất thành công, file mô hình sẽ được tạo trên máy của bạn.

**Lưu ý:** Cần chạy lệnh huấn luyện lần đầu trước khi khởi chạy ứng dụng nếu máy chưa có file mô hình. Khi mô hình đã được tạo, không cần huấn luyện lại mỗi lần mở website, trừ khi muốn tạo mô hình mới.

### Bước 6: Khởi chạy website

Sau khi huấn luyện mô hình thành công, chạy:

```bash
python -m streamlit run app.py
```

Streamlit sẽ khởi chạy ứng dụng web. Mở địa chỉ sau trên trình duyệt nếu trình duyệt không tự mở:

http://localhost:8501
