# Student Score Prediction

Đồ án môn Khai phá dữ liệu: **Xây dựng ứng dụng web dự đoán điểm số học sinh bằng các thuật toán hồi quy**.

## 1. Chức năng chính

- Đọc dữ liệu học sinh từ file CSV.
- Tiền xử lý dữ liệu: xử lý thiếu, mã hóa dữ liệu chữ, chuẩn hóa dữ liệu số.
- Huấn luyện và so sánh 3 mô hình hồi quy:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Đánh giá bằng MAE, MSE, RMSE, R².
- Lưu mô hình tốt nhất thành `models/best_model.pkl`.
- Xây dựng web Streamlit để nhập thông tin sinh viên và dự đoán điểm.

## 2. Cấu trúc thư mục

```text
StudentScorePrediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── student_performance.csv
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── model_metrics.csv
│   ├── correlation_chart.png
│   └── correlation_heatmap.png
│
└── static/
    └── style.css
```

## 3. Cài đặt

Mở terminal tại thư mục project và chạy:

```bash
pip install -r requirements.txt
```

## 4. Huấn luyện mô hình

```bash
python train_model.py
```

Sau khi chạy xong, chương trình sẽ tạo:

```text
models/best_model.pkl
models/feature_info.json
reports/model_metrics.csv
reports/correlation_chart.png
reports/correlation_heatmap.png
```

## 5. Chạy ứng dụng web

```bash
streamlit run app.py
```

Streamlit sẽ tự mở trình duyệt tại:

```text
http://localhost:8501
```

## 6. Dataset

File dữ liệu đặt tại `data/student_performance.csv`, gồm các cột:

```text
study_hours            - Số giờ tự học mỗi tuần
assignments_completed  - Số bài tập đã hoàn thành
attendance_rate        - Tỷ lệ chuyên cần (%)
midterm_score          - Điểm giữa kỳ (thang 100)
final_score            - Điểm cuối kỳ (thang 100)  [BIẾN MỤC TIÊU]
```

Dataset được sinh bằng script `make_dataset.py` (mô phỏng có tương quan thực tế).
Tạo lại bằng lệnh:

```bash
python make_dataset.py
```

Nếu thay bằng dataset Kaggle khác, hãy giữ cột mục tiêu là một trong các tên đã khai báo trong `TARGET_CANDIDATES` của `train_model.py` (ví dụ `final_score`, `Exam Score`, `G3`...).

## 7. Ý nghĩa các chỉ số đánh giá

- MAE: Sai số tuyệt đối trung bình, càng thấp càng tốt.
- MSE: Sai số bình phương trung bình, càng thấp càng tốt.
- RMSE: Căn bậc hai của MSE, càng thấp càng tốt.
- R²: Mức độ mô hình giải thích biến mục tiêu, càng gần 1 càng tốt.

## 8. Ghi chú nộp bài

Trước khi nộp, nên chạy đủ hai lệnh:

```bash
python train_model.py
streamlit run app.py
```

Sau đó chụp ảnh:
- Màn hình terminal kết quả đánh giá mô hình.
- Trang nhập thông tin sinh viên.
- Trang kết quả dự đoán.
- File `reports/model_metrics.csv`.
- Biểu đồ `reports/correlation_chart.png` và `reports/correlation_heatmap.png`.

## Cập nhật giao diện

Phiên bản này đã hỗ trợ:
- Chuyển đổi giao diện Tiếng Việt / English.
- Hiển thị điểm dự đoán theo hệ 10 điểm.
- Vẫn giữ thêm điểm theo thang 100 để đối chiếu với dataset gốc.

## Cập nhật hệ 10 hoàn toàn

Phiên bản này dùng giao diện hệ 10:
- Người dùng nhập `Điểm trước đó` theo thang 10.
- Ứng dụng tự quy đổi điểm trước đó sang thang 100 ở bên trong để tương thích với dataset/model gốc.
- Kết quả dự đoán chỉ hiển thị theo thang 10.
