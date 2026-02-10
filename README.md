# Nhận diện Biển số xe(YOLOv8)

## Mô tả dự án
Dự án xây dựng hệ thống nhận diện biển số xe và nhận diện ký tự trên biển số từ ảnh tĩnh,
sử dụng YOLOv8 với hai mô hình đã được train sẵn:
- Mô hình phát hiện biển số
- Mô hình nhận diện ký tự trên biển số

Hệ thống được triển khai dưới dạng **demo chạy local**, phục vụ mục đích học tập.

---

## Công nghệ sử dụng
- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Streamlit

## Hướng dẫn chạy project
Bước 1: Clone source code

git clone <github_repository_url>
cd BIENSOXE

Bước 2: Cài đặt thư viện

pip install -r requirements.txt

Bước 3: Chạy ứng dụng

streamlit run main.py

Sau khi chạy, mở trình duyệt và truy cập:
http://localhost:8501

---
