# Nhận diện Biển số xe Việt Nam (YOLOv8)

## Mô tả dự án
Hệ thống nhận diện biển số xe và đọc ký tự quang học (OCR) cho biển số 1 dòng và 2 dòng tại Việt Nam từ ảnh tĩnh, sử dụng kiến trúc **YOLOv8 2 giai đoạn (Two-stage Cascade Pipeline)**:
1. **Giai đoạn 1 (`weights/plate_best.pt`):** Phát hiện và cắt vùng chứa biển số xe từ ảnh gốc (`conf = 0.40`, `IoU = 0.50`).
2. **Giai đoạn 2 (`weights/chars_best.pt`):** Nhận diện từng ký tự bên trong vùng biển số đã cắt (`imgsz = 640`, `conf = 0.35`, `IoU = 0.50`).
3. **Thuật toán hậu xử lý không gian (Spatial Sorting):** Tự động phân tách biển số 1 dòng / 2 dòng theo ngưỡng trung bình tung độ tâm `(y_min + y_max) / 2`, sắp xếp các ký tự trên từng dòng từ trái sang phải theo hoành độ tâm `x_center` để tái tạo chuỗi biển số hoàn chỉnh và ánh xạ ngược khung giới hạn lên ảnh gốc.

---

## Bộ dữ liệu huấn luyện (Dataset)
Mô hình được huấn luyện và đánh giá trên tổng cộng **4,824** ảnh gán nhãn qua Roboflow:
- **Mô hình phát hiện biển số:** **1,005** ảnh biển số xe Việt Nam.
- **Mô hình nhận diện ký tự (OCR):** **3,819** ảnh ký tự thuộc **31 lớp (classes)** (`0–9` và các chữ cái biển số chuẩn Việt Nam).

---

## Công nghệ sử dụng
- **Ngôn ngữ:** Python
- **Deep Learning & Thị giác máy tính:** YOLOv8 (Ultralytics), PyTorch, OpenCV, NumPy, Pillow
- **Giao diện Web:** Streamlit

---

## Cấu trúc thư mục
```text
license-plate-recognition/
├── weights/
│   ├── plate_best.pt    # Trọng số YOLOv8 phát hiện vùng biển số
│   └── chars_best.pt    # Trọng số YOLOv8 nhận diện 31 lớp ký tự
├── main.py              # Ứng dụng Streamlit & thuật toán suy luận 2 giai đoạn
├── requirements.txt     # Danh sách thư viện phụ thuộc
└── README.md
```

---

## Hướng dẫn cài đặt & Chạy ứng dụng

### Bước 1: Clone source code
```bash
git clone https://github.com/thinhdang1104/license-plate-recognition.git
cd license-plate-recognition
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng Streamlit
```bash
streamlit run main.py
```
Sau khi khởi chạy, mở trình duyệt và truy cập: **http://localhost:8501**
