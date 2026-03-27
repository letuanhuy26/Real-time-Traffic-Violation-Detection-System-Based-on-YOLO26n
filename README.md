# 🚦 Traffic Violation Detection System

> **Hệ thống phát hiện vi phạm giao thông thời gian thực** sử dụng YOLO Object Detection, CLAHE Image Enhancement.

---

## 📋 Tổng Quan Dự Án

| Thông tin | Chi tiết |
|-----------|----------|
| **Tên dự án** | Red Light Violation Detection using YOLO-based Object Detection |
| **Mục tiêu** | Phát hiện xe vượt đèn đỏ từ video camera giao thông |
| **Công nghệ AI** | YOLOv8/YOLO26, BotSORT Tracking |
| **Image Enhancement** | CLAHE (Double-pass) cho nhận diện biển số |
| **Dataset** | 5 classes: Vạch dừng, Đèn Xanh/Đỏ/Vàng, Đèn phụ |

---

## 🏗️ Kiến Trúc Hệ Thống

1. **YOLO Traffic Light Model**: Nhận diện trạng thái đèn giao thông và vạch dừng.
2. **YOLO Vehicle Detection**: Phát hiện xe và gán ID bằng thuật toán tracking (BotSORT).
3. **Rule-based Logic**: Xe vượt qua vạch dừng khi đèn đỏ/vàng → Xác định Vi Phạm.
4. **License Plate Detection**: Cắt ảnh xe vi phạm, nhận diện biển số và làm rõ nét bằng Double CLAHE.

---

## 📁 Cấu Trúc Thư Mục

Phiên bản GitHub này được tối giản hóa, chỉ chứa Source Code cốt lõi và dữ liệu mẫu.

```text
violation_detection/
├── Code/                                 # Toàn bộ source code dự án
│   ├── detection/                        # Pipeline phát hiện vi phạm chính (Final_RUN.py)
│   ├── data_processing/                  # Các script xử lý, gán nhãn, dọn dẹp data
│   ├── validation/                       # Script kiểm tra độ chính xác của model/data
│   └── training/                         # Script training model
│
├── data_samples/                         # Dữ liệu hình ảnh và nhãn mẫu (để tham khảo format)
│   ├── images/                           
│   └── labels/                           
│
├── Model/weights/                        # Placeholder cho file model nhận diện đèn giao thông (.pt)
├── LP_v26/model/plate_model/weights/     # Placeholder cho file model nhận diện biển số (.pt)
│
├── dataset_samples.png                   # Hình ảnh mẫu mô tả dataset
├── .gitignore                            # Cấu hình bỏ qua các file không cần thiết trên Git
└── README.md                             # File tài liệu này
```

> **Lưu ý:** Các file mô hình đã huấn luyện (như `best.pt`) và toàn bộ Dataset đầy đủ đã được loại bỏ khỏi repository này để đảm bảo dung lượng nhẹ nhất khi clone/push. Vui lòng đặt các file weights vào thư mục tương ứng trước khi chạy code.

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### Yêu Cầu Cài Đặt
- Python 3.9+
- Khuyên dùng GPU hỗ trợ CUDA để chạy Real-time.

1. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install ultralytics opencv-python numpy tqdm albumentations pillow easyocr
   ```

2. **Chạy Phát Hiện Vi Phạm (offline từ video):**
   Mở file `Code/detection/Final_RUN.py` để cấu hình đường dẫn mạng nơ-ron (weights) và đường dẫn Video đầu vào, sau đó chạy:
   ```bash
   cd Code/detection
   python Final_RUN.py
   ```

3. **Huấn Luyện Model Lại Từ Đầu:**
   Chỉnh sửa config trong `Code/train.py` và chạy:
   ```bash
   cd Code
   python train.py
   ```

---

## 👥 Đóng Góp
- **Pipeline chính** nằm tại thư mục `Code/detection/`.
- Các file rác hoặc ảnh/video dung lượng lớn xin vui lòng cấu hình bằng `.gitignore` trước khi push.
