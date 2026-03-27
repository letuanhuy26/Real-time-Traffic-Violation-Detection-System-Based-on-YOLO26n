import torch
from ultralytics import YOLO
import cv2
import os

# 1. Cấu hình đường dẫn
model_path = r"D:\KY_4\DAP\Dap391\Project\LP_v26\model\char_train\weights\best.pt"
video_input_path = r"D:\KY_4\DAP\Dap391\Project\Source\Video\video_unlabeled\GOVAP4K.mp4"
output_dir = r"D:\KY_4\DAP\Dap391\Project\output"

# 2. Load model
model = YOLO(model_path)

# 3. Mở video
cap = cv2.VideoCapture(video_input_path)

if not cap.isOpened():
    print("Không mở được video!")
    exit()

print("Bắt đầu xử lý video...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 4. Chạy dự đoán bình thường (không lọc)
    results = model.predict(
        frame,
        conf=0.25,        # ngưỡng mặc định an toàn
        device=0,        # GPU (đổi thành 'cpu' nếu lỗi)
        verbose=False
    )

    # 5. Vẽ toàn bộ detection
    annotated_frame = results[0].plot()

    # 6. Hiển thị
    small_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
    cv2.imshow("YOLO Detection", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Đã xử lý xong video.")
