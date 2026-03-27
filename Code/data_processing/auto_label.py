import os
import shutil
from ultralytics import YOLO

# ========================================================
# ĐƯỜNG DẪN CỦA BẠN 
# ========================================================
MODEL_PATH = r"D:\KY_4\DAP\Dap391\Project\Sources(new)\Sources\Model\traffic_light_modelYOLO26n_new\weights\best.pt"
IMAGE_DIR = r"C:\Users\Admin\Downloads\Image_Bonus\Image_Bonus"
OUTPUT_DIR = r'D:\KY_4\DAP\Dap391' 
# ========================================================

# Vẫn giữ nguyên 5 class để đưa vào Make Sense (Đảm bảo ID 1, 2, 3, 4 không bị xô lệch)
CLASS_NAMES = ['stopline', 'green light', 'red light', 'yellow light','sub light']

def prepare_makesense_data():
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy model tại:\n{MODEL_PATH}")
        return

    labels_output_dir = os.path.join(OUTPUT_DIR, 'labels_Image_Bonus')
    os.makedirs(labels_output_dir, exist_ok=True)

    # 1. Tải model
    print("Đang tải model best.pt...")
    model = YOLO(MODEL_PATH)

    # 2. Chạy Auto-label (BỎ QUA CLASS 0)
    print("Đang tự động dán nhãn (CHỈ LẤY stopline, red, green, yellow)...")
    model.predict(
        source=IMAGE_DIR,
        conf=0.5,           
        iou=0.45,            
        classes=[0,1, 2, 3, 4],
        save=False,          
        save_txt=True,       
        save_conf=False,     
        project=OUTPUT_DIR,  
        name='temp_predict', 
        exist_ok=True        
    )

    # 3. Chuyển file .txt về đúng chỗ
    temp_labels_dir = os.path.join(OUTPUT_DIR, 'temp_predict', 'labels')
    if os.path.exists(temp_labels_dir):
        for file_name in os.listdir(temp_labels_dir):
            shutil.move(os.path.join(temp_labels_dir, file_name), os.path.join(labels_output_dir, file_name))
    
    # Xóa thư mục tạm 
    shutil.rmtree(os.path.join(OUTPUT_DIR, 'temp_predict'), ignore_errors=True)

    # 4. Tự động tạo file labels.txt cho Make Sense
    labels_file_path = os.path.join(labels_output_dir, 'labels.txt')
    with open(labels_file_path, 'w') as f:
        for cls in CLASS_NAMES:
            f.write(f"{cls}\n")

    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH! ĐÃ BỎ QUA CLASS 'traffic light' (ID 0)")
    print(f"File nhãn (.txt) và file 'labels.txt' đã được lưu tại:\n   {labels_output_dir}")
    print("="*60)

if __name__ == "__main__":
    prepare_makesense_data()