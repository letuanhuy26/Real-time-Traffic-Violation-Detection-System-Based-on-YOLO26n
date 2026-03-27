import os

# 1. Bạn copy đường dẫn của thư mục trong ảnh vào đây
DATASET_ROOT = r"D:\KY_4\DAP\Dap391\Project\Sources(new)\Sources"

def remove_and_shift_labels():
    # Trỏ thẳng vào thư mục 'labels' trong ảnh của bạn
    labels_dir = os.path.join(DATASET_ROOT, 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"Lỗi: Không tìm thấy thư mục {labels_dir}")
        return

    print("Bắt đầu quét và dọn dẹp file nhãn...")
    count_files = 0

    # os.walk giúp quét sạch mọi file .txt bên trong (dù có thư mục con hay không)
    for root, dirs, files in os.walk(labels_dir):
        for filename in files:
            # Bỏ qua nếu không phải file txt hoặc là file classes.txt
            if not filename.endswith('.txt') or filename in ['classes.txt', 'labels.txt']:
                continue
                
            filepath = os.path.join(root, filename)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                
                # BỎ QUA CLASS 0 (Traffic light)
                if class_id == 0:
                    continue
                
                # Các class còn lại: giảm ID đi 1
                new_class_id = class_id - 1
                
                # Ghép lại dòng
                new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                new_lines.append(new_line)
                
            # Ghi đè lại file
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            
            count_files += 1

    print(f"Hoàn tất! Đã dọn dẹp và cập nhật lại {count_files} file labels.")

if __name__ == "__main__":
    # KHUYẾN CÁO: Hãy copy dự phòng thư mục 'labels' ra một nơi khác trước khi chạy để phòng rủi ro.
    remove_and_shift_labels()