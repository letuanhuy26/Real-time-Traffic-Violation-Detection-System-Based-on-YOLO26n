import os
import shutil

def delete_labels(base_dir, start_idx, end_idx):
    """
    Xóa các tệp label nằm trong khoảng [start_idx, end_idx] (bao gồm cả end)
    trong các thư mục con 'train', 'val', 'test' của thư mục base_dir.
    
    Args:
        base_dir: Đường dẫn đến thư mục chứa các thư mục con (ví dụ: 'labels').
                  Bên trong thư mục này dự kiến có chứa 'train', 'val', 'test'.
        start_idx: Số bắt đầu (bao gồm).
        end_idx: Số kết thúc (bao gồm).
    """

    # Danh sách các thư mục con cần quét
    sub_dirs = ['Train', 'Validate', 'Test']

    # Tạo một set các số cần xóa để tra cứu nhanh
    targets_to_delete = {str(i) for i in range(start_idx, end_idx + 1)}

    deleted_count = 0

    # Lặp qua từng thư mục con
    for sub_dir in sub_dirs:
        target_dir = os.path.join(base_dir, sub_dir)
        
        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(target_dir):
            print(f"Cảnh báo: Không tìm thấy thư mục {target_dir}")
            continue

        print(f"Đang kiểm tra thư mục: {target_dir}")

        # Lặp qua các tệp trong thư mục
        for filename in os.listdir(target_dir):
            # Tách tên tệp và phần mở rộng
            name, ext = os.path.splitext(filename)
            
            # Kiểm tra xem tên tệp có nằm trong tập hợp các số cần xóa không
            if name in targets_to_delete:
                file_path = os.path.join(target_dir, filename)
                try:
                    os.remove(file_path)
                    print(f" Đã xóa: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Lỗi khi xóa {file_path}: {e}")

    print(f"\nĐã hoàn thành! Đã xóa tổng cộng {deleted_count} tệp.")

# --- Cách sử dụng đoạn code ---
# 1. Thay đổi 'path/to/your/labels_folder' bằng đường dẫn THỰC TẾ 
#    tới thư mục 'labels' của bạn.
labels_folder_path = "D:\KY_4\DAP\Dap391\Project\Sources(new)\Sources\labels"  # <--- Sửa dòng này
start = 451
end = 500

delete_labels(labels_folder_path, start, end)