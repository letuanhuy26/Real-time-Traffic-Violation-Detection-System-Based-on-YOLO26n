import os

def delete_images(base_dir, start_idx, end_idx):
    """
    Xóa các tệp hình ảnh nằm trong khoảng [start_idx, end_idx]
    trong các thư mục con 'Train', 'Validate', 'Test' của images_dir.
    """

    # Danh sách các thư mục con (Đảm bảo viết hoa/thường khớp với máy bạn)
    sub_dirs = ['Train', 'Validate', 'Test']

    # Tạo tập hợp các số cần xóa (dạng chuỗi)
    targets_to_delete = {str(i) for i in range(start_idx, end_idx + 1)}

    deleted_count = 0

    for sub_dir in sub_dirs:
        target_dir = os.path.join(base_dir, sub_dir)
        
        if not os.path.exists(target_dir):
            print(f" Cảnh báo: Không tìm thấy thư mục {target_dir}")
            continue

        print(f" Đang kiểm tra thư mục ảnh: {target_dir}")

        for filename in os.listdir(target_dir):
            # Tách tên tệp (name) và phần mở rộng (ext)
            name, ext = os.path.splitext(filename)
            
            # Kiểm tra xem tên tệp (không tính đuôi .jpg, .png) có trong danh sách xóa không
            if name in targets_to_delete:
                file_path = os.path.join(target_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"  Đã xóa ảnh: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Lỗi khi xóa {filename}: {e}")

    print(f"\n✨ Hoàn thành! Đã xóa tổng cộng {deleted_count} hình ảnh.")

# --- CẤU HÌNH TẠI ĐÂY ---
# Lưu ý: Thêm chữ 'r' trước dấu ngoặc kép để tránh lỗi đường dẫn Windows (\)
images_folder_path = r"D:\KY_4\DAP\Dap391\Project\Sources(new)\Sources\images" 
start = 451
end = 500

delete_images(images_folder_path, start, end)