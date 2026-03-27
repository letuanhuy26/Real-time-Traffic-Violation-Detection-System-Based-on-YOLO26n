import os
import shutil

# Đường dẫn đến 2 thư mục
img_folder = r"D:\KY_4\DAP\Dap391\Project\Sources(new)\dataset\images"
lbl_folder = r"D:\KY_4\DAP\Dap391\Project\Sources(new)\dataset\labels"

image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Lấy danh sách file ảnh và sắp xếp để đảm bảo thứ tự khớp nhau
img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(image_exts)]
img_files.sort()

start_num = 1
# Giả sử bạn muốn đổi toàn bộ file trong danh sách đã lấy
print(f"🔄 Đang xử lý khớp nhãn cho {len(img_files)} bộ file...")

for i, img_name in enumerate(img_files):
    # 1. Lấy tên gốc (không bao gồm đuôi file)
    base_name = os.path.splitext(img_name)[0]
    img_ext = os.path.splitext(img_name)[1]
    
    # 2. Xác định đường dẫn cũ
    old_img_path = os.path.join(img_folder, img_name)
    old_lbl_path = os.path.join(lbl_folder, base_name + ".txt")
    
    # 3. Xác định tên mới và đường dẫn mới
    new_name_no_ext = str(start_num + i)
    new_img_path = os.path.join(img_folder, new_name_no_ext + img_ext)
    new_lbl_path = os.path.join(lbl_folder, new_name_no_ext + ".txt")

    # 4. Đổi tên file ảnh (ghi đè nếu trùng)
    if old_img_path != new_img_path:
        shutil.move(old_img_path, new_img_path)

    # 5. Đổi tên file nhãn tương ứng (nếu có)
    if os.path.exists(old_lbl_path):
        if old_lbl_path != new_lbl_path:
            shutil.move(old_lbl_path, new_lbl_path)
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy file nhãn cho {img_name}")

print(f"✅ Đã đồng bộ xong! Tên mới chạy từ {start_num} đến {start_num + len(img_files) - 1}")