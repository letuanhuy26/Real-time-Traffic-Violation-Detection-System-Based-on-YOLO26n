import os

def check_mismatched_data(images_dir, labels_dir):
    """
    Kiểm tra và liệt kê các ảnh không có nhãn và nhãn không có ảnh
    trong các thư mục con 'Train', 'Validate', 'Test'.
    """
    sub_dirs = ['Train', 'Validate', 'Test']
    
    total_missing_labels = 0
    total_missing_images = 0

    for sub_dir in sub_dirs:
        print(f"\n" + "="*40)
        print(f"🔍 ĐANG KIỂM TRA THƯ MỤC: {sub_dir}")
        print(f"="*40)
        
        img_target_dir = os.path.join(images_dir, sub_dir)
        lbl_target_dir = os.path.join(labels_dir, sub_dir)
        
        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(img_target_dir):
            print(f"⚠️ Không tìm thấy thư mục ảnh: {img_target_dir}")
            continue
        if not os.path.exists(lbl_target_dir):
            print(f"⚠️ Không tìm thấy thư mục nhãn: {lbl_target_dir}")
            continue
            
        # Lấy tên file (bỏ đuôi .jpg, .txt...) và cho vào một Set (tập hợp)
        # Bỏ qua các file ẩn/cấu hình như classes.txt hoặc .DS_Store
        image_names = {
            os.path.splitext(f)[0] for f in os.listdir(img_target_dir) 
            if os.path.isfile(os.path.join(img_target_dir, f)) and not f.startswith('.')
        }
        
        label_names = {
            os.path.splitext(f)[0] for f in os.listdir(lbl_target_dir) 
            if os.path.isfile(os.path.join(lbl_target_dir, f)) and f != "classes.txt" and not f.startswith('.')
        }

        # Tìm sự khác biệt bằng phép toán tập hợp
        images_without_labels = image_names - label_names
        labels_without_images = label_names - image_names
        
        # Thống kê
        total_missing_labels += len(images_without_labels)
        total_missing_images += len(labels_without_images)
        
        # In kết quả cho từng thư mục con
        if not images_without_labels and not labels_without_images:
            print("✅ Dữ liệu khớp hoàn toàn! (1 ảnh đi kèm 1 nhãn)")
        else:
            if images_without_labels:
                sorted_list = sorted(images_without_labels, key=lambda x: (len(x), x))
                print(f"❌ CÓ {len(images_without_labels)} ẢNH CHƯA CÓ NHÃN (LABEL):")
                for name in sorted_list:
                    print(f"   - {name}")
                
            if labels_without_images:
                sorted_list = sorted(labels_without_images, key=lambda x: (len(x), x))
                print(f"❌ CÓ {len(labels_without_images)} NHÃN CHƯA CÓ ẢNH:")
                for name in sorted_list:
                    print(f"   - {name}")

    # Tổng kết
    print("\n" + "*"*40)
    print("📊 TỔNG KẾT TOÀN BỘ DATASET:")
    print(f"Tổng số ảnh bị thiếu nhãn: {total_missing_labels}")
    print(f"Tổng số nhãn bị thiếu ảnh: {total_missing_images}")
    print("*"*40)


def report_dataset_count(images_dir, labels_dir):
    """
    Báo cáo tổng số lượng images và labels trong từng thư mục con
    và tổng cộng toàn bộ dataset.
    """
    sub_dirs = ['Train', 'Validate', 'Test']
    
    grand_total_images = 0
    grand_total_labels = 0

    print("\n" + "="*50)
    print("📊 BÁO CÁO SỐ LƯỢNG DATASET")
    print("="*50)

    for sub_dir in sub_dirs:
        img_target_dir = os.path.join(images_dir, sub_dir)
        lbl_target_dir = os.path.join(labels_dir, sub_dir)

        # Đếm images
        if os.path.exists(img_target_dir):
            img_count = sum(
                1 for f in os.listdir(img_target_dir)
                if os.path.isfile(os.path.join(img_target_dir, f)) and not f.startswith('.')
            )
        else:
            img_count = 0

        # Đếm labels
        if os.path.exists(lbl_target_dir):
            lbl_count = sum(
                1 for f in os.listdir(lbl_target_dir)
                if os.path.isfile(os.path.join(lbl_target_dir, f))
                and f != "classes.txt" and not f.startswith('.')
            )
        else:
            lbl_count = 0

        grand_total_images += img_count
        grand_total_labels += lbl_count

        print(f"\n📁 {sub_dir}:")
        print(f"   🖼️  Images: {img_count}")
        print(f"   🏷️  Labels: {lbl_count}")

    print("\n" + "-"*50)
    print(f"📦 TỔNG CỘNG:")
    print(f"   🖼️  Tổng Images: {grand_total_images}")
    print(f"   🏷️  Tổng Labels: {grand_total_labels}")
    print("="*50)


# --- CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY ---
images_folder = r"D:\KY_4\DAP\Dap391\Project\Source_noaug\images"
labels_folder = r"D:\KY_4\DAP\Dap391\Project\Source_noaug\labels"

check_mismatched_data(images_folder, labels_folder)
report_dataset_count(images_folder, labels_folder)