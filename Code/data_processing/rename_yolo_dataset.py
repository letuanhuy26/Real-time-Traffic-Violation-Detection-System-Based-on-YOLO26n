"""
RENAME YOLO DATASET - Từ SỐ (1.jpg) thành PREFIX_SỐ
"""

import pathlib
import re


def get_split_dir(base_path, split_name):
    """Tìm folder con"""
    for child in base_path.iterdir():
        if child.is_dir() and child.name.lower() == split_name.lower():
            return child
    raise FileNotFoundError(f"Không tìm thấy folder '{split_name}'")


def extract_number(filename_stem):
    """Lấy số đầu tiên từ tên file"""
    match = re.match(r'^(\d+)', filename_stem)
    if match:
        return int(match.group(1))
    return None


def get_files(img_dir):
    """Lấy tất cả file ảnh"""
    files = [p for p in img_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    files.sort(key=lambda p: extract_number(p.stem) or 999999)
    return files


def show_files(img_dir):
    """Hiển thị danh sách file"""
    files = get_files(img_dir)
    
    print("\n" + "=" * 70)
    print(f"  📋 DANH SÁCH FILE ({len(files)} file)")
    print("=" * 70)
    
    show_count = min(len(files), 20)
    for idx, f in enumerate(files[:show_count], start=1):
        print(f"{idx:3d}. {f.name}")
    
    if len(files) > 20:
        print(f"\n... ({len(files) - 20} file khác) ...\n")
    
    print("=" * 70)
    return files


def filter_files(files, start_num, end_num):
    """Lọc file trong range"""
    selected = []
    for f in files:
        num = extract_number(f.stem)
        if num and start_num <= num <= end_num:
            selected.append(f)
    return selected


def rename_files(img_dir, lbl_dir, selected, prefix, new_num_start):
    """Đổi tên file"""
    
    first_num = extract_number(selected[0].stem)
    offset = new_num_start - first_num
    
    print(f"\n✅ Đang đổi tên {len(selected)} file...\n")

    for f in selected:
        old_num = extract_number(f.stem)
        new_num = old_num + offset
        
        # Tên mới
        new_name = f"{prefix}_{new_num}{f.suffix.lower()}"
        new_path = img_dir / new_name
        
        # Đổi tên ảnh
        f.rename(new_path)
        print(f"  ✔ {f.name} → {new_name}")

        # Đổi tên label
        old_lbl = lbl_dir / f"{f.stem}.txt"
        if old_lbl.exists():
            new_lbl = lbl_dir / f"{prefix}_{new_num}.txt"
            old_lbl.rename(new_lbl)

    print(f"\n✅ Xong! Đã đổi tên {len(selected)} file.")


if __name__ == "__main__":
    print("=" * 70)
    print("  RENAME YOLO DATASET")
    print("=" * 70)

    # Input
    root = input("\n📁 Đường dẫn dataset: ").strip()
    if not root:
        root = r"D:\KY_4\DAP\Dap391\Project\Source_noaug"
    
    split = input("\n📂 Split (train/test/val): ").strip().lower()
    
    try:
        img_dir = get_split_dir(pathlib.Path(root) / "images", split)
        lbl_dir = get_split_dir(pathlib.Path(root) / "labels", split)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        exit(1)

    # Hiển thị file
    all_files = show_files(img_dir)
    
    # Input range
    print("\n" + "=" * 70)
    start = int(input("🔢 File bắt đầu (vd: 1): "))
    end = int(input("🔢 File kết thúc (vd: 100): "))
    prefix = input("✏️  Prefix mới (vd: NgaTu_LTH): ").strip()
    
    new_start_input = input("🔢 Số mới bắt đầu (bỏ trống = giữ số cũ): ").strip()
    new_start = int(new_start_input) if new_start_input else start

    # Lọc file
    selected = filter_files(all_files, start, end)
    
    if not selected:
        print(f"\n❌ Không tìm file từ {start} đến {end}")
        exit(1)

    # Preview
    first_num = extract_number(selected[0].stem)
    offset = new_start - first_num
    
    print("\n" + "=" * 70)
    print(f"  🎯 XEM TRƯỚC ({len(selected)} file)")
    print("=" * 70)
    
    for f in selected[:5]:
        old_num = extract_number(f.stem)
        new_num = old_num + offset
        print(f"  {f.name} → {prefix}_{new_num}{f.suffix.lower()}")
    
    if len(selected) > 10:
        print(f"  ... ({len(selected) - 10} file khác) ...")
        
    for f in selected[-5:]:
        old_num = extract_number(f.stem)
        new_num = old_num + offset
        print(f"  {f.name} → {prefix}_{new_num}{f.suffix.lower()}")

    # Confirm
    confirm = input("\n✅ Tiếp tục? (Enter/q): ").strip().lower()
    if confirm == "q":
        print("Hủy.")
        exit(0)

    # Chạy
    try:
        rename_files(img_dir, lbl_dir, selected, prefix, new_start)
    except Exception as e:
        print(f"❌ Lỗi: {e}")