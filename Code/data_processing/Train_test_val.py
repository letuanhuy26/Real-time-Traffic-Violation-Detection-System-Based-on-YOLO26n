"""
Train_test_val.py — Chia dataset thành Train / Val / Test
═══════════════════════════════════════════════════════════
Mục đích:
    Chia bộ dữ liệu (ảnh .jpg + nhãn .txt YOLO format) thành 3 tập:
        • Train 60%  — dùng để huấn luyện model
        • Val   20%  — dùng để đánh giá trong quá trình train
        • Test  20%  — dùng để đánh giá cuối cùng

Cách dùng:
    1. Sửa `src_img`, `src_label` trỏ tới thư mục chứa ảnh/nhãn gốc.
    2. Sửa `dst` trỏ tới thư mục Sources (đích).
    3. Đảm bảo cấu trúc thư mục đích đã có sẵn:
         Sources/images/train/  |  Sources/labels/train/
         Sources/images/val/    |  Sources/labels/val/
         Sources/images/test/   |  Sources/labels/test/
    4. Chạy: python Train_test_val.py

Lưu ý:
    - Dữ liệu được shuffle ngẫu nhiên trước khi chia → kết quả mỗi lần chạy sẽ khác nhau.
    - File ảnh (.jpg) và nhãn (.txt) phải có cùng tên (stem) để ghép đôi đúng.
    - Script COPY file (không move), dữ liệu gốc giữ nguyên.
"""

import os, shutil, random
from pathlib import Path

# ──────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN (SỬA TRƯỚC KHI CHẠY)
# ──────────────────────────────────────────────
src_img   = r"C:\Users\Admin\Downloads\tong_anh"   # Thư mục chứa ảnh gốc (.jpg)
src_label = r"C:\Users\Admin\Downloads\tong_labels"   # Thư mục chứa nhãn gốc (.txt)

dst = r"D:\KY_4\DAP\Dap391\Project\Sources"                # Thư mục đích (Sources)

# Tỷ lệ chia: tổng phải = 1.0
splits = {"Train": 0.6, "Validate": 0.2, "Test": 0.2}

# ──────────────────────────────────────────────
# BƯỚC 1: Lấy danh sách file và shuffle ngẫu nhiên
# ──────────────────────────────────────────────
files = [f.stem for f in Path(src_img).glob("*.jpg")]
random.shuffle(files)

# ──────────────────────────────────────────────
# BƯỚC 2: Tính điểm cắt theo tỷ lệ 60/20/20
# ──────────────────────────────────────────────
total = len(files)
train_end = int(total * 0.6)   # index kết thúc tập train
val_end   = int(total * 0.8)   # index kết thúc tập val (= train + val)

split_files = {
    "Train": files[:train_end],         # 0 → 60%
    "Validate": files[train_end:val_end],   # 60% → 80%
    "Test":  files[val_end:]             # 80% → 100%
}

# ──────────────────────────────────────────────
# BƯỚC 3: Copy ảnh + nhãn vào thư mục tương ứng
# ──────────────────────────────────────────────
for split, file_list in split_files.items():
    for name in file_list:
        # Copy ảnh (.jpg) → dst/images/{split}/
        shutil.copy(
            f"{src_img}/{name}.jpg",
            f"{dst}/images/{split}/{name}.jpg"
        )
        # Copy nhãn (.txt) → dst/labels/{split}/
        shutil.copy(
            f"{src_label}/{name}.txt",
            f"{dst}/labels/{split}/{name}.txt"
        )
    print(f"{split}: {len(file_list)} files")