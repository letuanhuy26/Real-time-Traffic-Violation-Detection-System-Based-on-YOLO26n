import albumentations as A
import cv2
import os
from pathlib import Path

# ======= CẤU HÌNH =======
IMG_DIR   = r"D:\KY_4\DAP\Dap391\Project\Sources\images\Train"
LABEL_DIR = r"D:\KY_4\DAP\Dap391\Project\Sources\labels\Train"

TARGET_CLASSES = { 
    1:1, #Green
    3: 2,   # Yellow light: 1949 → ~3898
    4: 4,   # Sub light:     464 → ~2320
}

transform_heavy = A.Compose([
    A.HorizontalFlip(p=0.5),                                                        # lật ngang 50%
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),   # thay đổi độ sáng/tương phản
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.8),  # thay đổi màu sắc
    A.GaussNoise(p=0.5),                                                            # thêm nhiễu
    A.MotionBlur(blur_limit=7, p=0.5),                                              # giả lập chuyển động
    A.RandomShadow(p=0.5),                                                          # thêm bóng đổ
    A.RandomFog(p=0.3),                                                             # thêm sương mù
    A.RandomRain(p=0.2),                                                            # thêm mưa
    A.Affine(translate_percent=0.05, scale=(0.8, 1.2), rotate=(-5, 5), p=0.6),    # dịch chuyển/zoom/xoay
    A.CLAHE(p=0.4),                                                                 # tăng độ tương phản cục bộ
    A.ImageCompression(quality_range=(75, 100), p=0.3),                            # giả lập ảnh nén
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.5, min_area=100))


def has_target_class(label_path, target_classes):
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        lines = f.readlines()
    found = set()
    for line in lines:
        cls = int(line.split()[0])
        if cls in target_classes:
            found.add(cls)
    if not found:
        return None
    return max(found, key=lambda c: target_classes[c])


def read_labels(label_path):
    bboxes, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if w <= 0 or h <= 0:
                print(f"⚠️ Bỏ qua bbox lỗi class {cls}: w={w}, h={h}")
                continue
            bboxes.append([x_c, y_c, w, h])
            class_labels.append(cls)
    return bboxes, class_labels


def save_labels(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{int(cls)} {' '.join([f'{x:.6f}' for x in bbox])}\n")


# ===== CHẠY AUGMENT =====
img_files = list(Path(IMG_DIR).glob("*.jpg")) + list(Path(IMG_DIR).glob("*.png"))

total = 0
for img_path in img_files:
    label_path = Path(LABEL_DIR) / (img_path.stem + ".txt")

    priority_class = has_target_class(label_path, TARGET_CLASSES)
    if priority_class is None:
        continue

    augment_times = TARGET_CLASSES[priority_class]

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = read_labels(label_path)

    for i in range(augment_times):
        try:
            augmented = transform_heavy(image=img, bboxes=bboxes, class_labels=class_labels)  # ← fix ở đây

            new_name = f"{img_path.stem}_aug{i}"
            new_img_path   = Path(IMG_DIR)   / f"{new_name}.jpg"
            new_label_path = Path(LABEL_DIR) / f"{new_name}.txt"

            aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_labels(new_label_path, augmented["bboxes"], augmented["class_labels"])
            total += 1

        except Exception as e:
            print(f"Lỗi {img_path.stem}_aug{i}: {e}")

print(f"✅ Đã tạo thêm {total} ảnh mới!")