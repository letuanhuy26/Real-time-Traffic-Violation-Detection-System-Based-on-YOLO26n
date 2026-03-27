import os
from pathlib import Path

IMG_DIR   = r"D:\KY_4\DAP\Dap391\Project\Source_noaug\images\Train"
LABEL_DIR = r"D:\KY_4\DAP\Dap391\Project\Source_noaug\labels\Train"

count = 0
for folder in [IMG_DIR, LABEL_DIR]:
    for f in Path(folder).glob("*_aug*"):
        os.remove(f)
        count += 1
print(f"✅ Đã xóa {count} file aug cũ!")