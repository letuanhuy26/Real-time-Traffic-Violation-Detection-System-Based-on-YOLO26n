
from ultralytics import YOLO
import os, shutil, csv, time, multiprocessing
from datetime import datetime
import torch
print(torch.cuda.is_available())
print(torch.__version__)

# ============================================================
# CẤU HÌNH
# ============================================================
DATA_YAML  = r"D:\KY_4\DAP\Dap391\Project\Sources\data.yaml"
OUTPUT_DIR = r"D:\KY_4\DAP\Dap391\Project\Benchmark"
EPOCHS     = 100
IMGSZ      = 1920
WORKERS    = 2

# LOCAL: 4 model nhẹ (RTX 4070 Laptop 8GB)
MODELS = [
    #"yolov8n.pt",    # YOLOv8 Nano  - baseline nhỏ nhất
    "yolov8s.pt",    # YOLOv8 Small - baseline trung bình
    "yolo11n.pt",    # YOLOv11 Nano - thế hệ trước YOLO26
    "yolo26n.pt",    # YOLO26 Nano  - mới nhất 2025
]
# ============================================================

def train_model(model_name):
    print(f"\n{'='*60}")
    print(f"🚀 Bắt đầu train: {model_name}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
    print(f"{'='*60}\n")

    model_tag  = model_name.replace(".pt", "")
    start_time = time.time()

    try:
        model = YOLO(model_name)
        model.train(
            data      = DATA_YAML,
            epochs    = EPOCHS,
            imgsz     = IMGSZ,
            batch     = 4,          # 8GB VRAM + imgsz=1920
            workers   = WORKERS,
            rect      = True,       # giữ tỉ lệ 16:9
            optimizer = "AdamW",
            lr0       = 0.001,
            cos_lr    = True,
            patience  = 30,
            project   = OUTPUT_DIR,
            name      = f"benchmark_{model_tag}",
            exist_ok  = True,
        )

        elapsed = time.time() - start_time
        hours   = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        result_dir = os.path.join(OUTPUT_DIR, f"benchmark_{model_tag}")
        csv_path   = os.path.join(result_dir, "results.csv")

        map50, map50_95, precision, recall = "-", "-", "-", "-"
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
                if rows:
                    best      = max(rows, key=lambda x: float(x.get("metrics/mAP50(B)", 0) or 0))
                    map50     = round(float(best.get("metrics/mAP50(B)", 0)), 4)
                    map50_95  = round(float(best.get("metrics/mAP50-95(B)", 0)), 4)
                    precision = round(float(best.get("metrics/precision(B)", 0)), 4)
                    recall    = round(float(best.get("metrics/recall(B)", 0)), 4)

        # Copy best.pt ra output với tên model
        best_src = os.path.join(result_dir, "weights", "best.pt")
        best_dst = os.path.join(OUTPUT_DIR, f"best_{model_tag}.pt")
        if os.path.exists(best_src):
            shutil.copy(best_src, best_dst)

        print(f"\n✅ Xong {model_tag} | {hours}h{minutes}m")
        print(f"   mAP50:    {map50}")
        print(f"   mAP50-95: {map50_95}")
        print(f"   P:        {precision}")
        print(f"   R:        {recall}")

        return {
            "model":     model_tag,
            "time":      f"{hours}h{minutes}m",
            "mAP50":     map50,
            "mAP50-95":  map50_95,
            "Precision": precision,
            "Recall":    recall,
            "status":    "✅ Success"
        }

    except Exception as e:
        print(f"\n❌ LỖI {model_name}: {e}")
        return {
            "model":     model_tag,
            "time":      "-",
            "mAP50":     "-",
            "mAP50-95":  "-",
            "Precision": "-",
            "Recall":    "-",
            "status":    f"❌ {str(e)[:50]}"
        }

def save_summary(results):
    path = os.path.join(OUTPUT_DIR, "benchmark_summary_local.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("BENCHMARK SUMMARY (LOCAL) - TRAFFIC LIGHT DATASET\n")
        f.write(f"Hoàn thành: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}\n")
        f.write(f"Hardware: RTX 4070 Laptop 8GB | imgsz={IMGSZ} | batch=4 | epochs={EPOCHS}\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"{'Model':<15} {'mAP50':<10} {'mAP50-95':<12} {'P':<10} {'R':<10} {'Time':<10} Status\n")
        f.write("-" * 75 + "\n")
        for r in results:
            f.write(
                f"{r['model']:<15} "
                f"{str(r['mAP50']):<10} "
                f"{str(r['mAP50-95']):<12} "
                f"{str(r['Precision']):<10} "
                f"{str(r['Recall']):<10} "
                f"{r['time']:<10} "
                f"{r['status']}\n"
            )
    print(f"📄 Saved: {path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🎯 AUTO BENCHMARK LOCAL")
    print(f"📦 Models: {', '.join(MODELS)}")
    print(f"🖼️  imgsz={IMGSZ} | batch=4 | epochs={EPOCHS}")
    print(f"📁 Output: {OUTPUT_DIR}\n")

    all_results = []
    total_start = time.time()

    for model_name in MODELS:
        result = train_model(model_name)
        all_results.append(result)
        save_summary(all_results)  # lưu sau mỗi model phòng tắt máy đột ngột

    total_elapsed = time.time() - total_start
    total_hours   = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"🏁 XONG {len(MODELS)} MODEL! Tổng: {total_hours}h{total_minutes}m")
    print(f"{'='*60}\n")

    print(f"{'Model':<15} {'mAP50':<10} {'mAP50-95':<12} {'P':<10} {'R':<10} {'Time'}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['model']:<15} {str(r['mAP50']):<10} {str(r['mAP50-95']):<12} {str(r['Precision']):<10} {str(r['Recall']):<10} {r['time']}")

    save_summary(all_results)
    print("\n✅ Check benchmark_summary_local.txt!")