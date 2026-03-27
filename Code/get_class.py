from ultralytics import YOLO

if __name__ == '__main__':
    BEST_PT   = r"D:\KY_4\DAP\Dap391\Project\Benchmark\benchmark_yolo26n\weights\best.pt"
    DATA_YAML = r"D:\KY_4\DAP\Dap391\Sources\data.yaml"

    model = YOLO(BEST_PT)
    metrics = model.val(
        data    = DATA_YAML,
        split   = "test",
        imgsz   = 1920,
        batch   = 4,
        rect    = True,
        workers = 2,
    )

    names = model.names  # {0: 'stop_line', 1: 'green', ...}
    print("\n── Per-class AP50 (Table III) ──")
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {names[i]:<15}: {ap:.4f}")
    print(f"\n  mAP50 overall : {metrics.box.map50:.4f}")