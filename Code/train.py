from ultralytics import YOLO
import os

if __name__ == '__main__':

    # ── Config ───────────────────────────────────────────────
    DATA_YAML  = r"D:\KY_4\DAP\Dap391\Project\Sources\data.yaml"
    MODEL_PATH = r"D:\KY_4\DAP\Dap391\Project\yolo26n.pt"
    OUTPUT_DIR = r"D:\KY_4\DAP\Dap391\Project\Sources\Model"
    RUN_NAME   = "yolo26n_before_aug"

    # ── Train ────────────────────────────────────────────────
    model = YOLO(MODEL_PATH)
    model.train(
        data       = DATA_YAML,
        epochs     = 100,
        imgsz      = 1920,
        batch      = 4,
        workers    = 2,
        rect       = True,
        optimizer  = "AdamW",
        lr0        = 0.001,
        cos_lr     = True,
        patience   = 30,
        project    = OUTPUT_DIR,
        name       = RUN_NAME,
        exist_ok   = True,
        augment    = False,
        mosaic     = 0.0,
        mixup      = 0.0,
        copy_paste = 0.0,
        degrees    = 0.0,
        translate  = 0.0,
        scale      = 0.0,
        shear      = 0.0,
        perspective= 0.0,
        flipud     = 0.0,
        fliplr     = 0.0,
        hsv_h      = 0.0,
        hsv_s      = 0.0,
        hsv_v      = 0.0,
        erasing    = 0.0,
    )

    # ── Val trên test set ────────────────────────────────────
    best_pt = os.path.join(OUTPUT_DIR, RUN_NAME, "weights", "best.pt")
    model_val = YOLO(best_pt)
    metrics = model_val.val(
        data    = DATA_YAML,
        split   = "test",
        imgsz   = 1920,
        batch   = 4,
        rect    = True,
        workers = 2,
    )

    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"P:        {metrics.box.mp:.4f}")
    print(f"R:        {metrics.box.mr:.4f}")
    print(f"\nDelta mAP50 vs w/ aug: {0.9726 - metrics.box.map50:+.4f}")