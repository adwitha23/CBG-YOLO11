import os
import glob
import torch
from ultralytics import YOLO


# ==========================================
# PATHS - Update these before running
# ==========================================
DATASET_ROOT      = "/path/to/output/dataset"       # same as OUT_DATASET_PATH in preprocbr35.py
FOLDS_DIR         = os.path.join(DATASET_ROOT, "5_folds")
ARCHITECTURE_YAML = "/path/to/architecture.yaml"    # e.g. ultralytics/cfg/models/11/your_model.yaml
PRETRAINED_WEIGHTS = "/path/to/pretrained.pt"       # e.g. yolo11s.pt
RUN_NAME_PREFIX   = "your_run_name"


# ==========================================
# COLLECT FOLD YAMLS
# ==========================================
fold_yamls = sorted(glob.glob(os.path.join(FOLDS_DIR, "data_fold_*.yaml")))


# ==========================================
# 5-FOLD TRAINING LOOP
# ==========================================
for fold, yaml_path in enumerate(fold_yamls):
    fold_num = fold + 1

    model = YOLO(ARCHITECTURE_YAML)

    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location="cpu", weights_only=False)
    pretrained_dict = (ckpt.get("ema") or ckpt.get("model")).state_dict()
    model_dict = model.model.state_dict()

    matched_weights = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

    model.model.load_state_dict(matched_weights, strict=False)

    model.train(
        data=yaml_path,
        epochs=200,
        imgsz=1024,
        batch=8,
        amp=True,

        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.001,
        warmup_epochs=5.0,
        label_smoothing=0.05,

        box=12.0,
        cls=1.0,
        dfl=3.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        perspective=0.0001,
        fliplr=0.5,
        flipud=0.5,

        mosaic=0.4,
        mixup=0.1,
        copy_paste=0.3,
        close_mosaic=30,

        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,
        name=f'{RUN_NAME_PREFIX}_fold_{fold_num}',
        patience=0
    )

    best_model = YOLO(f"runs/detect/{RUN_NAME_PREFIX}_fold_{fold_num}/weights/best.pt")

    metrics = best_model.val(
        data=yaml_path,
        split='test',
        imgsz=640,
        batch=8,
        augment=True,
        conf=0.45,
        iou=0.35,
        device=0 if torch.cuda.is_available() else 'cpu'
    )

    print(f"Fold {fold_num} Results:")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")
