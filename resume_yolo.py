from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import os
import gc

def find_latest_yolov8_checkpoint(runs_dir: Path):
    """Find the latest YOLOv8 checkpoint only."""
    runs = list(runs_dir.iterdir())
    if not runs:
        return None

    # sort by last modification
    runs_sorted = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

    for run in runs_sorted:
        # ✅ Skip YOLOv9 or other runs automatically
        if "yolov9" in run.name.lower():
            continue

        weights_dir = run / "weights"
        if weights_dir.exists():
            last = weights_dir / "last.pt"
            best = weights_dir / "best.pt"
            for ckpt in [last, best]:
                if ckpt.exists():
                    # verify checkpoint metadata
                    try:
                        meta = torch.load(ckpt, map_location='cpu')
                        if 'model' in meta and 'yaml' in meta:
                            if 'yolov8' in meta['model'].__class__.__name__.lower():
                                return ckpt
                    except Exception:
                        continue
    return None

def main():
    gc.collect()
    torch.cuda.empty_cache()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    project_root = Path.cwd()
    runs_dir = project_root / "runs" / "detect"
    ckpt = find_latest_yolov8_checkpoint(runs_dir) if runs_dir.exists() else None

    if ckpt:
        print(f"✅ Found YOLOv8 checkpoint: {ckpt}")
        model = YOLO(str(ckpt))
        resume_flag = True
    else:
        print("🚀 No YOLOv8 checkpoint found — starting fresh from yolov8n.pt")
        model = YOLO('yolov8n.pt')
        resume_flag = False

    data_path = r"C:/Users/ASHIRWAD PRATAPSINGH/Downloads/Innovation Hangar v2.v1-aug.yolov9/data.yaml"

    results = model.train(
    data=data_path,
    epochs=100,
    imgsz=640,
    batch=-1,
    device=device,
    cache=True,
    workers=4,
    name='airplane_defect_yolov8_refined',
    resume=resume_flag,

    # Training configuration
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    momentum=0.937,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    label_smoothing=0.05,
    dropout=0.1,
    cos_lr=True,
    patience=10,
    amp=True  # keep mixed precision
)


    print("Training complete ✅")
    print("Results saved in:", results.save_dir)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
 