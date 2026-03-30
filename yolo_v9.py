# yolov9c_defect_resume_fixed.py
# Resume YOLOv9 training for defect detection (safe, defect-friendly, two-phase)
from ultralytics import YOLO
import os, math, time, multiprocessing, torch

# --- CONFIG ---
DATA_YAML = r"C:\Users\ASHIRWAD PRATAPSINGH\Aircraft_AI\data.yaml"
PROJECT = "runs/train"
RUN_NAME = "yolov9c_airplane_defect_resume_v1_fixed"
BASE_MODEL = "yolov9c.pt"
IMG_SIZE = 704
EFFECTIVE_BATCH = 24
RESUME_EPOCHS = 10          # train only 10 more epochs
PHASE1_HEAD_EPOCHS = 6      # freeze backbone for first X epochs (≤ RESUME_EPOCHS)
PATIENCE = 25
SAVE_PERIOD = 3
WARMUP_EPOCHS = 1
CLOSE_MOSAIC = 10
# -------------------------------------------------

CLASS_WEIGHTS = {
    'crack': 1.0,
    'dent': 1.0,
    'missing-head': 4.5,
    'paint-off': 5.0,
    'scratch': 10.0
}

def find_latest_yolov9_checkpoint(search_root="runs"):
    ckpts = []
    for root, _, files in os.walk(search_root):
        for f in files:
            if f in ("best.pt", "last.pt") and "yolov9" in root.lower():
                ckpts.append(os.path.join(root, f))
    if not ckpts:
        for root, _, files in os.walk(search_root):
            for f in files:
                if f in ("best.pt", "last.pt"):
                    ckpts.append(os.path.join(root, f))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for c in ckpts:
        if c.endswith("best.pt"):
            return c
    return ckpts[0]

def detect_gpu_batch(target=EFFECTIVE_BATCH):
    if not torch.cuda.is_available():
        return "cpu", 1, 1, 0, 1
    dev = 0
    prop = torch.cuda.get_device_properties(dev)
    gb = int(prop.total_memory / (1024 ** 3))
    if gb >= 24: step = 24
    elif gb >= 16: step = 12
    elif gb >= 12: step = 8
    elif gb >= 8: step = 6
    else: step = 4
    acc = max(1, math.ceil(target / step))
    acc = min(acc, 16)
    return dev, step, acc, gb, step * acc

def freeze_backbone(model):
    try:
        backbone = getattr(model.model, "backbone", None)
        if backbone is None:
            backbone = getattr(model.model, "model", None)
        if backbone is None:
            print("⚠️ Could not find `model.model.backbone` - skipping freeze.")
            return False
        count = 0
        for p in backbone.parameters():
            p.requires_grad = False
            count += 1
        print(f"🔒 Frozen backbone parameters: {count}")
        return True
    except Exception as e:
        print("⚠️ Exception while freezing backbone:", e)
        return False

def unfreeze_backbone(model):
    try:
        backbone = getattr(model.model, "backbone", None)
        if backbone is None:
            backbone = getattr(model.model, "model", None)
        if backbone is None:
            print("⚠️ Could not find `model.model.backbone` - skipping unfreeze.")
            return False
        count = 0
        for p in backbone.parameters():
            p.requires_grad = True
            count += 1
        print(f"🔓 Unfroze backbone parameters: {count}")
        return True
    except Exception as e:
        print("⚠️ Exception while unfreezing backbone:", e)
        return False

def apply_loss_adjustments_if_possible(trainer):
    try:
        loss_obj = getattr(trainer, 'criterion', None)
        if loss_obj is None:
            print("ℹ️ Trainer has no `criterion` attribute—skipping loss tweaks.")
            return
        if hasattr(loss_obj, 'cls_weight'):
            w = torch.tensor(list(CLASS_WEIGHTS.values()), device=getattr(trainer, 'device', 'cpu'))
            loss_obj.cls_weight = w
            print("🧠 Applied class weights to loss.")
        if hasattr(loss_obj, 'fl_gamma'):
            loss_obj.fl_gamma = 1.8
            print("🧠 Set focal gamma to 1.8.")
    except Exception as e:
        print("⚠️ Exception while applying loss adjustments:", e)

def train_phase(model, start_epoch, end_epoch, resume_flag, train_args_common, freeze_backbone_flag=False):
    """
    Train one phase.
    - start_epoch, end_epoch are absolute counters from previous runs for bookkeeping.
    - If resume_flag==True -> Ultralytics expects epochs as absolute end_epoch.
    - If resume_flag==False -> we must pass relative epoch count (end_epoch - start_epoch).
    """
    print(f"\n=== Training phase: epochs {start_epoch+1} -> {end_epoch} (resume={resume_flag}) ===")

    if freeze_backbone_flag:
        freeze_backbone(model)

    args = train_args_common.copy()

    # decide epochs argument based on resume_flag
    if resume_flag:
        args["epochs"] = end_epoch   # absolute end epoch (used when truly resuming)
    else:
        n_epochs = max(1, end_epoch - start_epoch)
        args["epochs"] = n_epochs   # run this many epochs for a fresh fine-tune

    # force resume to False for safety unless caller explicitly wants resume semantics
    args["resume"] = resume_flag

    print("🚀 Phase train args (summary):")
    for k in ("data","imgsz","epochs","batch","workers","device","optimizer","lr0","lrf","cache","rect","project","name"):
        if k in args:
            print(f"  {k}: {args[k]}")
    trainer = model.train(**args)

    try:
        apply_loss_adjustments_if_possible(trainer)
    except Exception:
        pass

    return trainer

def main():
    print(">>> YOLOv9c Defect-centric Resume (fixed)")

    ckpt = find_latest_yolov9_checkpoint()
    if ckpt:
        print(f"✅ Found previous checkpoint: {ckpt}")
        model = YOLO(ckpt)     # load weights only
        resume_flag = False    # ALWAYS start a NEW fine-tune session
        # try read last epoch from results.csv (if present)
        epochs_done = 0
        results_csv = os.path.join(os.path.dirname(ckpt), "results.csv")
        if os.path.exists(results_csv):
            try:
                with open(results_csv, "r") as f:
                    lines = f.read().strip().splitlines()
                    if len(lines) >= 2:
                        last_line = lines[-1].split(",")
                        epochs_done = int(float(last_line[0]))
                        print(f"🕓 Detected previous epoch: {epochs_done}")
            except Exception:
                print("⚠️ Couldn't parse results.csv reliably; assuming 0 epochs completed.")
        start_epoch = epochs_done
        target_total = epochs_done + RESUME_EPOCHS
    else:
        print(f"⚠️ No checkpoint found — loading base model {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        resume_flag = False
        start_epoch = 0
        target_total = RESUME_EPOCHS

    # Phase scheduling
    phase1_epochs = min(PHASE1_HEAD_EPOCHS, RESUME_EPOCHS)
    phase2_epochs = RESUME_EPOCHS - phase1_epochs

    dev, step, acc, gb, eff = detect_gpu_batch(EFFECTIVE_BATCH)
    print(f"✅ GPU: {dev} | VRAM: {gb} GB | per-step batch={step} | accumulate={acc}")

    workers = min(max(2, (os.cpu_count() or 4)//4), 8)

    train_args_common = dict(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=step,
        workers=workers,
        device=dev,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.1,
        cos_lr=True,
        weight_decay=0.0004,
        cache="ram",
        rect=True,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        close_mosaic=CLOSE_MOSAIC,
        warmup_epochs=WARMUP_EPOCHS,
        # Disabling destructive augmentations
        mosaic=0.0,
        mixup=0.0,
        perspective=0.0,
        degrees=0.0,
        shear=0.0,
        # Gentle augmentations (photometric + tiny geometry)
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.25,
        translate=0.03,
        scale=0.15,
        fliplr=0.25,
        flipud=0.0,
        label_smoothing=0.02,
    )

    # Phase 1: head-only (freeze backbone) — small number of epochs
    if phase1_epochs > 0:
        phase1_end_abs = start_epoch + phase1_epochs
        print(f"\n--- Phase 1: head-only training for {phase1_epochs} epochs (freeze backbone). ---")
        trainer1 = train_phase(model, start_epoch, phase1_end_abs, resume_flag=False, train_args_common=train_args_common, freeze_backbone_flag=True)
        try:
            apply_loss_adjustments_if_possible(trainer1)
        except Exception:
            pass
        # keep resume_flag False to avoid resume semantics for next phase
        start_epoch = phase1_end_abs

    # Phase 2: unfreeze backbone (if any epochs remain) and train remaining epochs
    if phase2_epochs > 0:
        phase2_end_abs = start_epoch + phase2_epochs
        print(f"\n--- Phase 2: unfreeze backbone for final {phase2_epochs} epochs. ---")
        unfreeze_backbone(model)
        # Slightly reduce lr for fine adaptation
        train_args_common["lr0"] = 0.0005
        trainer2 = train_phase(model, start_epoch, phase2_end_abs, resume_flag=False, train_args_common=train_args_common, freeze_backbone_flag=False)
        try:
            apply_loss_adjustments_if_possible(trainer2)
        except Exception:
            pass
        start_epoch = phase2_end_abs

    print("\n✅ Resume run finished (script completed).")
    print(f"➡ Logs and weights saved under: {os.path.join(PROJECT, RUN_NAME)}")
    print("ℹ️ Notes:")
    print("  - mosaic/mixup are OFF to preserve defect textures.")
    print("  - Phase 1 froze backbone to focus on heads; Phase 2 unfreezes for final adaptation.")
    print("  - This script starts NEW fine-tune runs using previous weights (no 'resume' of finished run).")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
