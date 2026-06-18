#!/usr/bin/env python3
"""Fine-tune a YOLO model on the simulation dataset.

Uses Ultralytics training API with settings optimized for:
- Small images (320x240, trained at imgsz=240)
- Person detection only (single class)
- RTX 2070 GPU (8GB VRAM → batch=16 for nano/small, batch=8 for medium)

Usage:
    # Train YOLO11s (recommended)
    python train_yolo.py --model yolo11s.pt --data ../data/yolo_dataset/data.yaml

    # Train YOLO12s
    python train_yolo.py --model yolo12s.pt --data ../data/yolo_dataset/data.yaml

    # Resume interrupted training
    python train_yolo.py --resume runs/detect/train/weights/last.pt

    # Quick test (fewer epochs)
    python train_yolo.py --model yolo11n.pt --epochs 10 --batch 16

Reference:
    https://docs.ultralytics.com/modes/train/
    https://agneya.medium.com/how-to-train-a-custom-yolo-model-for-free-d189178f1c66
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def train(args):
    """Run YOLO fine-tuning."""

    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
        return results

    # Load pretrained model
    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    # Verify data config
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    # Determine batch size based on model size if not specified
    batch = args.batch
    if batch is None:
        model_name = args.model.lower()
        if "n.pt" in model_name:
            batch = 16  # nano: small model, larger batch
        elif "s.pt" in model_name:
            batch = 16  # small: still fits
        elif "m.pt" in model_name:
            batch = 8   # medium: needs more VRAM
        else:
            batch = 8

    print(f"\n{'=' * 60}")
    print(f"  YOLO Fine-Tuning Configuration")
    print(f"{'=' * 60}")
    print(f"  Model:       {args.model}")
    print(f"  Data:        {data_path}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {batch}")
    print(f"  Image size:  {args.imgsz}")
    print(f"  Device:      {args.device}")
    print(f"  Workers:     {args.workers}")
    print(f"  Project:     {args.project}")
    print(f"  Name:        {args.name}")
    print(f"{'=' * 60}\n")

    # Train
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        # ── Hyperparameters ────────────────────────────────
        lr0=args.lr0,           # initial learning rate
        lrf=args.lrf,           # final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # ── Augmentation (light for simulation) ────────────
        hsv_h=0.015,            # hue augmentation
        hsv_s=0.7,              # saturation augmentation
        hsv_v=0.4,              # value augmentation
        degrees=0.0,            # rotation (keep 0 — actors upright)
        translate=0.1,          # translation
        scale=0.5,              # scale (important: varies distance)
        shear=0.0,
        flipud=0.0,             # no vertical flip (people don't flip)
        fliplr=0.5,             # horizontal flip
        mosaic=1.0,             # mosaic augmentation
        mixup=0.1,              # mixup augmentation
        copy_paste=0.1,         # copy-paste augmentation
        # ── Training settings ──────────────────────────────
        patience=args.patience, # early stopping patience
        save_period=10,         # save checkpoint every N epochs
        cache=False,            # cache images (set True if enough RAM)
        # ── Other ──────────────────────────────────────────
        exist_ok=args.exist_ok,
        pretrained=True,        # start from pretrained weights
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=False,    # slightly faster, minor reproducibility trade-off
        single_cls=False,       # keep as single class (person only)
        rect=False,             # rectangular training (set True for faster val)
        cos_lr=True,            # cosine learning rate schedule
        close_mosaic=10,        # disable mosaic for last N epochs
        resume=False,
        amp=True,               # automatic mixed precision
        # ── Regularization ─────────────────────────────────
        dropout=0.0,            # dropout (0 = disabled by default in YOLO)
    )

    # Print final results
    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"{'=' * 60}")

    # Best model path
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.exists():
        print(f"  Best model:   {best_path}")
    last_path = Path(args.project) / args.name / "weights" / "last.pt"
    if last_path.exists():
        print(f"  Last model:   {last_path}")

    # Validate on test set
    print(f"\n{'=' * 60}")
    print(f"  Validating on test set...")
    print(f"{'=' * 60}")
    best_model = YOLO(str(best_path))
    val_results = best_model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        verbose=True,
    )
    print(f"\n  Test mAP@0.5:    {val_results.box.map50:.4f}")
    print(f"  Test mAP@0.5:95: {val_results.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO model on simulation dataset"
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="yolo11s.pt",
        help="Pretrained YOLO model to fine-tune (default: yolo11s.pt)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to last.pt to resume training"
    )

    # Data
    parser.add_argument(
        "--data", type=str,
        default=str(Path(__file__).parent.parent / "data" / "yolo_dataset" / "data.yaml"),
        help="Path to data.yaml (default: ../data/yolo_dataset/data.yaml)"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (auto-detected if None)")
    parser.add_argument("--imgsz", type=int, default=240, help="Training image size (default: 240)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (default: 0, use 'cpu' for CPU)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers (default: 8)")

    # Hyperparameters
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor (default: 0.01)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20)")

    # Output
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project dir")
    parser.add_argument("--name", type=str, default="finetune_sim", help="Experiment name")
    parser.add_argument("--exist_ok", action="store_true", help="Overwrite existing experiment")

    args = parser.parse_args()

    # Auto-generate name based on model if not specified
    if args.name == "finetune_sim":
        model_tag = args.model.replace(".pt", "").replace("-", "_")
        args.name = f"finetune_{model_tag}"

    train(args)


if __name__ == "__main__":
    main()