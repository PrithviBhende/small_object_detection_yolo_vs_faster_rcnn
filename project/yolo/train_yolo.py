"""
=============================================================
yolo/train_yolo.py
YOLOv11 Training Pipeline on VisDrone-2019
=============================================================
Uses Ultralytics YOLO API for clean, reproducible training.

Key design choices for RTX 3050 Ti (4 GB VRAM):
  - batch=8 with imgsz=640 fits comfortably
  - AMP (mixed precision) enabled by default in Ultralytics
  - workers=4 prevents CPU bottleneck
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_DATA_YAML = "dataset/visdrone_prepared/dataset.yaml"
DEFAULT_OUTPUT_DIR = "yolo/runs"
DEFAULT_WEIGHTS = "yolo11n.pt"   # nano — good for 4 GB VRAM; swap for yolo11s/m


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11 on VisDrone")
    p.add_argument("--data",    default=DEFAULT_DATA_YAML,
                   help="Path to dataset.yaml")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Pretrained COCO weights (e.g. yolo11n.pt, yolo11s.pt)")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=8,
                   help="Batch size — reduce to 4 if OOM on 4 GB VRAM")
    p.add_argument("--device",  default="0",
                   help="'0' for GPU 0, 'cpu' for CPU")
    p.add_argument("--project", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--name",    default="visdrone_yolo11")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


def train(args):
    """
    Main training function.

    Augmentation strategy (specified via Ultralytics overrides):
      mosaic=1.0       — 4-image mosaic for small-object context
      fliplr=0.5       — horizontal flip
      scale=0.5        — random scale ±50 %
      hsv_h=0.015      — hue jitter
      hsv_s=0.7        — saturation jitter
      hsv_v=0.4        — value jitter
      degrees=0.0      — no rotation (UAV imagery is mostly top-down)
      translate=0.1    — slight translation
      mixup=0.1        — light MixUp for regularization
    """
    print("\n" + "="*60)
    print("  YOLOv11 Training — VisDrone-2019")
    print("="*60)
    print(f"  Weights : {args.weights}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"  Device  : {args.device}")
    print("="*60 + "\n")

    # Verify GPU availability
    if args.device != "cpu":
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available — falling back to CPU.")
            args.device = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(int(args.device))
            vram_gb = torch.cuda.get_device_properties(
                int(args.device)
            ).total_memory / 1e9
            print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)\n")

    # ── Load pretrained model ──────────────────────────────────────────
    model = YOLO(args.weights)

    # ── Training hyperparameters ───────────────────────────────────────
    # Cosine LR schedule, warmup, and augmentation flags
    train_args = dict(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,          # initial LR
        lrf=0.01,           # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Loss weights
        box=7.5,            # box regression loss gain
        cls=0.5,            # cls loss gain
        dfl=1.5,            # dfl loss gain
        # Augmentation — tuned for small UAV objects
        mosaic=1.0,         # 4-image mosaic
        mixup=0.1,          # MixUp probability
        copy_paste=0.1,     # Copy-paste augmentation
        degrees=0.0,        # rotation (disabled — UAV top-down)
        translate=0.1,      # translation fraction
        scale=0.5,          # scale ±50%
        shear=0.0,          # no shear
        perspective=0.0,    # no perspective
        flipud=0.0,         # no vertical flip
        fliplr=0.5,         # horizontal flip
        hsv_h=0.015,        # hue
        hsv_s=0.7,          # saturation
        hsv_v=0.4,          # brightness
        # Misc
        workers=4,
        save=True,
        save_period=10,     # save checkpoint every N epochs
        val=True,
        plots=True,         # saves training curves
        verbose=True,
        seed=42,
        amp=True,           # mixed precision (saves ~1 GB VRAM)
        resume=args.resume,
        patience=50,        # early stopping patience
        close_mosaic=10,    # disable mosaic last N epochs
    )

    # ── Start training ─────────────────────────────────────────────────
    t_start = time.time()
    results = model.train(**train_args)
    elapsed = time.time() - t_start

    print(f"\n  Training finished in {elapsed/3600:.2f} hours.")
    print(f"  Best weights: {args.project}/{args.name}/weights/best.pt")

    # ── Save timing info ───────────────────────────────────────────────
    timing = {
        "total_training_seconds": elapsed,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
    }
    out_dir = Path(args.project) / args.name
    with open(out_dir / "training_time.json", "w") as f:
        json.dump(timing, f, indent=2)

    return model, results, out_dir


if __name__ == "__main__":
    args = parse_args()
    train(args)
