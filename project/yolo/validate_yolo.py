"""
=============================================================
yolo/validate_yolo.py
YOLOv11 Validation & Inference Timing on VisDrone-2019
=============================================================
Runs model.val() and measures per-image inference latency.

Output saved to:
    yolo/runs/<name>/
        yolo_metrics.json   ← all metrics
        inference_times.json ← latency breakdown
"""

import os
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate YOLOv11 on VisDrone val split"
    )
    p.add_argument(
        "--weights",
        type=str,
        default="yolo/runs/visdrone_yolo11/weights/best.pt",
        help="Path to trained best.pt"
    )
    p.add_argument(
        "--data",
        type=str,
        default="dataset/visdrone_prepared/dataset.yaml",
        help="Path to dataset.yaml"
    )
    p.add_argument(
        "--imgsz",  type=int, default=640
    )
    p.add_argument(
        "--batch",  type=int, default=8
    )
    p.add_argument(
        "--device", type=str, default="0"
    )
    p.add_argument(
        "--conf",   type=float, default=0.001,
        help="Confidence threshold (low for mAP computation)"
    )
    p.add_argument(
        "--iou",    type=float, default=0.6,
        help="NMS IoU threshold"
    )
    p.add_argument(
        "--output_dir", type=str, default="yolo/runs/visdrone_yolo11",
        help="Directory to save metric JSONs"
    )
    p.add_argument(
        "--num_timing_images", type=int, default=200,
        help="Number of images used for inference latency measurement"
    )
    return p.parse_args()


# ─── Validation ───────────────────────────────────────────────────────────────

def validate(args):
    """
    Run official YOLO validation to compute:
        mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1

    Ultralytics returns a Results object with:
        results.box.map50   — mAP@0.5
        results.box.map     — mAP@0.5:0.95
        results.box.mp      — mean Precision
        results.box.mr      — mean Recall
    """
    print("\n" + "="*60)
    print("  YOLOv11 Validation")
    print("="*60)
    print(f"  Weights : {args.weights}")
    print(f"  Data    : {args.data}")
    print("="*60 + "\n")

    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            "Run train_yolo.py first."
        )

    # Load model
    model = YOLO(args.weights)

    # ── Run validation ─────────────────────────────────────────────────
    val_results = model.val(
        data=str(Path(args.data).resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
        save_json=True,
        plots=True,
    )

    # Extract core metrics
    precision  = float(val_results.box.mp)
    recall     = float(val_results.box.mr)
    map50      = float(val_results.box.map50)
    map50_95   = float(val_results.box.map)

    # F1 = harmonic mean of P and R
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # Per-class metrics
    per_class = {}
    class_names = val_results.names
    if hasattr(val_results.box, "ap_class_index"):
        for i, cls_idx in enumerate(val_results.box.ap_class_index):
            cls_name = class_names[int(cls_idx)]
            per_class[cls_name] = {
                "ap50": float(val_results.box.ap50[i]),
                "ap":   float(val_results.box.ap[i]),
            }

    print("\n" + "="*60)
    print("  Validation Results")
    print("="*60)
    print(f"  Precision      : {precision*100:.2f}%")
    print(f"  Recall         : {recall*100:.2f}%")
    print(f"  F1-score       : {f1*100:.2f}%")
    print(f"  mAP@0.5        : {map50*100:.2f}%")
    print(f"  mAP@0.5:0.95   : {map50_95*100:.2f}%")
    print("="*60)

    return {
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "map50":      map50,
        "map50_95":   map50_95,
        "per_class":  per_class,
    }


# ─── Inference latency measurement ────────────────────────────────────────────

def measure_inference_time(args, n_warmup: int = 10) -> dict:
    """
    Measure inference latency (ms/image) on the val set.

    Strategy:
      1. Warm up the GPU with n_warmup images.
      2. Time individual image inference using torch.cuda.synchronize()
         for accurate GPU timing.
      3. Report mean, std, min, max.

    Returns:
        dict with timing statistics
    """
    print("\n" + "="*60)
    print("  Measuring Inference Latency")
    print("="*60)

    from ultralytics.data.utils import check_det_dataset
    import yaml

    model = YOLO(args.weights)

    # Get val image paths from dataset.yaml
    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(data_cfg.get("path", "."))
    val_rel   = data_cfg.get("val", "images/val")
    val_dir   = data_root / val_rel

    img_paths = sorted(list(val_dir.glob("*.jpg")) +
                       list(val_dir.glob("*.png")))

    if len(img_paths) == 0:
        print(f"  [WARN] No images found in {val_dir}.")
        return {"mean_ms": 0, "std_ms": 0, "min_ms": 0, "max_ms": 0}

    # Limit to requested count
    img_paths = img_paths[:args.num_timing_images]
    print(f"  Timing on {len(img_paths)} images...\n")

    # Set to eval / no-gradient mode
    model.model.eval()
    device = torch.device(
        f"cuda:{args.device}" if args.device != "cpu"
        and torch.cuda.is_available() else "cpu"
    )

    latencies = []
    use_cuda = device.type == "cuda"

    # ── Warm-up ────────────────────────────────────────────────────────
    print(f"  Warming up ({n_warmup} images)...")
    for img_path in img_paths[:n_warmup]:
        _ = model.predict(
            str(img_path), device=args.device,
            verbose=False, conf=0.25
        )
    if use_cuda:
        torch.cuda.synchronize()

    # ── Timed inference ────────────────────────────────────────────────
    for img_path in tqdm(img_paths, desc="  Timing"):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model.predict(
            str(img_path), device=args.device,
            verbose=False, conf=0.25, imgsz=args.imgsz
        )

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # → ms

    latencies_np = np.array(latencies)
    timing = {
        "mean_ms": float(np.mean(latencies_np)),
        "std_ms":  float(np.std(latencies_np)),
        "min_ms":  float(np.min(latencies_np)),
        "max_ms":  float(np.max(latencies_np)),
        "n_images": len(latencies),
    }

    print(f"\n  Mean inference time : {timing['mean_ms']:.2f} ± "
          f"{timing['std_ms']:.2f} ms/image")
    print(f"  Min / Max           : {timing['min_ms']:.2f} / "
          f"{timing['max_ms']:.2f} ms")

    return timing


# ─── Save results ──────────────────────────────────────────────────────────────

def save_results(metrics: dict, timing: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    combined = {**metrics, "inference_time_ms": timing["mean_ms"], **timing}

    with open(out / "yolo_metrics.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Saved metrics → {out / 'yolo_metrics.json'}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    metrics = validate(args)
    timing  = measure_inference_time(args)
    save_results(metrics, timing, args.output_dir)

    print("\n" + "="*60)
    print("  FINAL YOLOv11 METRICS SUMMARY")
    print("="*60)
    print(f"  Precision        : {metrics['precision']*100:.1f}%")
    print(f"  Recall           : {metrics['recall']*100:.1f}%")
    print(f"  F1-score         : {metrics['f1']*100:.1f}%")
    print(f"  mAP@0.5          : {metrics['map50']*100:.1f}%")
    print(f"  mAP@0.5:0.95     : {metrics['map50_95']*100:.1f}%")
    print(f"  Inference time   : {timing['mean_ms']:.1f} ms/image")
    print("="*60 + "\n")
