"""
=============================================================
visualization/visualize_detections.py
Detection Visualizer — YOLOv11 & Faster R-CNN
=============================================================
Draws bounding boxes on val images and saves side-by-side
comparison panels (YOLO prediction | GT | Faster R-CNN prediction).

Highlights small objects with a distinct colour.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from PIL import Image

matplotlib.use("Agg")

# ─── VisDrone class colours ───────────────────────────────────────────────────

VISDRONE_CLASSES = [
    "__background__",
    "pedestrian", "people", "bicycle", "car",
    "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# Distinct palette per class
CLASS_COLORS = [
    "#000000",   # background (unused)
    "#E53935",   # pedestrian   — red
    "#D81B60",   # people       — pink
    "#8E24AA",   # bicycle      — purple
    "#1E88E5",   # car          — blue
    "#00ACC1",   # van          — cyan
    "#43A047",   # truck        — green
    "#FB8C00",   # tricycle     — orange
    "#F4511E",   # awning-tri   — deep orange
    "#FFB300",   # bus          — amber
    "#6D4C41",   # motor        — brown
]


def get_color(cat_id: int) -> str:
    return CLASS_COLORS[cat_id % len(CLASS_COLORS)]


# ─── Drawing primitives ───────────────────────────────────────────────────────

def draw_boxes(
    ax,
    boxes:      List[List[float]],
    labels:     List[int],
    scores:     Optional[List[float]] = None,
    small_thresh: int = 32,
    linewidth: float = 1.5,
):
    """
    Draw bounding boxes on a matplotlib Axes.

    Args:
        ax:           Axes object
        boxes:        list of [x1, y1, x2, y2] (pixel coords)
        labels:       list of class ids (1-indexed)
        scores:       list of confidence scores (optional)
        small_thresh: boxes with max_side < this are drawn thicker & dashed
    """
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        max_side = max(w, h)

        color = get_color(label)
        is_small = max_side < small_thresh

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=linewidth * (1.8 if is_small else 1.0),
            edgecolor=color,
            facecolor="none",
            linestyle="--" if is_small else "-",
            zorder=3,
        )
        ax.add_patch(rect)

        # Label
        cls_name = (VISDRONE_CLASSES[label]
                    if label < len(VISDRONE_CLASSES) else str(label))
        score_str = f" {scores[i]:.2f}" if scores else ""
        label_text = cls_name[:4] + score_str

        ax.text(
            x1, y1 - 2,
            label_text,
            fontsize=6,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                      edgecolor="none", alpha=0.8),
            zorder=4,
        )


# ─── Load predictions ─────────────────────────────────────────────────────────

def load_coco_preds(pred_json: str) -> Dict[int, List]:
    """Index COCO-format predictions by image_id."""
    if not pred_json or not Path(pred_json).exists():
        return {}
    with open(pred_json) as f:
        preds = json.load(f)
    indexed = {}
    for p in preds:
        iid = p["image_id"]
        if iid not in indexed:
            indexed[iid] = []
        x, y, w, h = p["bbox"]
        indexed[iid].append({
            "box":   [x, y, x+w, y+h],
            "label": p["category_id"],
            "score": p["score"],
        })
    return indexed


def load_yolo_preds_txt(label_dir: str, img_name: str,
                         img_w: int, img_h: int) -> List[Dict]:
    """
    Load YOLO-format prediction .txt file and convert to pixel boxes.
    Expects file: <label_dir>/<img_stem>.txt
    Format: class_id cx_norm cy_norm w_norm h_norm [conf]
    """
    stem = Path(img_name).stem
    txt  = Path(label_dir) / f"{stem}.txt"
    if not txt.exists():
        return []

    preds = []
    with open(txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls  = int(parts[0])
            cx, cy, nw, nh = (float(p) for p in parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0

            x1 = (cx - nw/2) * img_w
            y1 = (cy - nh/2) * img_h
            x2 = (cx + nw/2) * img_w
            y2 = (cy + nh/2) * img_h
            preds.append({
                "box":   [x1, y1, x2, y2],
                "label": cls + 1,    # YOLO is 0-indexed; COCO is 1-indexed
                "score": conf,
            })
    return preds


# ─── Main visualisation ───────────────────────────────────────────────────────

def visualize_samples(
    img_dir:       str,
    gt_json:       str,
    yolo_label_dir: Optional[str],
    frcnn_pred_json: Optional[str],
    out_dir:       str,
    n_samples:     int = 8,
    seed:          int = 42,
    small_thresh:  int = 32,
):
    """
    For each sample image, produce a 1×3 panel:
        [YOLOv11 predictions | Ground Truth | Faster R-CNN predictions]
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(gt_json) as f:
        coco_gt = json.load(f)

    id_to_info  = {img["id"]: img for img in coco_gt["images"]}
    id_to_anns  = {}
    for ann in coco_gt["annotations"]:
        iid = ann["image_id"]
        if iid not in id_to_anns:
            id_to_anns[iid] = []
        id_to_anns[iid].append(ann)

    # Load prediction indexes
    frcnn_preds = load_coco_preds(frcnn_pred_json) if frcnn_pred_json else {}

    # Sample image IDs
    random.seed(seed)
    sample_ids = random.sample(
        sorted(id_to_info.keys()),
        min(n_samples, len(id_to_info))
    )

    print(f"\n  Visualizing {len(sample_ids)} images → {out_dir}")

    for idx, img_id in enumerate(sample_ids):
        info   = id_to_info[img_id]
        img_w  = info["width"]
        img_h  = info["height"]
        fname  = info["file_name"]
        img_path = Path(img_dir) / fname

        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")

        # ── Ground truth ──────────────────────────────────────────────
        gt_boxes  = []
        gt_labels = []
        for ann in id_to_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x+w, y+h])
            gt_labels.append(ann["category_id"])

        # ── YOLO predictions ──────────────────────────────────────────
        yolo_dets = []
        if yolo_label_dir:
            yolo_dets = load_yolo_preds_txt(
                yolo_label_dir, fname, img_w, img_h
            )

        # ── Faster R-CNN predictions ──────────────────────────────────
        frcnn_dets = frcnn_preds.get(img_id, [])

        # ── Plot ──────────────────────────────────────────────────────
        has_yolo  = len(yolo_dets)  > 0
        has_frcnn = len(frcnn_dets) > 0
        n_panels  = 1 + int(has_yolo) + int(has_frcnn)

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        ax_idx = 0

        # GT panel
        axes[ax_idx].imshow(img)
        draw_boxes(axes[ax_idx], gt_boxes, gt_labels,
                   small_thresh=small_thresh)
        n_small_gt = sum(
            1 for b in gt_boxes if max(b[2]-b[0], b[3]-b[1]) < small_thresh
        )
        axes[ax_idx].set_title(
            f"Ground Truth  |  {len(gt_boxes)} objects  "
            f"({n_small_gt} small)",
            fontsize=10, fontweight="bold"
        )
        axes[ax_idx].axis("off")
        ax_idx += 1

        # YOLO panel
        if has_yolo:
            axes[ax_idx].imshow(img)
            draw_boxes(
                axes[ax_idx],
                [d["box"] for d in yolo_dets],
                [d["label"] for d in yolo_dets],
                scores=[d["score"] for d in yolo_dets],
                small_thresh=small_thresh,
                linewidth=1.2,
            )
            axes[ax_idx].set_title(
                f"YOLOv11  |  {len(yolo_dets)} detections",
                fontsize=10, fontweight="bold", color="#1565C0"
            )
            axes[ax_idx].axis("off")
            ax_idx += 1

        # Faster R-CNN panel
        if has_frcnn:
            axes[ax_idx].imshow(img)
            draw_boxes(
                axes[ax_idx],
                [d["box"]   for d in frcnn_dets],
                [d["label"] for d in frcnn_dets],
                scores=[d["score"] for d in frcnn_dets],
                small_thresh=small_thresh,
                linewidth=1.2,
            )
            axes[ax_idx].set_title(
                f"Faster R-CNN  |  {len(frcnn_dets)} detections",
                fontsize=10, fontweight="bold", color="#B71C1C"
            )
            axes[ax_idx].axis("off")

        # Legend for small objects
        legend_handles = [
            mpatches.Patch(facecolor="none", edgecolor="black",
                           linestyle="--", linewidth=2,
                           label=f"Small object (< {small_thresh}px)"),
            mpatches.Patch(facecolor="none", edgecolor="black",
                           linestyle="-",  linewidth=1.5,
                           label="Normal object"),
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=2, fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        img_stem = Path(fname).stem
        out_path = Path(out_dir) / f"sample_{idx:02d}_{img_stem}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out_path.name}")

    print(f"\n  Detection visualizations complete → {out_dir}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize YOLOv11 and Faster R-CNN detections"
    )
    p.add_argument("--img_dir",
                   default="dataset/visdrone_prepared/images/val")
    p.add_argument("--gt_json",
                   default="dataset/visdrone_prepared/annotations/val.json")
    p.add_argument("--yolo_label_dir",
                   default=None,
                   help="Directory with YOLO .txt prediction files")
    p.add_argument("--frcnn_pred_json",
                   default=None,
                   help="COCO JSON with Faster R-CNN predictions")
    p.add_argument("--output_dir",
                   default="results/detection_samples")
    p.add_argument("--n_samples",  type=int, default=8)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--small_thresh", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_samples(
        img_dir=args.img_dir,
        gt_json=args.gt_json,
        yolo_label_dir=args.yolo_label_dir,
        frcnn_pred_json=args.frcnn_pred_json,
        out_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        small_thresh=args.small_thresh,
    )
