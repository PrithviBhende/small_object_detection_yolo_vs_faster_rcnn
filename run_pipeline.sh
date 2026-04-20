#!/usr/bin/env bash
# =============================================================
# run_pipeline.sh
# Master script: end-to-end pipeline for the full project
# =============================================================
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Set VISDRONE_ROOT to your dataset location before running.
# =============================================================

set -e  # exit on error

# ── Configuration ──────────────────────────────────────────────
VISDRONE_ROOT="${VISDRONE_ROOT:-data/VisDrone}"
OUTPUT_ROOT="dataset/visdrone_prepared"
YOLO_WEIGHTS="yolo11n.pt"    # swap to yolo11s.pt for better accuracy
DEVICE="0"                   # GPU index; use "cpu" for CPU

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  UAV Small Object Detection — Full Pipeline                  ║"
echo "║  YOLOv11 vs Faster R-CNN on VisDrone-2019                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Dataset preparation ───────────────────────────────
echo "━━━  STEP 1: Prepare Dataset  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python dataset/prepare_visdrone.py \
    --visdrone_root "$VISDRONE_ROOT" \
    --output_root   "$OUTPUT_ROOT"
echo "  ✓ Dataset ready"

# ── Step 2: YOLOv11 Training ──────────────────────────────────
echo ""
echo "━━━  STEP 2: Train YOLOv11  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python yolo/train_yolo.py \
    --data    "$OUTPUT_ROOT/dataset.yaml" \
    --weights "$YOLO_WEIGHTS" \
    --epochs  100 \
    --batch   8 \
    --imgsz   640 \
    --device  "$DEVICE"
echo "  ✓ YOLOv11 training complete"

# ── Step 3: YOLOv11 Validation ────────────────────────────────
echo ""
echo "━━━  STEP 3: Validate YOLOv11  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python yolo/validate_yolo.py \
    --weights   "yolo/runs/visdrone_yolo11/weights/best.pt" \
    --data      "$OUTPUT_ROOT/dataset.yaml" \
    --device    "$DEVICE" \
    --output_dir "yolo/runs/visdrone_yolo11"
echo "  ✓ YOLOv11 validation complete"

# ── Step 4: Faster R-CNN Training ────────────────────────────
echo ""
echo "━━━  STEP 4: Train Faster R-CNN  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python faster_rcnn/train_faster_rcnn.py \
    --train_img_dir "$OUTPUT_ROOT/images/train" \
    --train_ann     "$OUTPUT_ROOT/annotations/train.json" \
    --val_img_dir   "$OUTPUT_ROOT/images/val" \
    --val_ann       "$OUTPUT_ROOT/annotations/val.json" \
    --epochs        36 \
    --batch         2 \
    --grad_accum    8 \
    --device        cuda
echo "  ✓ Faster R-CNN training complete"

# ── Step 5: Faster R-CNN Validation ──────────────────────────
echo ""
echo "━━━  STEP 5: Validate Faster R-CNN  ━━━━━━━━━━━━━━━━━━━━━━━━━"
python faster_rcnn/validate_faster_rcnn.py \
    --weights     "faster_rcnn/runs/visdrone_frcnn/best_model.pth" \
    --val_img_dir "$OUTPUT_ROOT/images/val" \
    --val_ann     "$OUTPUT_ROOT/annotations/val.json" \
    --device      cuda \
    --output_dir  "faster_rcnn/runs/visdrone_frcnn"
echo "  ✓ Faster R-CNN validation complete"

# ── Step 6: Unified Evaluation ───────────────────────────────
echo ""
echo "━━━  STEP 6: Unified Evaluation  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python evaluation/evaluate_both.py \
    --yolo_metrics  "yolo/runs/visdrone_yolo11/yolo_metrics.json" \
    --frcnn_metrics "faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json" \
    --output_dir    "results"
echo "  ✓ Evaluation table saved to results/"

# ── Step 7: Small Object Analysis ────────────────────────────
echo ""
echo "━━━  STEP 7: Small Object Analysis  ━━━━━━━━━━━━━━━━━━━━━━━━━"
python evaluation/small_object_analysis.py \
    --val_ann       "$OUTPUT_ROOT/annotations/val.json" \
    --yolo_metrics  "yolo/runs/visdrone_yolo11/yolo_metrics.json" \
    --frcnn_metrics "faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json" \
    --size_thresh   32 \
    --output_dir    "results"
echo "  ✓ Small object analysis saved to results/"

# ── Step 8: Visualizations ────────────────────────────────────
echo ""
echo "━━━  STEP 8: Generate Figures  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python visualization/plot_results.py \
    --yolo_metrics  "yolo/runs/visdrone_yolo11/yolo_metrics.json" \
    --frcnn_metrics "faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json" \
    --yolo_run_dir  "yolo/runs/visdrone_yolo11" \
    --frcnn_run_dir "faster_rcnn/runs/visdrone_frcnn" \
    --ann_json      "$OUTPUT_ROOT/annotations/val.json" \
    --output_dir    "results/figures"
echo "  ✓ Figures saved to results/figures/"

# ── Step 9: Detection Visualization ──────────────────────────
echo ""
echo "━━━  STEP 9: Visualize Detections  ━━━━━━━━━━━━━━━━━━━━━━━━━━"
python visualization/visualize_detections.py \
    --img_dir    "$OUTPUT_ROOT/images/val" \
    --gt_json    "$OUTPUT_ROOT/annotations/val.json" \
    --output_dir "results/detection_samples" \
    --n_samples  8
echo "  ✓ Detection samples saved to results/detection_samples/"

# ── Step 10: Final Comparison ─────────────────────────────────
echo ""
echo "━━━  STEP 10: Final Comparison Table  ━━━━━━━━━━━━━━━━━━━━━━━"
python comparison/compare_models.py \
    --yolo_metrics  "yolo/runs/visdrone_yolo11/yolo_metrics.json" \
    --frcnn_metrics "faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json" \
    --output_dir    "results"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline COMPLETE ✓                                         ║"
echo "║                                                              ║"
echo "║  Key outputs:                                               ║"
echo "║    results/figures/07_summary_figure.png  ← main figure    ║"
echo "║    results/comparison_table.csv           ← metrics CSV    ║"
echo "║    results/comparison_table.md            ← metrics MD     ║"
echo "║    results/detection_samples/             ← annotated imgs ║"
echo "╚══════════════════════════════════════════════════════════════╝"
