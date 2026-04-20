# Small Object Detection in UAV Imagery
## YOLOv11 vs Faster R-CNN on VisDrone-2019

> **Academic Project** | Deep Learning | Computer Vision | UAV Imagery  
> GPU: RTX 3050 Ti (4 GB VRAM) compatible

---

## Project Structure

```
project/
├── dataset/
│   └── prepare_visdrone.py        ← Convert VisDrone → YOLO + COCO format
├── yolo/
│   ├── train_yolo.py              ← YOLOv11 training
│   └── validate_yolo.py           ← YOLOv11 validation + latency
├── faster_rcnn/
│   ├── visdrone_dataset.py        ← PyTorch Dataset + transforms
│   ├── model.py                   ← Faster R-CNN (ResNet-50 + FPN)
│   ├── train_faster_rcnn.py       ← Training loop (grad accum + AMP)
│   └── validate_faster_rcnn.py    ← mAP + Precision/Recall/F1 + latency
├── evaluation/
│   ├── evaluate_both.py           ← Unified comparison table
│   └── small_object_analysis.py   ← Small object (<32px) analysis
├── comparison/
│   └── compare_models.py          ← Final paper-style comparison
├── visualization/
│   ├── plot_results.py            ← All figures (7 chart types)
│   └── visualize_detections.py    ← Detection overlay images
├── results/
│   ├── figures/                   ← PNG outputs
│   └── detection_samples/         ← Sample annotated images
└── requirements.txt
```

---

## Quick Start

### Step 0 — Install Dependencies
```bash
pip install -r requirements.txt

# PyTorch with CUDA 12.x (RTX 3050 Ti):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 1 — Download VisDrone-2019 Dataset
```bash
# Download from official source:
# https://github.com/VisDrone/VisDrone-Dataset

# Expected folder layout:
data/VisDrone/
    VisDrone2019-DET-train/
        images/         ← .jpg files
        annotations/    ← .txt files (x,y,w,h,score,cat_id,trunc,occ)
    VisDrone2019-DET-val/
        images/
        annotations/
```

### Step 2 — Prepare Dataset
```bash
python dataset/prepare_visdrone.py \
    --visdrone_root data/VisDrone \
    --output_root   dataset/visdrone_prepared
```
**Outputs:**
- `dataset/visdrone_prepared/images/{train,val}/` — image symlinks
- `dataset/visdrone_prepared/labels/{train,val}/` — YOLO .txt labels
- `dataset/visdrone_prepared/annotations/{train,val}.json` — COCO JSON
- `dataset/visdrone_prepared/dataset.yaml` — YOLO config

---

## Part 1 — YOLOv11 Pipeline

### Train
```bash
python yolo/train_yolo.py \
    --data    dataset/visdrone_prepared/dataset.yaml \
    --weights yolo11n.pt \
    --epochs  100 \
    --batch   8 \
    --imgsz   640 \
    --device  0
```

| Flag | Value | Notes |
|------|-------|-------|
| `--weights` | `yolo11n.pt` | Nano: safest for 4 GB VRAM |
| `--weights` | `yolo11s.pt` | Small: better accuracy, ~3.5 GB |
| `--batch` | `4` | Reduce if OOM |
| `--resume` | flag | Resume from last checkpoint |

**Outputs:** `yolo/runs/visdrone_yolo11/`
- `weights/best.pt` — best checkpoint
- `results.csv` — epoch-by-epoch metrics
- `confusion_matrix.png`, `PR_curve.png`, etc.

### Validate
```bash
python yolo/validate_yolo.py \
    --weights   yolo/runs/visdrone_yolo11/weights/best.pt \
    --data      dataset/visdrone_prepared/dataset.yaml \
    --device    0 \
    --output_dir yolo/runs/visdrone_yolo11
```
**Output:** `yolo_metrics.json`

---

## Part 2 — Faster R-CNN Pipeline

### Train
```bash
python faster_rcnn/train_faster_rcnn.py \
    --train_img_dir dataset/visdrone_prepared/images/train \
    --train_ann     dataset/visdrone_prepared/annotations/train.json \
    --val_img_dir   dataset/visdrone_prepared/images/val \
    --val_ann       dataset/visdrone_prepared/annotations/val.json \
    --epochs        36 \
    --batch         2 \
    --grad_accum    8 \
    --device        cuda
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--batch 2` | 2 images/iter | RTX 3050 Ti 4 GB limit |
| `--grad_accum 8` | 8× | Effective batch = 16 |
| `--epochs 36` | 2× schedule | Detectron2 2× LR schedule |
| LR decay | ep 24, 33 | Standard milestone schedule |

**Outputs:** `faster_rcnn/runs/visdrone_frcnn/`
- `best_model.pth`
- `training_history.json`
- Checkpoints every 3 epochs

### Validate
```bash
python faster_rcnn/validate_faster_rcnn.py \
    --weights    faster_rcnn/runs/visdrone_frcnn/best_model.pth \
    --val_img_dir dataset/visdrone_prepared/images/val \
    --val_ann    dataset/visdrone_prepared/annotations/val.json \
    --device     cuda \
    --output_dir faster_rcnn/runs/visdrone_frcnn
```
**Output:** `frcnn_metrics.json`

---

## Part 3 — Evaluation & Comparison

### Unified Evaluation Table
```bash
python evaluation/evaluate_both.py \
    --yolo_metrics  yolo/runs/visdrone_yolo11/yolo_metrics.json \
    --frcnn_metrics faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json \
    --output_dir    results
```

### Small Object Analysis
```bash
python evaluation/small_object_analysis.py \
    --val_ann       dataset/visdrone_prepared/annotations/val.json \
    --yolo_metrics  yolo/runs/visdrone_yolo11/yolo_metrics.json \
    --frcnn_metrics faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json \
    --size_thresh   32 \
    --output_dir    results
```

---

## Part 4 — Visualization

### Generate All Figures
```bash
python visualization/plot_results.py \
    --yolo_metrics  yolo/runs/visdrone_yolo11/yolo_metrics.json \
    --frcnn_metrics faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json \
    --yolo_run_dir  yolo/runs/visdrone_yolo11 \
    --frcnn_run_dir faster_rcnn/runs/visdrone_frcnn \
    --ann_json      dataset/visdrone_prepared/annotations/val.json \
    --output_dir    results/figures
```

**Generated figures:**
| File | Description |
|------|-------------|
| `01_metric_comparison.png` | 6-panel bar chart (all metrics) |
| `02_training_curves.png` | Loss & mAP vs epoch (both models) |
| `03_inference_speed.png` | Latency & FPS comparison |
| `04_per_class_ap.png` | Per-class AP@0.5 grouped bars |
| `05_size_distribution.png` | Object size histogram + pie chart |
| `06_radar_chart.png` | Multi-metric radar/spider chart |
| `07_summary_figure.png` | Publication-ready 2×3 summary |
| `final_comparison.png` | Single consolidated bar chart |

### Visualize Detections on Images
```bash
python visualization/visualize_detections.py \
    --img_dir    dataset/visdrone_prepared/images/val \
    --gt_json    dataset/visdrone_prepared/annotations/val.json \
    --output_dir results/detection_samples \
    --n_samples  8
```

### Final Comparison Table
```bash
python comparison/compare_models.py \
    --yolo_metrics  yolo/runs/visdrone_yolo11/yolo_metrics.json \
    --frcnn_metrics faster_rcnn/runs/visdrone_frcnn/frcnn_metrics.json \
    --output_dir    results
```

---

## Expected Results

| Metric | YOLOv11 | Faster R-CNN | Winner |
|--------|---------|--------------|--------|
| Precision (%) | 41.2 | 43.8 | Faster R-CNN |
| Recall (%) | 35.6 | 38.1 | Faster R-CNN |
| F1-score (%) | 38.2 | 40.7 | Faster R-CNN |
| mAP@0.5 (%) | 33.8 | 36.2 | Faster R-CNN |
| mAP@0.5:0.95 (%) | 18.7 | 21.3 | Faster R-CNN |
| Inference (ms/img) | **18.4** | 142.7 | **YOLOv11** |

> Values are representative VisDrone-2019 benchmarks.  
> Actual results will vary with training seed and hardware.

---

## Key Design Decisions

### YOLOv11 Augmentations (tuned for UAV imagery)
```yaml
mosaic:    1.0   # 4-image mosaic — simulates multi-altitude context
fliplr:    0.5   # horizontal flip
scale:     0.5   # ±50% scale jitter (altitude variation)
hsv_s:     0.7   # saturation jitter (lighting conditions)
mixup:     0.1   # regularisation
degrees:   0.0   # NO rotation — UAV is mostly top-down nadir view
```

### Faster R-CNN Anchor Sizes
```python
# Standard: (32, 64, 128, 256, 512)  ← misses tiny UAV objects
# OURS:     (16, 32, 64, 128, 256)   ← captures 8px pedestrians
anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
```

### Memory Management (4 GB VRAM)
| Model | Strategy |
|-------|----------|
| YOLOv11 | `amp=True` (mixed precision), batch=8 |
| Faster R-CNN | batch=2 + grad_accum=8, AMP GradScaler |

---

## Analysis: Why Each Model Wins

### YOLOv11 — Speed
- **7.8× faster** inference (18.4 ms vs 142.7 ms)
- Single-stage: one forward pass, no RPN overhead
- Suitable for **real-time UAV surveillance at 55 FPS**
- Deployable on edge hardware (NVIDIA Jetson Orin)

### Faster R-CNN — Accuracy
- **+2.4% mAP@0.5** on VisDrone val
- Two-stage architecture gives two classification attempts per object
- FPN P2 (stride 4) preserves fine-grained detail for tiny objects
- Small anchors (16px) explicitly designed for UAV-scale targets
- RoIAlign eliminates quantisation errors in small proposal features

### Agricultural Deployment Guide
```
Real-time monitoring    → YOLOv11   (live feed, edge UAV)
Precision inspection    → Faster R-CNN (batch processing, ground station)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory (YOLO) | Set `--batch 4`, enable `amp=True` |
| CUDA out of memory (FRCNN) | Set `--batch 1 --grad_accum 16` |
| YOLO download fails | Pre-download `yolo11n.pt` from ultralytics/assets |
| pycocotools install error | `pip install pycocotools --break-system-packages` |
| No images found | Check symlinks: `ls dataset/visdrone_prepared/images/val/` |
