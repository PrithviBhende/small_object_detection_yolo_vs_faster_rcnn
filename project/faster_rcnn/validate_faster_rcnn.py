"""
faster_rcnn/validate_faster_rcnn.py  (FIXED)
Fixes: background label skip, image_id alignment, scale correction
"""

import os, sys, json, time, argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent.parent))
from faster_rcnn.model import build_faster_rcnn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",      default="faster_rcnn/runs/visdrone_frcnn/best_model.pth")
    p.add_argument("--val_img_dir",  default="dataset/visdrone_prepared/images/val")
    p.add_argument("--val_ann",      default="dataset/visdrone_prepared/annotations/val.json")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--conf",         type=float, default=0.01)
    p.add_argument("--output_dir",   default="faster_rcnn/runs/visdrone_frcnn")
    p.add_argument("--num_timing_images", type=int, default=200)
    return p.parse_args()


class FlatValDataset(Dataset):
    def __init__(self, img_dir, ann_json):
        self.img_dir = Path(img_dir)
        with open(ann_json) as f:
            coco = json.load(f)
        self.records = []
        for img_info in coco["images"]:
            p = self.img_dir / img_info["file_name"]
            if p.exists():
                self.records.append({"image_id": img_info["id"],
                                     "file_name": img_info["file_name"],
                                     "path": str(p)})
        print(f"  FlatValDataset: {len(self.records)} images found")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        image = Image.open(rec["path"]).convert("RGB")
        w, h  = image.size
        scale = 640 / min(h, w)
        image = image.resize((int(round(w*scale)), int(round(h*scale))), Image.BILINEAR)
        t = TF.to_tensor(image)
        t = TF.normalize(t, [0.485,0.456,0.406], [0.229,0.224,0.225])
        return t, rec["image_id"], scale

    @staticmethod
    def collate(batch):
        return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


@torch.no_grad()
def collect_predictions(model, dataset, device, conf_thresh=0.01):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=FlatValDataset.collate, pin_memory=True)
    all_preds = []
    label_counts = {}
    for imgs, img_ids, scales in tqdm(loader, desc="  Inference"):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for output, img_id, scale in zip(outputs, img_ids, scales):
            boxes  = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if int(label) == 0 or score < conf_thresh:
                    continue
                x1, y1, x2, y2 = box / scale
                w = float(x2 - x1); h = float(y2 - y1)
                if w <= 0 or h <= 0: continue
                all_preds.append({"image_id": int(img_id),
                                  "category_id": int(label),
                                  "bbox": [float(x1), float(y1), w, h],
                                  "score": float(score)})
                label_counts[int(label)] = label_counts.get(int(label), 0) + 1
    print(f"\n  Total predictions (conf>{conf_thresh}): {len(all_preds)}")
    print(f"  Per-category counts: {label_counts}")
    return all_preds


def sanity_check(predictions, gt_json):
    with open(gt_json) as f: gt = json.load(f)
    gt_img_ids  = {i["id"] for i in gt["images"]}
    gt_cat_ids  = {c["id"] for c in gt["categories"]}
    pred_img_ids = {p["image_id"]    for p in predictions}
    pred_cat_ids = {p["category_id"] for p in predictions}
    print(f"\n  Sanity check:")
    print(f"    GT images: {len(gt_img_ids)}  |  Pred images: {len(pred_img_ids)}  |  Overlap: {len(gt_img_ids & pred_img_ids)}")
    print(f"    GT cats: {sorted(gt_cat_ids)}  |  Pred cats: {sorted(pred_cat_ids)}")


def evaluate_map(gt_json, predictions):
    if not predictions:
        return {"map50": 0.0, "map50_95": 0.0}
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"map50_95": float(ev.stats[0]), "map50": float(ev.stats[1])}


def compute_prf(gt_json, predictions, iou_thresh=0.5, conf_thresh=0.25):
    with open(gt_json) as f: gt = json.load(f)
    gt_by_img   = {}
    for a in gt["annotations"]: gt_by_img.setdefault(a["image_id"], []).append(a)
    pred_by_img = {}
    for p in predictions:
        if p["score"] >= conf_thresh: pred_by_img.setdefault(p["image_id"], []).append(p)
    TP = FP = FN = 0
    for img_id in set(list(gt_by_img)+list(pred_by_img)):
        gts = gt_by_img.get(img_id, [])
        prs = pred_by_img.get(img_id, [])
        gb = np.array([[a["bbox"][0],a["bbox"][1],a["bbox"][0]+a["bbox"][2],a["bbox"][1]+a["bbox"][3]] for a in gts], dtype=np.float32) if gts else np.empty((0,4))
        pb = np.array([[p["bbox"][0],p["bbox"][1],p["bbox"][0]+p["bbox"][2],p["bbox"][1]+p["bbox"][3]] for p in prs], dtype=np.float32) if prs else np.empty((0,4))
        matched = set()
        for pr in pb:
            if len(gb)==0: FP+=1; continue
            x1=np.maximum(pr[0],gb[:,0]); y1=np.maximum(pr[1],gb[:,1])
            x2=np.minimum(pr[2],gb[:,2]); y2=np.minimum(pr[3],gb[:,3])
            inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
            union=(pr[2]-pr[0])*(pr[3]-pr[1])+(gb[:,2]-gb[:,0])*(gb[:,3]-gb[:,1])-inter
            ious=inter/np.maximum(union,1e-6)
            best=int(np.argmax(ious))
            if ious[best]>=iou_thresh and best not in matched: TP+=1; matched.add(best)
            else: FP+=1
        FN += len(gb) - len(matched)
    P = TP/(TP+FP) if (TP+FP)>0 else 0.0
    R = TP/(TP+FN) if (TP+FN)>0 else 0.0
    F = 2*P*R/(P+R) if (P+R)>0 else 0.0
    return {"precision": P, "recall": R, "f1": F}


@torch.no_grad()
def measure_latency(model, dataset, device, n=200, n_warmup=10):
    model.eval()
    use_cuda = device.type == "cuda"
    indices  = list(range(min(n+n_warmup, len(dataset))))
    print(f"  Warming up ({n_warmup} imgs)...")
    for i in indices[:n_warmup]:
        img, _, _ = dataset[i]
        _ = model([img.to(device)])
    if use_cuda: torch.cuda.synchronize()
    latencies = []
    print(f"  Timing {min(n, len(dataset))} images...")
    for i in tqdm(indices[n_warmup:n_warmup+n]):
        img, _, _ = dataset[i]
        img = img.to(device)
        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model([img])
        if use_cuda: torch.cuda.synchronize()
        latencies.append((time.perf_counter()-t0)*1000)
    arr = np.array(latencies)
    return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)), "max_ms": float(np.max(arr))}


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("\n"+"="*60)
    print("  Faster R-CNN Evaluation (FIXED)")
    print("="*60)

    model = build_faster_rcnn(num_classes=11, pretrained_backbone=False)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"  Loaded: {args.weights}")

    dataset     = FlatValDataset(args.val_img_dir, args.val_ann)
    predictions = collect_predictions(model, dataset, device, args.conf)
    sanity_check(predictions, args.val_ann)

    print("\n  Computing mAP...")
    map_m  = evaluate_map(args.val_ann, predictions)
    print("\n  Computing Precision/Recall/F1...")
    prf_m  = compute_prf(args.val_ann, predictions)
    print("\n  Measuring latency...")
    timing = measure_latency(model, dataset, device, args.num_timing_images)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {**prf_m, **map_m, "inference_time_ms": timing["mean_ms"], **timing}
    with open(out / "frcnn_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n"+"="*60)
    print("  FINAL Faster R-CNN METRICS")
    print("="*60)
    print(f"  Precision      : {prf_m['precision']*100:.1f}%")
    print(f"  Recall         : {prf_m['recall']*100:.1f}%")
    print(f"  F1-score       : {prf_m['f1']*100:.1f}%")
    print(f"  mAP@0.5        : {map_m['map50']*100:.1f}%")
    print(f"  mAP@0.5:0.95   : {map_m['map50_95']*100:.1f}%")
    print(f"  Inference time : {timing['mean_ms']:.1f} ms/image")
    print("="*60)
    print(f"\n  Saved → {out / 'frcnn_metrics.json'}")


if __name__ == "__main__":
    main()
