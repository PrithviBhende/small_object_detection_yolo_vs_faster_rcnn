"""
=============================================================
faster_rcnn/visdrone_dataset.py
VisDrone-2019 PyTorch Dataset for Faster R-CNN
=============================================================
Returns images + target dicts compatible with torchvision
Faster R-CNN.

Target format (per image):
    {
      "boxes":   FloatTensor[N, 4]   ← [x1, y1, x2, y2]
      "labels":  Int64Tensor[N]      ← 1-indexed class ids
      "image_id": Int64Tensor[1]
      "area":    FloatTensor[N]
      "iscrowd": UInt8Tensor[N]
    }
"""

import os
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ─── Class metadata ───────────────────────────────────────────────────────────

VISDRONE_CLASSES = [
    "__background__",   # 0  ← torchvision expects background at 0
    "pedestrian",       # 1
    "people",           # 2
    "bicycle",          # 3
    "car",              # 4
    "van",              # 5
    "truck",            # 6
    "tricycle",         # 7
    "awning-tricycle",  # 8
    "bus",              # 9
    "motor",            # 10
]
NUM_CLASSES = len(VISDRONE_CLASSES)   # 11 (including background)


# ─── Transforms ───────────────────────────────────────────────────────────────

class VisDroneTransform:
    """
    Data augmentation pipeline for Faster R-CNN training.

    All transforms preserve bounding box coordinates.
    Multi-scale training: randomly resize the shorter side
    to one of several target sizes.
    """

    TRAIN_SCALES = [480, 512, 544, 576, 608, 640, 672]
    MAX_SIZE = 1333   # cap long side (memory budget)

    def __init__(self, train: bool = True):
        self.train = train

    def __call__(
        self,
        image: Image.Image,
        target: Dict
    ) -> Tuple[torch.Tensor, Dict]:

        # ── Horizontal flip ───────────────────────────────────────────
        if self.train and torch.rand(1).item() < 0.5:
            image, target = self._hflip(image, target)

        # ── Multi-scale resize ────────────────────────────────────────
        if self.train:
            scale = self.TRAIN_SCALES[
                torch.randint(len(self.TRAIN_SCALES), (1,)).item()
            ]
        else:
            scale = 640   # fixed for validation

        image, target = self._resize(image, target, scale)

        # ── To tensor & normalize ─────────────────────────────────────
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return img_tensor, target

    # ── Helpers ───────────────────────────────────────────────────────

    def _hflip(self, image, target):
        w, _ = image.size
        image = TF.hflip(image)
        boxes = target["boxes"].clone()
        boxes[:, 0], boxes[:, 2] = w - target["boxes"][:, 2], \
                                    w - target["boxes"][:, 0]
        target["boxes"] = boxes
        return image, target

    def _resize(self, image, target, min_size):
        w, h = image.size
        scale = min_size / min(h, w)

        # Cap long side
        if max(h, w) * scale > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(h, w)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = TF.resize(image, [new_h, new_w])

        if target["boxes"].numel() > 0:
            target["boxes"] = target["boxes"] * scale

        return image, target


# ─── Dataset class ────────────────────────────────────────────────────────────

class VisDroneDetectionDataset(Dataset):
    """
    Dataset that reads images and annotations from the COCO JSON
    produced by prepare_visdrone.py.

    Args:
        img_dir:    Folder containing .jpg images
        ann_json:   COCO-format JSON annotation file
        transform:  VisDroneTransform (or None)
        max_boxes:  Cap on max boxes per image (memory safety)
    """

    def __init__(
        self,
        img_dir: str,
        ann_json: str,
        transform: Optional[Callable] = None,
        max_boxes: int = 500,
    ):
        self.img_dir  = Path(img_dir)
        self.transform = transform
        self.max_boxes = max_boxes

        # Load COCO JSON
        print(f"  Loading annotations from {ann_json} ...")
        with open(ann_json, "r") as f:
            coco = json.load(f)

        # Index: image_id → image info
        self.images = {img["id"]: img for img in coco["images"]}

        # Index: image_id → list of annotation dicts
        self.ann_index: Dict[int, List] = {
            img["id"]: [] for img in coco["images"]
        }
        for ann in coco["annotations"]:
            self.ann_index[ann["image_id"]].append(ann)

        self.img_ids = sorted(self.images.keys())
        print(f"  Found {len(self.img_ids)} images, "
              f"{len(coco['annotations'])} annotations.\n")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info["file_name"]

        # ── Load image ────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")

        # ── Build target ──────────────────────────────────────────────
        anns = self.ann_index[img_id]

        # Cap to max_boxes (extremely dense images can hurt memory)
        if len(anns) > self.max_boxes:
            anns = anns[:self.max_boxes]

        boxes  = []
        labels = []
        areas  = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # Convert [x, y, w, h] → [x1, y1, x2, y2]
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Sanity check: positive area
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])   # 1-indexed (background=0)
            areas.append(ann["area"])

        if boxes:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas_t  = torch.as_tensor(areas,  dtype=torch.float32)
        else:
            # Empty image — return empty tensors
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
            areas_t  = torch.zeros((0,),   dtype=torch.float32)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area":     areas_t,
            "iscrowd":  torch.zeros(len(labels_t), dtype=torch.uint8),
        }

        # ── Apply transforms ──────────────────────────────────────────
        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = TF.to_tensor(image)

        return image, target


# ─── Collate function ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Custom collate: images and targets stay as lists because
    each image can have a different number of boxes.
    Torchvision's detection models expect this format.
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ─── Quick sanity test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--img_dir",  default="dataset/visdrone_prepared/images/val")
    p.add_argument("--ann_json", default="dataset/visdrone_prepared/annotations/val.json")
    args = p.parse_args()

    transform = VisDroneTransform(train=False)
    ds = VisDroneDetectionDataset(args.img_dir, args.ann_json, transform)

    print(f"Dataset size: {len(ds)}")
    img, tgt = ds[0]
    print(f"Image shape : {img.shape}")
    print(f"Boxes       : {tgt['boxes'].shape}")
    print(f"Labels      : {tgt['labels']}")
    print("\nDataset loader OK.")
