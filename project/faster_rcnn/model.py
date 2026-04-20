"""
=============================================================
faster_rcnn/model.py
Faster R-CNN (ResNet-50 + FPN) for VisDrone Small Objects
=============================================================
Key modifications over the torchvision default:
  1. Small anchor sizes: 16, 32, 64, 128, 256
     (default is 32,64,128,256,512 — too large for UAV objects)
  2. Aspect ratios: 0.5, 1.0, 2.0 (add taller boxes for people)
  3. Higher RPN NMS topk (3000 pre, 2000 post) for dense scenes
  4. Lower detection NMS threshold (0.5) for accuracy
  5. More detections per image (300) for dense UAV scenes
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import (
    AnchorGenerator, RPNHead
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone
)
from torchvision.ops import MultiScaleRoIAlign


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_faster_rcnn(
    num_classes: int = 11,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
) -> FasterRCNN:
    """
    Build Faster R-CNN with ResNet-50 FPN backbone.

    Args:
        num_classes:               Including background (11 for VisDrone)
        pretrained_backbone:       Use ImageNet-pretrained ResNet-50
        trainable_backbone_layers: How many ResNet stages to fine-tune
                                   (0=frozen, 3=default, 5=full)
        min_size / max_size:       Image resize range at inference

    Returns:
        FasterRCNN model (ready for .train() / .eval())
    """

    # ── 1. Backbone: ResNet-50 + FPN ──────────────────────────────────
    # Returns feature maps at strides 4, 8, 16, 32, 64 (P2–P6)
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="ResNet50_Weights.IMAGENET1K_V1" if pretrained_backbone
                else None,
        trainable_layers=trainable_backbone_layers,
    )

    # ── 2. Custom anchor generator for small objects ───────────────────
    # Each FPN level gets its own anchor size list.
    # Smaller sizes (16, 32) detect tiny UAV objects.
    anchor_generator = AnchorGenerator(
        sizes=(
            (16,),    # P2 — finest features
            (32,),    # P3
            (64,),    # P4
            (128,),   # P5
            (256,),   # P6 — coarsest features
        ),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        # → 5 levels × 3 ratios = 15 anchors/location
    )

    # ── 3. ROI Align pooler ────────────────────────────────────────────
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # ── 4. Assemble Faster R-CNN ───────────────────────────────────────
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Image size range (multi-scale training done in Dataset)
        min_size=min_size,
        max_size=max_size,
        # RPN parameters — dense UAV scenes need more proposals
        rpn_pre_nms_top_n_train=3000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1500,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # ROI head parameters
        box_score_thresh=0.05,        # low threshold → compute mAP across range
        box_nms_thresh=0.5,
        box_detections_per_img=300,   # VisDrone has very dense scenes
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
    )

    # ── 5. Replace box predictor with correct class count ─────────────
    # (needed even if num_classes == default, to make it explicit)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features,
        num_classes=num_classes,
    )

    return model


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_faster_rcnn(num_classes=11)
    model.eval()

    # Dummy batch
    imgs = [torch.rand(3, 640, 800), torch.rand(3, 640, 800)]
    with torch.no_grad():
        out = model(imgs)

    print("Model forward pass OK.")
    print(f"Output keys: {list(out[0].keys())}")
    print(f"Boxes shape: {out[0]['boxes'].shape}")
    print(f"\nTotal parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
