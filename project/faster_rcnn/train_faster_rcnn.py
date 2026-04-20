"""
=============================================================
faster_rcnn/train_faster_rcnn.py
Faster R-CNN Training Pipeline on VisDrone-2019
=============================================================
Follows the 2× schedule (36 epochs) from the Detectron2
baseline with modifications for 4 GB VRAM:
  - batch_size = 2  (vs 16 in original)
  - gradient accumulation × 8  → effective batch = 16
  - mixed precision (torch.cuda.amp)
  - frozen BN in backbone to save memory
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from faster_rcnn.visdrone_dataset import (
    VisDroneDetectionDataset, VisDroneTransform, collate_fn
)
from faster_rcnn.model import build_faster_rcnn


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Faster R-CNN on VisDrone-2019"
    )
    p.add_argument("--train_img_dir",
                   default="dataset/visdrone_prepared/images/train")
    p.add_argument("--train_ann",
                   default="dataset/visdrone_prepared/annotations/train.json")
    p.add_argument("--val_img_dir",
                   default="dataset/visdrone_prepared/images/val")
    p.add_argument("--val_ann",
                   default="dataset/visdrone_prepared/annotations/val.json")
    p.add_argument("--epochs",    type=int,   default=36)
    p.add_argument("--batch",     type=int,   default=2,
                   help="Batch size — keep 2 for 4 GB VRAM")
    p.add_argument("--grad_accum", type=int,  default=8,
                   help="Gradient accumulation steps (effective batch = batch × grad_accum)")
    p.add_argument("--lr",        type=float, default=0.02,
                   help="Base LR (will be scaled by batch×accum / 16)")
    p.add_argument("--workers",   type=int,   default=4)
    p.add_argument("--output_dir",default="faster_rcnn/runs/visdrone_frcnn")
    p.add_argument("--resume",    type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device",    type=str,   default="cuda")
    return p.parse_args()


# ─── LR schedule: linear warmup + step decay ─────────────────────────────────

class WarmupMultiStepLR:
    """
    Warmup for first 500 iterations, then step decay.
    Steps at epochs 24 and 33 (following Detectron2 2× schedule).
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma: float = 0.1,
        warmup_iters: int = 500,
        warmup_factor: float = 1.0 / 1000,
    ):
        self.optimizer     = optimizer
        self.milestones    = sorted(milestones)
        self.gamma         = gamma
        self.warmup_iters  = warmup_iters
        self.warmup_factor = warmup_factor
        self._step         = 0
        self._base_lrs     = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self._step += 1
        factor = self._get_factor()
        for lr0, group in zip(self._base_lrs, self.optimizer.param_groups):
            group["lr"] = lr0 * factor

    def _get_factor(self):
        # Warmup phase
        if self._step < self.warmup_iters:
            alpha = self._step / self.warmup_iters
            return self.warmup_factor * (1 - alpha) + alpha

        # Step decay at milestones (in global iteration count)
        factor = 1.0
        for m in self.milestones:
            if self._step >= m:
                factor *= self.gamma
        return factor


# ─── Training epoch ───────────────────────────────────────────────────────────

def train_one_epoch(
    model, optimizer, data_loader,
    device, scaler, scheduler,
    grad_accum: int, epoch: int
) -> dict:
    """
    Run one training epoch.

    Returns dict with mean losses.
    """
    model.train()

    total_loss      = 0.0
    loss_cls        = 0.0
    loss_box_reg    = 0.0
    loss_objectness = 0.0
    loss_rpn_box    = 0.0
    n_batches       = 0

    optimizer.zero_grad()

    for i, (images, targets) in enumerate(data_loader):
        # Move to device
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        # ── Forward with mixed precision ───────────────────────────────
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            # Normalize by accumulation steps
            losses_scaled = losses / grad_accum

        # ── Backward ──────────────────────────────────────────────────
        scaler.scale(losses_scaled).backward()

        # ── Optimizer step every grad_accum mini-batches ───────────────
        if (i + 1) % grad_accum == 0:
            # Gradient clipping prevents exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # ── Accumulate losses for logging ──────────────────────────────
        total_loss      += losses.item()
        loss_cls        += loss_dict.get("loss_classifier",   0)
        loss_box_reg    += loss_dict.get("loss_box_reg",      0)
        loss_objectness += loss_dict.get("loss_objectness",   0)
        loss_rpn_box    += loss_dict.get("loss_rpn_box_reg",  0)
        n_batches       += 1

        # ── Progress print ─────────────────────────────────────────────
        if (i + 1) % 50 == 0 or (i + 1) == len(data_loader):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    Ep {epoch:3d} [{i+1:4d}/{len(data_loader)}] "
                  f"loss={total_loss/n_batches:.4f}  "
                  f"cls={loss_cls/n_batches:.3f}  "
                  f"box={loss_box_reg/n_batches:.3f}  "
                  f"rpn={loss_objectness/n_batches:.3f}  "
                  f"lr={lr_now:.5f}")

    return {
        "loss":         total_loss      / max(n_batches, 1),
        "loss_cls":     loss_cls        / max(n_batches, 1),
        "loss_box_reg": loss_box_reg    / max(n_batches, 1),
        "loss_rpn":     loss_objectness / max(n_batches, 1),
    }


# ─── Main training loop ───────────────────────────────────────────────────────

def train(args):
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & loaders ────────────────────────────────────────────
    print("\n  Loading datasets...")
    train_ds = VisDroneDetectionDataset(
        img_dir=args.train_img_dir,
        ann_json=args.train_ann,
        transform=VisDroneTransform(train=True),
    )
    val_ds = VisDroneDetectionDataset(
        img_dir=args.val_img_dir,
        ann_json=args.val_ann,
        transform=VisDroneTransform(train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,              # val always batch=1 for mAP
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"  Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} images")

    # ── Model ─────────────────────────────────────────────────────────
    print("\n  Building Faster R-CNN (ResNet-50 + FPN)...")
    model = build_faster_rcnn(
        num_classes=11,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"  Total params   : {total_params:,}")
    print(f"  Trainable      : {train_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────
    # Scale LR linearly with effective batch size (linear scaling rule)
    effective_batch = args.batch * args.grad_accum
    lr = args.lr * effective_batch / 16.0
    print(f"\n  Effective batch: {effective_batch}")
    print(f"  Scaled LR      : {lr:.5f}")

    # Separate backbone (lower LR) from heads (full LR)
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params,     "lr": lr},
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Compute iteration milestones for step decay
    iters_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    milestone_epochs = [24, 33]    # Detectron2 2× schedule
    milestones = [e * iters_per_epoch for e in milestone_epochs]
    warmup_iters = 500

    scheduler = WarmupMultiStepLR(
        optimizer, milestones=milestones,
        warmup_iters=warmup_iters
    )

    scaler = GradScaler()

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 1
    history = {"train_loss": [], "epochs": []}

    if args.resume and Path(args.resume).exists():
        print(f"\n  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler._step = ckpt.get("scheduler_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("history", history)
        print(f"  Resumed at epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Starting training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    t_total_start = time.time()
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()
        print(f"\n  ── Epoch {epoch}/{args.epochs} ──────────────────────")

        ep_metrics = train_one_epoch(
            model, optimizer, train_loader,
            device, scaler, scheduler,
            grad_accum=args.grad_accum,
            epoch=epoch,
        )

        ep_time = time.time() - t_ep
        print(f"  → Epoch {epoch} done in {ep_time:.1f}s | "
              f"loss={ep_metrics['loss']:.4f}")

        # ── Update history ─────────────────────────────────────────────
        history["epochs"].append(epoch)
        history["train_loss"].append(ep_metrics["loss"])

        # ── Save checkpoint every 3 epochs ────────────────────────────
        if epoch % 3 == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"checkpoint_ep{epoch:03d}.pth"
            torch.save({
                "epoch":          epoch,
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scaler":         scaler.state_dict(),
                "scheduler_step": scheduler._step,
                "loss":           ep_metrics["loss"],
                "history":        history,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        # ── Save best model ────────────────────────────────────────────
        if ep_metrics["loss"] < best_loss:
            best_loss = ep_metrics["loss"]
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"  ★ New best loss: {best_loss:.4f} — saved best_model.pth")

    # ── Final save ────────────────────────────────────────────────────
    torch.save(model.state_dict(), out_dir / "last_model.pth")
    total_time = time.time() - t_total_start
    print(f"\n  Training complete in {total_time/3600:.2f} hours.")

    # ── Save training history ──────────────────────────────────────────
    history["total_time_sec"] = total_time
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Training history → {out_dir / 'training_history.json'}")
    return model, history


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "="*60)
    print("  Faster R-CNN Training — VisDrone-2019")
    print("="*60)
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch} × {args.grad_accum} accum = "
          f"{args.batch * args.grad_accum} effective")
    print(f"  Base LR    : {args.lr}")
    print(f"  Output dir : {args.output_dir}")
    print("="*60 + "\n")

    train(args)
