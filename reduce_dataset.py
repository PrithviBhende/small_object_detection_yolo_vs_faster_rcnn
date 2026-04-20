"""
reduce_dataset.py
=================
Copies a random subset of VisDrone images + annotations
from the full dataset into a new folder.

Usage:
    python reduce_dataset.py

Edit the CONFIG block below to change paths or counts.
"""

import os
import random
import shutil
from pathlib import Path


# ─── CONFIG — edit these values ───────────────────────────────────────────────

# Where your FULL dataset currently lives
FULL_DATA_ROOT = r"C:\Users\prith\Documents\My Stuff\Projects\uav_detection_project\project\data1\VisDrone"

# Where the REDUCED dataset will be saved (will be created automatically)
OUT_DATA_ROOT  = r"C:\Users\prith\Documents\My Stuff\Projects\uav_detection_project\project\data\VisDrone"

# How many images you want
TRAIN_COUNT = 3000   # recommended sweet spot
VAL_COUNT   = 500    # recommended sweet spot

# Reproducibility — keep this fixed so your results are reproducible
SEED = 42

# ──────────────────────────────────────────────────────────────────────────────


def copy_subset(
    full_split_dir: Path,
    out_split_dir:  Path,
    n:              int,
    seed:           int,
    split_name:     str,
):
    """
    Randomly samples n images from full_split_dir/images/
    and copies both the image and its annotation file to out_split_dir.
    """
    src_img_dir = full_split_dir / "images"
    src_ann_dir = full_split_dir / "annotations"

    dst_img_dir = out_split_dir / "images"
    dst_ann_dir = out_split_dir / "annotations"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)

    # Collect all available jpg images
    all_images = sorted(src_img_dir.glob("*.jpg"))
    total_available = len(all_images)

    if total_available == 0:
        print(f"  [ERROR] No .jpg images found in {src_img_dir}")
        return

    if n > total_available:
        print(f"  [WARN] Requested {n} but only {total_available} available.")
        print(f"         Using all {total_available} images for {split_name}.")
        n = total_available

    # Reproducible random sample
    random.seed(seed)
    selected = random.sample(all_images, n)

    print(f"\n  [{split_name.upper()}]")
    print(f"  Available : {total_available} images")
    print(f"  Selected  : {n} images")
    print(f"  Copying   ...")

    missing_anns = 0

    for img_path in selected:
        stem = img_path.stem
        ann_path = src_ann_dir / f"{stem}.txt"

        # Copy image
        shutil.copy2(img_path, dst_img_dir / img_path.name)

        # Copy annotation (warn if missing but continue)
        if ann_path.exists():
            shutil.copy2(ann_path, dst_ann_dir / ann_path.name)
        else:
            missing_anns += 1
            # Write empty annotation file so downstream scripts don't break
            (dst_ann_dir / f"{stem}.txt").touch()

    print(f"  Done. ({missing_anns} missing annotation files — empty .txt written)")
    print(f"  Output → {out_split_dir}")


def verify_output(out_split_dir: Path, expected_n: int):
    """Quick sanity check on the output folder."""
    img_count = len(list((out_split_dir / "images").glob("*.jpg")))
    ann_count = len(list((out_split_dir / "annotations").glob("*.txt")))

    status = "✓" if img_count == expected_n else "✗"
    print(f"\n  Verification {status}")
    print(f"    Images      : {img_count}  (expected {expected_n})")
    print(f"    Annotations : {ann_count}")


def main():
    full_root = Path(FULL_DATA_ROOT)
    out_root  = Path(OUT_DATA_ROOT)

    print("=" * 60)
    print("  VisDrone Dataset Reducer")
    print("=" * 60)
    print(f"  Source : {full_root}")
    print(f"  Output : {out_root}")
    print(f"  Train  : {TRAIN_COUNT} images")
    print(f"  Val    : {VAL_COUNT} images")
    print(f"  Seed   : {SEED}")
    print("=" * 60)

    # Verify source exists
    if not full_root.exists():
        print(f"\n  [ERROR] Source folder not found:\n  {full_root}")
        print("  Please check the FULL_DATA_ROOT path in the CONFIG block.")
        return

    splits = {
        "train": (
            full_root / "VisDrone2019-DET-train",
            out_root  / "VisDrone2019-DET-train",
            TRAIN_COUNT,
        ),
        "val": (
            full_root / "VisDrone2019-DET-val",
            out_root  / "VisDrone2019-DET-val",
            VAL_COUNT,
        ),
    }

    for split_name, (src_dir, dst_dir, count) in splits.items():
        if not src_dir.exists():
            print(f"\n  [ERROR] Split folder not found: {src_dir}")
            continue

        copy_subset(src_dir, dst_dir, count, SEED, split_name)
        verify_output(dst_dir, count)

    print("\n" + "=" * 60)
    print("  Reduction complete!")
    print(f"  Your reduced dataset is at:")
    print(f"  {out_root}")
    print("=" * 60)
    print("""
  Next step — run the preparation script:

      python dataset/prepare_visdrone.py \\
          --visdrone_root "data\\VisDrone" \\
          --output_root   "dataset\\visdrone_prepared"
""")


if __name__ == "__main__":
    main()
