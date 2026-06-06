"""
Merge the synthetic dataset (output/) with the Roboflow-annotated
real-world set (roboflow_export/.../train/) into one final training-ready
dataset (dataset/), converting label formats as needed along the way.

Why conversion is needed: the Roboflow export's labels are in
polygon/segmentation format (`<class> x1 y1 x2 y2 ... xn yn`, variable
length -- likely from Roboflow's "Smart Polygon" annotation tool), but the
synthetic set and the standard YOLOv7 detection trainer both expect plain
axis-aligned bounding-box format (`<class> x_center y_center width height`,
exactly 5 values, normalized). This script converts each polygon to its
axis-aligned bounding box (min/max of its vertices) -- an exact operation,
since YOLO detection training only ever needed a box anyway.

The Roboflow export only contained a `train` split (valid/test were empty),
so this script re-splits those real-world images into train/val at the same
ratio as the synthetic set, then copies everything into:

    dataset/
      images/{train,val}/...
      labels/{train,val}/...
      data.yaml

Usage:
    python merge_datasets.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = ROOT / "output"
ROBOFLOW_DIR = ROOT / "roboflow_export" / "MTG.yolov7pytorch" / "train"
DATASET_DIR = ROOT / "dataset"

VAL_FRACTION = 0.15
CLASS_NAMES = ["Cloud", "Sephiroth", "Tifa"]

rng = np.random.default_rng(seed=42)  # deterministic split


def convert_label_line(line: str):
    """Convert one label line (bbox or polygon) to standard bbox format.
    Returns a normalized "<class> cx cy w h" string, or None if malformed."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = parts[0]
    nums = [float(p) for p in parts[1:]]

    if len(nums) == 4:
        # Already a plain bbox: cx cy w h
        cx, cy, w, h = nums
    elif len(nums) >= 6 and len(nums) % 2 == 0:
        # Polygon: x1 y1 x2 y2 ... -- take axis-aligned bbox of the vertices
        xs = nums[0::2]
        ys = nums[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
    else:
        return None

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_label_file(src_path: Path, dst_path: Path) -> bool:
    out_lines = []
    for line in src_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        converted = convert_label_line(line)
        if converted is not None:
            out_lines.append(converted)
    if not out_lines:
        return False
    dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return True


def reset_dataset_dir():
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_synthetic():
    count = 0
    for split in ("train", "val"):
        for img_path in sorted((SYNTHETIC_DIR / "images" / split).glob("*.jpg")):
            label_path = SYNTHETIC_DIR / "labels" / split / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            shutil.copy2(img_path, DATASET_DIR / "images" / split / img_path.name)
            shutil.copy2(label_path, DATASET_DIR / "labels" / split / label_path.name)
            count += 1
    return count


def copy_roboflow_converted():
    images_dir = ROBOFLOW_DIR / "images"
    labels_dir = ROBOFLOW_DIR / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"  ! Roboflow export not found at {ROBOFLOW_DIR}")
        return 0, 0

    img_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    converted = 0
    skipped = 0
    for img_path in img_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            continue

        split = "val" if rng.random() < VAL_FRACTION else "train"
        dst_label = DATASET_DIR / "labels" / split / f"{img_path.stem}.txt"
        if not convert_label_file(label_path, dst_label):
            skipped += 1
            continue

        shutil.copy2(img_path, DATASET_DIR / "images" / split / img_path.name)
        converted += 1

    return converted, skipped


def write_data_yaml():
    # Absolute paths: train.py resolves train/val relative to its own
    # working directory (not the yaml's location), so relative paths here
    # would break depending on where the trainer is invoked from.
    lines = [
        f"train: {(DATASET_DIR / 'images' / 'train').as_posix()}",
        f"val: {(DATASET_DIR / 'images' / 'val').as_posix()}",
        f"nc: {len(CLASS_NAMES)}",
        "names: [" + ", ".join(f"'{n}'" for n in CLASS_NAMES) + "]",
        "",
    ]
    (DATASET_DIR / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main():
    print(f"Synthetic set:  {SYNTHETIC_DIR}")
    print(f"Roboflow export: {ROBOFLOW_DIR}")
    print(f"Output:         {DATASET_DIR}\n")

    reset_dataset_dir()

    n_synthetic = copy_synthetic()
    print(f"Copied {n_synthetic} synthetic images+labels as-is (already bbox format).")

    n_real, n_skipped = copy_roboflow_converted()
    print(f"Converted & copied {n_real} real-world images+labels "
          f"(polygon -> bbox), skipped {n_skipped} (missing/malformed labels).")

    write_data_yaml()

    for split in ("train", "val"):
        n_img = len(list((DATASET_DIR / "images" / split).glob("*")))
        n_lbl = len(list((DATASET_DIR / "labels" / split).glob("*")))
        print(f"  {split}: {n_img} images, {n_lbl} labels")

    print(f"\nDone. Combined dataset ready at {DATASET_DIR} (data.yaml included).")


if __name__ == "__main__":
    main()
