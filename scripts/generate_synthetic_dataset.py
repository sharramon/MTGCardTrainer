"""
Composite masked card images onto varied backgrounds with randomized
geometry and lighting, auto-generating YOLO-format bounding box labels.

Because we control exactly how/where each card is placed, we know its
bounding box for free -- no manual annotation needed for this synthetic
portion of the dataset. (You'll still annotate a smaller real-world set
separately -- see frames/.)

Each output sample:
  1. Picks a random background and a random card.
  2. Scales, rotates, and perspective-warps the card (simulating the card
     lying at different angles/tilts on a surface or being held).
  3. Alpha-composites it onto the background at a random position.
  4. Optionally draws a random occluding shape over part of the card
     (simulating a finger/hand partially covering it).
  5. Applies brightness/contrast/blur/noise jitter to the whole frame
     (simulating different lighting and camera conditions).
  6. Writes the image plus a YOLO-format label
     ("<class_id> <x_center> <y_center> <width> <height>", normalized).

Output layout (matches what Roboflow/YOLOv7 expect):
  output/
    images/train/*.jpg, images/val/*.jpg
    labels/train/*.txt, labels/val/*.txt
    data.yaml

Usage:
    python generate_synthetic_dataset.py [count_per_class] [out_size]

Examples:
    python generate_synthetic_dataset.py            # 150 per class, 640px
    python generate_synthetic_dataset.py 300 640
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "processed"
BACKGROUNDS_DIR = ROOT / "backgrounds"
OUTPUT_DIR = ROOT / "output"

DEFAULT_COUNT_PER_CLASS = 150
DEFAULT_OUT_SIZE = 640
VAL_FRACTION = 0.15

# Card occupies roughly this fraction of the frame's shorter side
SCALE_RANGE = (0.18, 0.55)
ROTATION_RANGE_DEG = (-180, 180)
# Per-corner perspective jitter, as a fraction of the card's warped size
PERSPECTIVE_JITTER = 0.12
OCCLUSION_PROBABILITY = 0.25
# Minimum visible bbox size (as a fraction of frame width/height) for a
# sample to be kept -- filters out near-invisible slivers from aggressive
# edge-cropping + occlusion combinations.
MIN_BOX_FRACTION = 0.05

rng = np.random.default_rng()


def load_card_assets():
    """Returns a sorted list of (class_name, rgba_image)."""
    assets = []
    for card_dir in sorted(PROCESSED_DIR.iterdir()):
        if not card_dir.is_dir():
            continue
        pngs = list(card_dir.glob("*.png"))
        if not pngs:
            continue
        rgba = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            print(f"  ! skipping {pngs[0]} -- not a valid RGBA PNG (run mask_cards.py first)")
            continue
        assets.append((card_dir.name, rgba))
    return assets


def load_background_paths():
    paths = sorted(BACKGROUNDS_DIR.glob("*.jpg")) + sorted(BACKGROUNDS_DIR.glob("*.png"))
    return [p for p in paths if not p.name.startswith("_")]


def random_background(paths, out_size):
    path = paths[rng.integers(0, len(paths))]
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((out_size, out_size, 3), 200, dtype=np.uint8)
    # random-crop a square region, then resize -- adds positional variety
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = int(rng.integers(0, h - side + 1))
    x0 = int(rng.integers(0, w - side + 1))
    crop = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def warped_card_corners(card_w, card_h, out_size):
    """Build source/destination corner arrays for a randomized placement."""
    src = np.array([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]], dtype=np.float32)

    # Target size: scale so the card's longer side is `scale * out_size`
    scale = rng.uniform(*SCALE_RANGE)
    target_long = scale * out_size
    aspect = card_h / card_w
    if aspect >= 1.0:
        target_h = target_long
        target_w = target_long / aspect
    else:
        target_w = target_long
        target_h = target_long * aspect

    # Base rectangle centered at origin
    rect = np.array([
        [-target_w / 2, -target_h / 2],
        [target_w / 2, -target_h / 2],
        [target_w / 2, target_h / 2],
        [-target_w / 2, target_h / 2],
    ], dtype=np.float32)

    # Perspective jitter: nudge each corner independently
    jitter = (rng.uniform(-1, 1, size=(4, 2)) * PERSPECTIVE_JITTER
              * np.array([target_w, target_h]))
    rect += jitter

    # Rotation
    angle = np.radians(rng.uniform(*ROTATION_RANGE_DEG))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rect = rect @ rot.T

    # Random translation. Mostly keep the card fully visible; allow only a
    # small amount of edge-cropping (`overflow` fraction of its half-extent)
    # so we get occasional realistic partial-frame examples without ever
    # degenerating into a barely-visible sliver.
    margin = 0.03 * out_size
    overflow = 0.15
    half_extent = max(rect[:, 0].max() - rect[:, 0].min(),
                      rect[:, 1].max() - rect[:, 1].min()) / 2
    lo = half_extent * (1 - overflow) + margin
    hi = out_size - half_extent * (1 - overflow) - margin
    lo, hi = min(lo, hi), max(lo, hi)
    cx = rng.uniform(lo, hi) if hi > lo else out_size / 2
    cy = rng.uniform(lo, hi) if hi > lo else out_size / 2

    dst = rect + np.array([cx, cy], dtype=np.float32)
    return src, dst.astype(np.float32)


def composite_card(background, card_rgba):
    out_size = background.shape[0]
    card_h, card_w = card_rgba.shape[:2]
    src, dst = warped_card_corners(card_w, card_h, out_size)

    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        card_rgba, transform, (out_size, out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    alpha = warped[:, :, 3].astype(np.float32) / 255.0

    # Optional occlusion: draw a soft random ellipse ("finger/hand") into
    # the alpha mask before compositing, simulating partial coverage.
    if rng.random() < OCCLUSION_PROBABILITY:
        occluder = np.zeros((out_size, out_size), dtype=np.float32)
        ys, xs = np.where(alpha > 0.5)
        if len(xs) > 0:
            i = rng.integers(0, len(xs))
            cx_o, cy_o = int(xs[i]), int(ys[i])
            axes = (int(rng.integers(out_size // 12, out_size // 5)),
                    int(rng.integers(out_size // 12, out_size // 5)))
            cv2.ellipse(occluder, (cx_o, cy_o), axes,
                        angle=float(rng.uniform(0, 180)), startAngle=0, endAngle=360,
                        color=1.0, thickness=-1)
            occluder = cv2.GaussianBlur(occluder, (21, 21), 0)
            alpha = alpha * (1.0 - occluder)

    a3 = alpha[:, :, None]
    composited = (warped[:, :, :3].astype(np.float32) * a3
                  + background.astype(np.float32) * (1 - a3))

    return composited.astype(np.uint8), alpha, dst


def bbox_from_alpha(alpha, min_visible_alpha=0.3):
    """Axis-aligned bbox (in pixels) of the visible (non-occluded) card region."""
    ys, xs = np.where(alpha > min_visible_alpha)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def to_yolo_line(class_id, bbox_px, out_size):
    x0, y0, x1, y1 = bbox_px
    w = x1 - x0
    h = y1 - y0
    cx = x0 + w / 2
    cy = y0 + h / 2
    return (f"{class_id} "
            f"{cx / out_size:.6f} {cy / out_size:.6f} "
            f"{w / out_size:.6f} {h / out_size:.6f}")


def apply_lighting_jitter(img):
    img = img.astype(np.float32)

    brightness = rng.uniform(-40, 40)
    contrast = rng.uniform(0.75, 1.3)
    img = (img - 127.5) * contrast + 127.5 + brightness

    # Slight per-channel color cast (simulates white balance differences)
    cast = rng.uniform(-15, 15, size=3)
    img = img + cast[None, None, :]

    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    if rng.random() < 0.35:
        noise = rng.normal(0, rng.uniform(3, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def write_data_yaml(class_names):
    lines = [
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names: [" + ", ".join(f"'{n}'" for n in class_names) + "]",
        "",
    ]
    (OUTPUT_DIR / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = sys.argv[1:]
    count_per_class = int(args[0]) if len(args) > 0 else DEFAULT_COUNT_PER_CLASS
    out_size = int(args[1]) if len(args) > 1 else DEFAULT_OUT_SIZE

    assets = load_card_assets()
    if not assets:
        print(f"No masked card PNGs found in {PROCESSED_DIR}. Run mask_cards.py first.")
        return

    bg_paths = load_background_paths()
    if not bg_paths:
        print(f"No background images found in {BACKGROUNDS_DIR}. "
              f"Run download_backgrounds.py / generate_procedural_backgrounds.py first.")
        return

    class_names = [name for name, _ in assets]
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Backgrounds available: {len(bg_paths)}")
    print(f"Generating {count_per_class} images per class at {out_size}x{out_size} "
          f"(val fraction {VAL_FRACTION})")

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for class_id, (class_name, card_rgba) in enumerate(assets):
        for i in range(count_per_class):
            split = "val" if rng.random() < VAL_FRACTION else "train"

            # Retry placement a few times if it produces a degenerate
            # (near-invisible) box -- e.g. mostly cropped off-frame and
            # then also occluded. Keeps the label set free of noise.
            for _attempt in range(5):
                background = random_background(bg_paths, out_size)
                composited, alpha, _ = composite_card(background, card_rgba)
                bbox_px = bbox_from_alpha(alpha)
                if bbox_px is None:
                    continue
                x0, y0, x1, y1 = bbox_px
                if (x1 - x0) / out_size >= MIN_BOX_FRACTION and (y1 - y0) / out_size >= MIN_BOX_FRACTION:
                    break
            else:
                continue

            final = apply_lighting_jitter(composited)

            stem = f"{class_name}_{i:04d}"
            img_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
            label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(img_path), final, [cv2.IMWRITE_JPEG_QUALITY, 90])
            label_path.write_text(to_yolo_line(class_id, bbox_px, out_size) + "\n", encoding="utf-8")

            written += 1
            if written % 100 == 0:
                print(f"  ...{written} images written")

    write_data_yaml(class_names)
    print(f"Done. Wrote {written} images + labels to {OUTPUT_DIR}")
    print(f"data.yaml written with classes: {class_names}")


if __name__ == "__main__":
    main()
