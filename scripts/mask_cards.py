"""
Convert flat-scan card JPGs into alpha-masked PNGs.

The reference card images have rounded corners but were exported as
rectangular JPGs, so the area outside the rounded corners is filled with
solid white (JPGs can't store transparency). If we composite these directly
onto backgrounds for synthetic training data, the model would learn the
white corner triangles as part of the card's appearance -- a visual
artifact that doesn't exist on a real card sitting on a real surface.

This script flood-fills the white corner regions starting from the four
image corners and converts them to a transparent alpha channel, leaving
only the true rounded-rectangle card art. Run it once per new reference
image; outputs go to processed/<CardName>/<CardName>.png.

Usage:
    python mask_cards.py
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "Cards"
PROCESSED_DIR = ROOT / "processed"

# How close to pure white (0-255) a pixel must be to seed the flood fill,
# and how much neighboring pixels may drift (handles JPEG compression noise).
WHITE_SEED_THRESHOLD = 235
FLOOD_TOLERANCE = 18
FEATHER_PX = 3  # blur radius applied to the alpha edge for smoother compositing


def mask_white_corners(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    work = bgr.copy()

    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    tol = (FLOOD_TOLERANCE,) * 3
    for x, y in corners:
        b, g, r = work[y, x]
        if b >= WHITE_SEED_THRESHOLD and g >= WHITE_SEED_THRESHOLD and r >= WHITE_SEED_THRESHOLD:
            cv2.floodFill(
                work,
                flood_mask,
                seedPoint=(x, y),
                newVal=(255, 255, 255),
                loDiff=tol,
                upDiff=tol,
                flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
            )

    background = flood_mask[1:-1, 1:-1].astype(bool)
    alpha = np.where(background, 0, 255).astype(np.uint8)
    if FEATHER_PX > 0:
        k = FEATHER_PX * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def main():
    jpgs = sorted(CARDS_DIR.glob("*/*.jpg")) + sorted(CARDS_DIR.glob("*/*.jpeg"))
    if not jpgs:
        print(f"No card JPGs found under {CARDS_DIR}")
        return

    for src in jpgs:
        card_name = src.parent.name
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  ! could not read {src}")
            continue

        rgba = mask_white_corners(bgr)

        out_dir = PROCESSED_DIR / card_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{src.stem}.png"
        cv2.imwrite(str(out_path), rgba)

        transparent_frac = float((rgba[:, :, 3] == 0).mean())
        print(f"{src.relative_to(ROOT)} -> {out_path.relative_to(ROOT)}  "
              f"(masked {transparent_frac:.1%} of pixels transparent)")


if __name__ == "__main__":
    main()
