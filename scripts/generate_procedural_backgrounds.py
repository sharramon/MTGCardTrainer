"""
Generate cheap, locally-made filler backgrounds (solid colors, gradients,
and simple noise/texture patterns) with zero network dependency.

These pad out background variety alongside the downloaded stock photos --
useful for teaching the detector that the card can appear over almost
anything, including plain surfaces (e.g. a solid-colored table or sleeve).

Usage:
    python generate_procedural_backgrounds.py [count] [width] [height]
"""

import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "backgrounds"

DEFAULT_COUNT = 60
DEFAULT_SIZE = (800, 600)

rng = np.random.default_rng()


def random_color():
    return rng.integers(0, 256, size=3, dtype=np.uint8)


def make_solid(w, h):
    color = random_color()
    return np.full((h, w, 3), color, dtype=np.uint8)


def make_gradient(w, h):
    c1 = random_color().astype(np.float32)
    c2 = random_color().astype(np.float32)
    horizontal = rng.random() < 0.5
    t = np.linspace(0.0, 1.0, w if horizontal else h, dtype=np.float32)[:, None]
    grad = c1 * (1 - t) + c2 * t
    if horizontal:
        img = np.tile(grad[None, :, :], (h, 1, 1))
    else:
        img = np.tile(grad[:, None, :], (1, w, 1))
    return img.astype(np.uint8)


def make_noise_texture(w, h):
    base = random_color().astype(np.int16)
    noise = rng.integers(-40, 41, size=(h, w, 3))
    img = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    # smooth slightly so it reads as a texture, not pure static
    k = int(rng.choice([3, 5, 7]))
    return cv2.GaussianBlur(img, (k, k), 0)


def make_stripes(w, h):
    c1 = random_color()
    c2 = random_color()
    stripe_w = int(rng.integers(10, 80))
    horizontal = rng.random() < 0.5
    img = np.empty((h, w, 3), dtype=np.uint8)
    if horizontal:
        for y in range(h):
            img[y, :] = c1 if (y // stripe_w) % 2 == 0 else c2
    else:
        for x in range(w):
            img[:, x] = c1 if (x // stripe_w) % 2 == 0 else c2
    return img


GENERATORS = [make_solid, make_gradient, make_noise_texture, make_stripes]


def main():
    args = sys.argv[1:]
    count = int(args[0]) if len(args) > 0 else DEFAULT_COUNT
    width = int(args[1]) if len(args) > 1 else DEFAULT_SIZE[0]
    height = int(args[2]) if len(args) > 2 else DEFAULT_SIZE[1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUT_DIR.glob("proc_*.jpg"))
    start_index = len(existing)

    for i in range(count):
        gen = GENERATORS[i % len(GENERATORS)]
        img = gen(width, height)
        dest = OUT_DIR / f"proc_{start_index + i:04d}.jpg"
        cv2.imwrite(str(dest), img)

    print(f"Wrote {count} procedural backgrounds to {OUT_DIR} "
          f"(total proc_*: {len(list(OUT_DIR.glob('proc_*.jpg')))})")


if __name__ == "__main__":
    main()
