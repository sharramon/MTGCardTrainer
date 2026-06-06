"""
Bulk-download generic background photos for synthetic compositing.

Pulls random real-world stock photos from Lorem Picsum (https://picsum.photos,
backed by Unsplash) -- no API key required. These give the synthetic dataset
generator broad visual variety (rooms, textures, objects, lighting) so the
detector learns to focus on the card rather than memorizing a background.

This is intentionally generic filler, not a substitute for the real
photos/video of YOUR actual environment (table, hands, lighting) -- that
real-world set is what closes the synthetic-to-real gap.

Usage:
    python download_backgrounds.py [count] [width] [height]

Examples:
    python download_backgrounds.py            # downloads 150 images at 800x600
    python download_backgrounds.py 300 1024 768
"""

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "backgrounds"

DEFAULT_COUNT = 150
DEFAULT_SIZE = (800, 600)
TIMEOUT_SECS = 15
RETRIES = 3


def download_one(seed: int, width: int, height: int, dest: Path) -> bool:
    url = f"https://picsum.photos/seed/mtgbg{seed}/{width}/{height}"
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MTGCardTrainer/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == RETRIES:
                print(f"  ! failed seed {seed}: {exc}")
                return False
            time.sleep(1.5 * attempt)
    return False


def main():
    args = sys.argv[1:]
    count = int(args[0]) if len(args) > 0 else DEFAULT_COUNT
    width = int(args[1]) if len(args) > 1 else DEFAULT_SIZE[0]
    height = int(args[2]) if len(args) > 2 else DEFAULT_SIZE[1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUT_DIR.glob("bg_*.jpg"))
    start_index = len(existing)

    print(f"Downloading {count} background images ({width}x{height}) into {OUT_DIR}")
    print(f"Starting at index {start_index} (found {len(existing)} existing)")

    downloaded = 0
    for i in range(count):
        idx = start_index + i
        dest = OUT_DIR / f"bg_{idx:04d}.jpg"
        if dest.exists():
            continue
        if download_one(seed=idx, width=width, height=height, dest=dest):
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  ...{downloaded}/{count} downloaded")

    print(f"Done. Downloaded {downloaded} new images. Total in folder: {len(list(OUT_DIR.glob('bg_*.jpg')))}")


if __name__ == "__main__":
    main()
