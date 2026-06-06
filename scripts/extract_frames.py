"""
Extract sampled frames from short videos of each physical card, producing a
smaller real-world image set ready for manual annotation in Roboflow.

This is the "real-world supplement" to the synthetic dataset -- a handful
of frames captured in YOUR actual lighting/surfaces/hands close the
synthetic-to-real gap that pure augmentation can't fully bridge. You don't
need to annotate everything the camera captured; this script thins a video
down to a manageable, de-duplicated, non-blurry sample for you to label.

Expected input layout -- drop your recordings here:
    videos/<CardName>/*.mp4  (or .mov, .avi, .mkv)

e.g.  videos/Cloud/cloud_table.mp4
      videos/Sephiroth/sephiroth_handheld.mov

Output:
    frames/<CardName>/<video_stem>_<frame_index>.jpg

Frames are sampled evenly across the video and filtered for sharpness
(via Laplacian variance) so motion-blurred frames are skipped -- you'll
get cleaner candidates to annotate.

Usage:
    python extract_frames.py [frames_per_video] [sharpness_percentile]

Examples:
    python extract_frames.py                # 25 frames/video, keep sharpest 70%
    python extract_frames.py 40 50
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "videos"
FRAMES_DIR = ROOT / "frames"

DEFAULT_FRAMES_PER_VIDEO = 25
DEFAULT_SHARPNESS_PERCENTILE = 60  # drop the blurriest this-many-percent of candidates
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# Oversample candidates before sharpness filtering, so that after dropping
# the blurriest ones we still end up near the requested count. Handheld
# phone footage tends to have a lot of motion blur, so we sample generously.
OVERSAMPLE_FACTOR = 6


def sharpness(gray_frame):
    """Higher = sharper. Variance of the Laplacian is a standard focus measure."""
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var()


def sample_indices(total_frames, n):
    if total_frames <= n:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num=n, dtype=int).tolist()


def extract_from_video(video_path: Path, out_dir: Path, frames_per_video: int, sharpness_pct: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  ! {video_path.name}: could not read frame count")
        cap.release()
        return 0

    candidate_count = min(total, frames_per_video * OVERSAMPLE_FACTOR)
    indices = sample_indices(total, candidate_count)

    candidates = []  # (sharpness, index, frame)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates.append((sharpness(gray), idx, frame))
    cap.release()

    if not candidates:
        return 0

    scores = np.array([c[0] for c in candidates])
    threshold = np.percentile(scores, sharpness_pct)
    sharp_enough = [c for c in candidates if c[0] >= threshold]
    sharp_enough.sort(key=lambda c: c[1])  # restore chronological order

    keep = sharp_enough[:frames_per_video]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for _score, idx, frame in keep:
        dest = out_dir / f"{video_path.stem}_{idx:06d}.jpg"
        cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1

    return written


def main():
    args = sys.argv[1:]
    frames_per_video = int(args[0]) if len(args) > 0 else DEFAULT_FRAMES_PER_VIDEO
    sharpness_pct = float(args[1]) if len(args) > 1 else DEFAULT_SHARPNESS_PERCENTILE

    if not VIDEOS_DIR.exists():
        print(f"No videos/ folder found at {VIDEOS_DIR}")
        return

    card_dirs = [d for d in sorted(VIDEOS_DIR.iterdir()) if d.is_dir()]
    if not card_dirs:
        print(f"No per-card subfolders found in {VIDEOS_DIR}. "
              f"Expected e.g. videos/Cloud/your_video.mp4")
        return

    total_written = 0
    for card_dir in card_dirs:
        videos = [p for p in sorted(card_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTENSIONS]
        if not videos:
            print(f"{card_dir.name}: no video files found, skipping")
            continue

        out_dir = FRAMES_DIR / card_dir.name
        print(f"{card_dir.name}: found {len(videos)} video(s)")
        for video_path in videos:
            written = extract_from_video(video_path, out_dir, frames_per_video, sharpness_pct)
            print(f"  {video_path.name}: wrote {written} frames -> {out_dir.relative_to(ROOT)}")
            total_written += written

    print(f"\nDone. Wrote {total_written} frames total to {FRAMES_DIR}")
    print("Next: upload frames/<CardName>/ to Roboflow and draw bounding boxes "
          "(label each box with the matching class name) to build the real-world annotation set.")


if __name__ == "__main__":
    main()
