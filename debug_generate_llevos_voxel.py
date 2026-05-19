import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from event_util import generate_voxel_grid


DATA_ROOT = Path(r"D:\download\lowlight_event_vos")
IMAGE_ROOT = DATA_ROOT / "Lowlight_images"
EVENT_ROOT = DATA_ROOT / "Lowlight_event"
TARGET_ROOT = DATA_ROOT / "EventVoxel"

EVENT_BINS = 5
SEPARATE_POL = False


def main():
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([p for p in EVENT_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(video_dirs)} event video folders.")

    for video_dir in tqdm(video_dirs, desc="Generating voxel"):
        video_name = video_dir.name
        image_dir = IMAGE_ROOT / video_name
        target_dir = TARGET_ROOT / video_name
        target_dir.mkdir(parents=True, exist_ok=True)

        event_files = sorted(video_dir.glob("*.npy"))

        for event_file in event_files:
            stem = event_file.stem

            # 用对应 RGB 图像确定 H,W
            image_path = image_dir / f"{stem}.png"
            if not image_path.exists():
                print(f"[Skip] image not found: {image_path}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[Skip] failed to read image: {image_path}")
                continue

            shape = image.shape[:2]  # H, W

            events = np.load(str(event_file))

            if events.ndim != 2 or events.shape[1] != 4:
                print(f"[Skip] bad event shape {events.shape}: {event_file}")
                continue

            voxel = generate_voxel_grid(
                events=events,
                shape=shape,
                nr_temporal_bins=EVENT_BINS,
                separate_pol=SEPARATE_POL,
            )

            out_path = target_dir / f"{stem}.npy"
            np.save(str(out_path), voxel.astype(np.float32))

    print(f"Done. Saved voxel files to: {TARGET_ROOT}")


if __name__ == "__main__":
    main()