from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an animated RGB normal-map GIF."
    )

    parser.add_argument(
        "--normal-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="GIF playback frame rate.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth rendered frame.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Integer output enlargement factor.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
    )

    return parser.parse_args()


def extract_frame_number(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)

    if not numbers:
        raise RuntimeError(
            f"Could not determine frame number from {path.name}"
        )

    return int(numbers[-1])


def load_normal_rgb(path: Path) -> np.ndarray:
    normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if normal is None:
        raise RuntimeError(f"Could not load normal EXR: {path}")

    if normal.ndim != 3 or normal.shape[2] < 3:
        raise RuntimeError(
            f"Normal EXR must have at least 3 channels: {path}"
        )

    normal = normal[:, :, :3].astype(np.float32)

    # Match timestamp_gen.py: OpenCV BGR -> RGB.
    normal = normal[:, :, ::-1]

    magnitude = np.linalg.norm(normal, axis=-1)
    valid = np.isfinite(magnitude) & (magnitude > 1e-8)

    normalized = np.zeros_like(normal)
    normalized[valid] = (
        normal[valid] / magnitude[valid, np.newaxis]
    )

    # Convert normal components from [-1, 1] into RGB [0, 255].
    rgb = np.clip(
        (normalized + 1.0) * 127.5,
        0.0,
        255.0,
    ).astype(np.uint8)

    # Make pixels without valid normals black.
    rgb[~valid] = 0

    return rgb


def main():
    args = parse_args()

    normal_files = sorted(
        args.normal_dir.glob("*.exr"),
        key=extract_frame_number,
    )

    if args.start_frame is not None:
        normal_files = [
            path
            for path in normal_files
            if extract_frame_number(path) >= args.start_frame
        ]

    if args.end_frame is not None:
        normal_files = [
            path
            for path in normal_files
            if extract_frame_number(path) <= args.end_frame
        ]

    normal_files = normal_files[::args.stride]

    if not normal_files:
        raise RuntimeError(
            f"No matching EXR files found in {args.normal_dir}"
        )

    gif_frames = []

    for index, normal_path in enumerate(normal_files, start=1):
        frame_number = extract_frame_number(normal_path)
        rgb = load_normal_rgb(normal_path)

        image = Image.fromarray(rgb, mode="RGB")

        if args.scale != 1:
            image = image.resize(
                (
                    image.width * args.scale,
                    image.height * args.scale,
                ),
                resample=Image.Resampling.NEAREST,
            )

        draw = ImageDraw.Draw(image)

        label = (
            f"VisionSIM normals | frame {frame_number} | "
            f"{rgb.shape[1]}x{rgb.shape[0]}"
        )

        box = draw.textbbox((0, 0), label)
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]

        draw.rectangle(
            (5, 5, 15 + box_width, 15 + box_height),
            fill=(0, 0, 0),
        )
        draw.text(
            (10, 10),
            label,
            fill=(255, 255, 255),
        )

        gif_frames.append(image)

        print(
            f"[{index}/{len(normal_files)}] "
            f"{normal_path.name}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = round(1000.0 / args.fps)

    gif_frames[0].save(
        args.output,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    print(f"Saved {len(gif_frames)} frames to {args.output}")


if __name__ == "__main__":
    main()