import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


C_LIGHT = 299_792_458.0


def load_depth_exr(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Could not read EXR: {path}")

    depth = depth.astype(np.float64)

    # VisionSIM may save the same depth in multiple channels.
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return depth


def build_ray_direction(
    render_y: float,
    render_x: float,
    image_h: int,
    image_w: int,
    fov_x_deg: float,
    fov_y_deg: float,
) -> np.ndarray:
    nx = ((render_x + 0.5) / image_w) * 2.0 - 1.0
    ny = ((render_y + 0.5) / image_h) * 2.0 - 1.0

    ray_x = nx * np.tan(np.deg2rad(fov_x_deg) / 2.0)
    ray_y = ny * np.tan(np.deg2rad(fov_y_deg) / 2.0)

    ray = np.array([ray_x, ray_y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def calculate_sample(
    name: str,
    depth: np.ndarray,
    render_y: int,
    render_x: int,
    fov_x_deg: float,
    fov_y_deg: float,
) -> dict:
    image_h, image_w = depth.shape
    z_depth_m = float(depth[render_y, render_x])

    ray = build_ray_direction(
        render_y,
        render_x,
        image_h,
        image_w,
        fov_x_deg,
        fov_y_deg,
    )

    if np.isfinite(z_depth_m) and z_depth_m > 0:
        slant_range_m = z_depth_m / ray[2]
        timestamp_ns = (2.0 * slant_range_m / C_LIGHT) * 1e9
    else:
        slant_range_m = np.nan
        timestamp_ns = np.nan

    return {
        "sample": name,
        "render_y": render_y,
        "render_x": render_x,
        "z_depth_m": z_depth_m,
        "ray_x": ray[0],
        "ray_y": ray[1],
        "ray_z": ray[2],
        "slant_range_m": slant_range_m,
        "expected_timestamp_ns": timestamp_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract corner and center depth values from one ToF pixel "
            "and convert them to expected photon round-trip timestamps."
        )
    )

    parser.add_argument("exr_path", type=Path)
    parser.add_argument("--tof-y", type=int, default=4)
    parser.add_argument("--tof-x", type=int, default=4)
    parser.add_argument("--tof-height", type=int, default=8)
    parser.add_argument("--tof-width", type=int, default=8)
    parser.add_argument("--fov-x-deg", type=float, default=48.8)
    parser.add_argument("--fov-y-deg", type=float, default=48.8)
    parser.add_argument("--output", type=Path, default=Path("tof_samples.csv"))

    args = parser.parse_args()

    depth = load_depth_exr(args.exr_path)
    image_h, image_w = depth.shape

    cell_h = image_h // args.tof_height
    cell_w = image_w // args.tof_width

    y0 = args.tof_y * cell_h
    y1 = (args.tof_y + 1) * cell_h - 1
    x0 = args.tof_x * cell_w
    x1 = (args.tof_x + 1) * cell_w - 1

    # Four central render pixels because a 20x20 footprint has no
    # single discrete center pixel.
    center_y0 = (y0 + y1) // 2
    center_y1 = center_y0 + 1
    center_x0 = (x0 + x1) // 2
    center_x1 = center_x0 + 1

    locations = [
        ("top_left", y0, x0),
        ("top_right", y0, x1),
        ("bottom_left", y1, x0),
        ("bottom_right", y1, x1),
        ("center_00", center_y0, center_x0),
        ("center_01", center_y0, center_x1),
        ("center_10", center_y1, center_x0),
        ("center_11", center_y1, center_x1),
    ]

    rows = [
        calculate_sample(
            name,
            depth,
            render_y,
            render_x,
            args.fov_x_deg,
            args.fov_y_deg,
        )
        for name, render_y, render_x in locations
    ]

    center_rows = rows[4:]

    center_average = {
        "sample": "center_average",
        "render_y": 0.5 * (center_y0 + center_y1),
        "render_x": 0.5 * (center_x0 + center_x1),
        "z_depth_m": np.nanmean(
            [row["z_depth_m"] for row in center_rows]
        ),
        "ray_x": np.nanmean(
            [row["ray_x"] for row in center_rows]
        ),
        "ray_y": np.nanmean(
            [row["ray_y"] for row in center_rows]
        ),
        "ray_z": np.nanmean(
            [row["ray_z"] for row in center_rows]
        ),
        "slant_range_m": np.nanmean(
            [row["slant_range_m"] for row in center_rows]
        ),
        "expected_timestamp_ns": np.nanmean(
            [row["expected_timestamp_ns"] for row in center_rows]
        ),
    }

    output_rows = rows[:4] + [center_average]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(output_rows[0].keys())

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"EXR: {args.exr_path}")
    print(f"Image size: {image_w} x {image_h}")
    print(
        f"ToF pixel ({args.tof_y}, {args.tof_x}) covers "
        f"rows {y0}-{y1}, columns {x0}-{x1}"
    )

    print()
    print(
        f"{'Sample':<16}"
        f"{'EXR depth (m)':>16}"
        f"{'Range (m)':>14}"
        f"{'Timestamp (ns)':>18}"
    )

    for row in output_rows:
        print(
            f"{row['sample']:<16}"
            f"{row['z_depth_m']:>16.6f}"
            f"{row['slant_range_m']:>14.6f}"
            f"{row['expected_timestamp_ns']:>18.6f}"
        )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()