import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_normal_exr(path: Path) -> np.ndarray:
    normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if normal is None:
        raise RuntimeError(f"Could not read normal EXR: {path}")

    if normal.ndim != 3 or normal.shape[2] < 3:
        raise RuntimeError(
            f"Expected at least 3 EXR channels, got shape {normal.shape}"
        )

    normal = normal[:, :, :3].astype(np.float64)

    # OpenCV returns BGR; timestamp_gen converts it to RGB.
    return normal[:, :, ::-1]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))

    if not np.isfinite(magnitude) or magnitude < 1e-12:
        return np.full(3, np.nan)

    return vector / magnitude


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


def calculate_values(
    name: str,
    render_y: float,
    render_x: float,
    raw_normal: np.ndarray,
    image_h: int,
    image_w: int,
    fov_x_deg: float,
    fov_y_deg: float,
) -> dict:
    normal = normalize_vector(raw_normal)

    ray = build_ray_direction(
        render_y,
        render_x,
        image_h,
        image_w,
        fov_x_deg,
        fov_y_deg,
    )

    # This matches timestamp_gen:
    # cos_incidence = dot(-ray_direction, surface_normal)
    if np.all(np.isfinite(normal)):
        cosine_raw = float(np.dot(-ray, normal))
        cosine_used = max(cosine_raw, 0.0)

        incidence_angle_deg = float(
            np.degrees(
                np.arccos(np.clip(cosine_raw, -1.0, 1.0))
            )
        )
    else:
        cosine_raw = np.nan
        cosine_used = np.nan
        incidence_angle_deg = np.nan

    return {
        "sample": name,
        "render_y": render_y,
        "render_x": render_x,
        "raw_normal_x": raw_normal[0],
        "raw_normal_y": raw_normal[1],
        "raw_normal_z": raw_normal[2],
        "raw_magnitude": np.linalg.norm(raw_normal),
        "normal_x": normal[0],
        "normal_y": normal[1],
        "normal_z": normal[2],
        "ray_x": ray[0],
        "ray_y": ray[1],
        "ray_z": ray[2],
        "cos_incidence_raw": cosine_raw,
        "cos_incidence_used": cosine_used,
        "incidence_angle_deg": incidence_angle_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract surface normals from the corners and center "
            "of one ToF pixel."
        )
    )

    parser.add_argument("normal_exr", type=Path)
    parser.add_argument("--tof-y", type=int, default=4)
    parser.add_argument("--tof-x", type=int, default=4)
    parser.add_argument("--tof-height", type=int, default=8)
    parser.add_argument("--tof-width", type=int, default=8)
    parser.add_argument("--fov-x-deg", type=float, default=48.8)
    parser.add_argument("--fov-y-deg", type=float, default=48.8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tof_normal_samples.csv"),
    )

    args = parser.parse_args()

    normal_map = load_normal_exr(args.normal_exr)
    image_h, image_w, _ = normal_map.shape

    cell_h = image_h // args.tof_height
    cell_w = image_w // args.tof_width

    y0 = args.tof_y * cell_h
    y1 = (
        (args.tof_y + 1) * cell_h
        if args.tof_y < args.tof_height - 1
        else image_h
    ) - 1

    x0 = args.tof_x * cell_w
    x1 = (
        (args.tof_x + 1) * cell_w
        if args.tof_x < args.tof_width - 1
        else image_w
    ) - 1

    center_y0 = (y0 + y1) // 2
    center_y1 = center_y0 + 1
    center_x0 = (x0 + x1) // 2
    center_x1 = center_x0 + 1

    corner_locations = [
        ("top_left", y0, x0),
        ("top_right", y0, x1),
        ("bottom_left", y1, x0),
        ("bottom_right", y1, x1),
    ]

    rows = []

    for name, render_y, render_x in corner_locations:
        rows.append(
            calculate_values(
                name=name,
                render_y=render_y,
                render_x=render_x,
                raw_normal=normal_map[render_y, render_x],
                image_h=image_h,
                image_w=image_w,
                fov_x_deg=args.fov_x_deg,
                fov_y_deg=args.fov_y_deg,
            )
        )

    # A 20x20 footprint has no one discrete center pixel.
    center_normals = np.array(
        [
            normal_map[center_y0, center_x0],
            normal_map[center_y0, center_x1],
            normal_map[center_y1, center_x0],
            normal_map[center_y1, center_x1],
        ]
    )

    center_raw_normal = np.mean(center_normals, axis=0)

    rows.append(
        calculate_values(
            name="center_average",
            render_y=(center_y0 + center_y1) / 2.0,
            render_x=(center_x0 + center_x1) / 2.0,
            raw_normal=center_raw_normal,
            image_h=image_h,
            image_w=image_w,
            fov_x_deg=args.fov_x_deg,
            fov_y_deg=args.fov_y_deg,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Normal EXR: {args.normal_exr}")
    print(f"Image size: {image_w} x {image_h}")
    print(
        f"ToF pixel ({args.tof_y}, {args.tof_x}) covers "
        f"rows {y0}-{y1}, columns {x0}-{x1}"
    )

    print()
    print(
        f"{'Sample':<16}"
        f"{'Normal (x, y, z)':>36}"
        f"{'Cosine':>12}"
        f"{'Angle':>12}"
    )

    for row in rows:
        normal_text = (
            f"({row['normal_x']:.5f}, "
            f"{row['normal_y']:.5f}, "
            f"{row['normal_z']:.5f})"
        )

        print(
            f"{row['sample']:<16}"
            f"{normal_text:>36}"
            f"{row['cos_incidence_used']:>12.6f}"
            f"{row['incidence_angle_deg']:>11.3f}°"
        )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()