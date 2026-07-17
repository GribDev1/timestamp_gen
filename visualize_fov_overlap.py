"""
Visualize Blender-camera / ToF-sensor field-of-view overlap and estimate
which rendered rays contribute to corresponding top and bottom ToF zones.

This tool intentionally distinguishes two concepts:

1. Angular FoV overlap
   The Blender render and ToF sensor may cover different angular extents.

2. Photon contribution inside a ToF zone
   Rendered depth/normal samples inside the angular intersection are integrated
   using the same style of weighting as timestamp_gen:
       weight ~ max(0, cos(incidence)) / range^2
   If no normal map is supplied, weight ~ 1 / range^2.

Outputs:
    fov_overlap.png
    tof_zone_overlay.png
    selected_zone_contribution.png
    selected_zone_ray_fan.png
    overlap_summary.txt

Example:
    python visualize_fov_overlap.py ^
        --sensor vl53l8ch ^
        --depth inputs/drone_flyby/depths/depth_000001.exr ^
        --normal inputs/drone_flyby/normals/normal_000001.exr ^
        --rgb inputs/drone_flyby/images/frame_000001.png ^
        --blender-fov-x 73.7398 ^
        --blender-fov-y 41.1121 ^
        --pixel-x 4 ^
        --output-dir outputs/examples/drone_flyby/fov_overlap

If RGB is unavailable, the depth map is used as the background.

Important:
    This script visualizes a physically angular mapping. If the current
    timestamp generator simply divides the entire render into equal 8x8 image
    cells, that implementation is not equivalent when Blender FoV and ToF FoV
    differ. The generated figures make that mismatch visible.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from sensor_presets import get_sensor_preset


@dataclass(frozen=True)
class AngularBounds:
    """Angular/projective bounds represented as ray slopes."""

    x0: float
    x1: float
    y0: float
    y1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Blender/ToF FoV overlap and integrated photon "
            "contributions for matching top and bottom ToF zones."
        )
    )

    parser.add_argument(
        "--sensor",
        default="vl53l8ch",
        help="Sensor preset name from configs/sensors. Default: vl53l8ch",
    )

    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="Rendered depth EXR file.",
    )

    parser.add_argument(
        "--normal",
        type=Path,
        default=None,
        help="Optional rendered normal EXR file.",
    )

    parser.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help="Optional rendered RGB image for the overlay background.",
    )

    parser.add_argument(
        "--blender-fov-x",
        type=float,
        default=73.7398,
        help="Blender horizontal FoV in degrees. Default: 73.7398",
    )

    parser.add_argument(
        "--blender-fov-y",
        type=float,
        default=41.1121,
        help="Blender vertical FoV in degrees. Default: 41.1121",
    )

    parser.add_argument(
        "--pixel-x",
        type=int,
        default=None,
        help="Selected ToF column. Default: center column.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fov_overlap_output"),
        help="Output directory. Default: fov_overlap_output",
    )

    parser.add_argument(
        "--uniform-contribution",
        action="store_true",
        help=(
            "Use equal weight for every valid rendered ray instead of "
            "incidence/distance weighting."
        ),
    )

    return parser.parse_args()


def load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Could not load depth image: {path}")

    depth = depth.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return depth


def load_normal(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None

    normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if normal is None:
        raise RuntimeError(f"Could not load normal image: {path}")

    normal = normal.astype(np.float32)

    if normal.ndim != 3 or normal.shape[2] < 3:
        raise RuntimeError(
            f"Normal image must have at least three channels: {path}"
        )

    # Match timestamp_gen: OpenCV BGR -> RGB.
    normal = normal[:, :, :3][:, :, ::-1]

    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-12)

    return normal.astype(np.float32)


def load_background(
    rgb_path: Path | None,
    depth: np.ndarray,
) -> tuple[np.ndarray, str]:
    if rgb_path is not None:
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not load RGB image: {rgb_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, "RGB render"

    finite = np.isfinite(depth)
    display = np.full(depth.shape, np.nan, dtype=np.float32)
    display[finite] = depth[finite]
    return display, "Rendered depth"


def fov_to_slope(fov_deg: float) -> float:
    return float(np.tan(np.deg2rad(fov_deg) / 2.0))


def build_blender_ray_dirs(
    image_h: int,
    image_w: int,
    fov_x_deg: float,
    fov_y_deg: float,
) -> np.ndarray:
    """
    Build normalized camera rays for the actual Blender render FoV.
    """
    sx = fov_to_slope(fov_x_deg)
    sy = fov_to_slope(fov_y_deg)

    xs = ((np.arange(image_w, dtype=np.float64) + 0.5) / image_w) * 2.0 - 1.0
    ys = ((np.arange(image_h, dtype=np.float64) + 0.5) / image_h) * 2.0 - 1.0

    nx, ny = np.meshgrid(xs, ys)

    rays = np.stack(
        [
            nx * sx,
            ny * sy,
            np.ones_like(nx),
        ],
        axis=-1,
    )

    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.astype(np.float32)


def tof_zone_bounds(
    pixel_y: int,
    pixel_x: int,
    tof_h: int,
    tof_w: int,
    tof_fov_x_deg: float,
    tof_fov_y_deg: float,
) -> AngularBounds:
    """
    Return one ToF zone's boundaries in ray-slope coordinates.

    The ToF model samples uniformly in normalized image-plane coordinates,
    matching sensor_presets.ToFSensor.build_fullres_ray_dirs().
    """
    sx = fov_to_slope(tof_fov_x_deg)
    sy = fov_to_slope(tof_fov_y_deg)

    x_edges = np.linspace(-sx, sx, tof_w + 1)
    y_edges = np.linspace(-sy, sy, tof_h + 1)

    return AngularBounds(
        x0=float(x_edges[pixel_x]),
        x1=float(x_edges[pixel_x + 1]),
        y0=float(y_edges[pixel_y]),
        y1=float(y_edges[pixel_y + 1]),
    )


def slope_to_pixel_x(slope_x: float, image_w: int, blender_fov_x_deg: float) -> float:
    blender_half_slope = fov_to_slope(blender_fov_x_deg)
    normalized = slope_x / blender_half_slope
    return 0.5 * (normalized + 1.0) * image_w


def slope_to_pixel_y(slope_y: float, image_h: int, blender_fov_y_deg: float) -> float:
    blender_half_slope = fov_to_slope(blender_fov_y_deg)
    normalized = slope_y / blender_half_slope
    return 0.5 * (normalized + 1.0) * image_h


def zone_pixel_rectangle(
    bounds: AngularBounds,
    image_h: int,
    image_w: int,
    blender_fov_x_deg: float,
    blender_fov_y_deg: float,
) -> tuple[float, float, float, float]:
    x0 = slope_to_pixel_x(bounds.x0, image_w, blender_fov_x_deg)
    x1 = slope_to_pixel_x(bounds.x1, image_w, blender_fov_x_deg)
    y0 = slope_to_pixel_y(bounds.y0, image_h, blender_fov_y_deg)
    y1 = slope_to_pixel_y(bounds.y1, image_h, blender_fov_y_deg)
    return x0, y0, x1, y1


def clip_rectangle(
    rectangle: tuple[float, float, float, float],
    image_h: int,
    image_w: int,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = rectangle

    cx0 = max(0.0, min(float(image_w), x0))
    cx1 = max(0.0, min(float(image_w), x1))
    cy0 = max(0.0, min(float(image_h), y0))
    cy1 = max(0.0, min(float(image_h), y1))

    if cx1 <= cx0 or cy1 <= cy0:
        return None

    return cx0, cy0, cx1, cy1


def angular_overlap_fraction(
    blender_fov_x_deg: float,
    blender_fov_y_deg: float,
    tof_fov_x_deg: float,
    tof_fov_y_deg: float,
) -> tuple[float, float, float]:
    """
    Return horizontal, vertical, and rectangular angular-coverage fractions
    of the ToF field that are represented by the Blender render.

    Fractions are calculated in image-plane slope space, consistent with
    pinhole projection area.
    """
    bx = fov_to_slope(blender_fov_x_deg)
    by = fov_to_slope(blender_fov_y_deg)
    tx = fov_to_slope(tof_fov_x_deg)
    ty = fov_to_slope(tof_fov_y_deg)

    horizontal = min(bx, tx) / tx
    vertical = min(by, ty) / ty
    return horizontal, vertical, horizontal * vertical


def save_global_fov_overlap(
    output_path: Path,
    blender_fov_x_deg: float,
    blender_fov_y_deg: float,
    tof_fov_x_deg: float,
    tof_fov_y_deg: float,
) -> None:
    """
    Draw angular rectangles in tangent/image-plane coordinates.
    """
    bx = fov_to_slope(blender_fov_x_deg)
    by = fov_to_slope(blender_fov_y_deg)
    tx = fov_to_slope(tof_fov_x_deg)
    ty = fov_to_slope(tof_fov_y_deg)

    figure, axis = plt.subplots(figsize=(8, 6))

    blender_rect = Rectangle(
        (-bx, -by),
        2.0 * bx,
        2.0 * by,
        fill=False,
        linewidth=2.0,
        label=(
            f"Blender {blender_fov_x_deg:.1f}° × "
            f"{blender_fov_y_deg:.1f}°"
        ),
    )

    tof_rect = Rectangle(
        (-tx, -ty),
        2.0 * tx,
        2.0 * ty,
        fill=False,
        linewidth=2.0,
        linestyle="--",
        label=(
            f"ToF {tof_fov_x_deg:.1f}° × "
            f"{tof_fov_y_deg:.1f}°"
        ),
    )

    axis.add_patch(blender_rect)
    axis.add_patch(tof_rect)

    max_x = 1.1 * max(bx, tx)
    max_y = 1.1 * max(by, ty)

    axis.set_xlim(-max_x, max_x)
    axis.set_ylim(-max_y, max_y)
    axis.set_aspect("equal", adjustable="box")
    axis.axhline(0.0, linewidth=0.8)
    axis.axvline(0.0, linewidth=0.8)
    axis.set_xlabel(r"Horizontal ray slope, $\tan(\theta_x)$")
    axis.set_ylabel(r"Vertical ray slope, $\tan(\theta_y)$")
    axis.set_title("Blender camera and ToF sensor FoV overlap")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_tof_zone_overlay(
    output_path: Path,
    background: np.ndarray,
    background_name: str,
    tof_h: int,
    tof_w: int,
    tof_fov_x_deg: float,
    tof_fov_y_deg: float,
    blender_fov_x_deg: float,
    blender_fov_y_deg: float,
    selected_y: int,
    selected_x: int,
) -> None:
    image_h, image_w = background.shape[:2]

    figure, axis = plt.subplots(figsize=(12, 6))

    if background.ndim == 2:
        image_artist = axis.imshow(background, origin="upper")
        figure.colorbar(image_artist, ax=axis, label="Camera-axis depth (m)")
    else:
        axis.imshow(background, origin="upper")

    for y in range(tof_h):
        for x in range(tof_w):
            bounds = tof_zone_bounds(
                y,
                x,
                tof_h,
                tof_w,
                tof_fov_x_deg,
                tof_fov_y_deg,
            )

            raw_rect = zone_pixel_rectangle(
                bounds,
                image_h,
                image_w,
                blender_fov_x_deg,
                blender_fov_y_deg,
            )

            clipped = clip_rectangle(raw_rect, image_h, image_w)

            if clipped is None:
                continue

            cx0, cy0, cx1, cy1 = clipped
            is_selected = y == selected_y and x == selected_x

            axis.add_patch(
                Rectangle(
                    (cx0, cy0),
                    cx1 - cx0,
                    cy1 - cy0,
                    fill=False,
                    linewidth=2.2 if is_selected else 0.7,
                    linestyle="-" if is_selected else ":",
                )
            )

            if is_selected:
                axis.text(
                    0.5 * (cx0 + cx1),
                    0.5 * (cy0 + cy1),
                    f"({y},{x})",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )

    axis.set_xlim(0, image_w)
    axis.set_ylim(image_h, 0)
    axis.set_xlabel("Rendered pixel x")
    axis.set_ylabel("Rendered pixel y")
    axis.set_title(
        f"Angular ToF-zone footprints on {background_name}\n"
        "Zones outside the Blender FoV are clipped or absent"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def selected_zone_mask(
    image_h: int,
    image_w: int,
    blender_fov_x_deg: float,
    blender_fov_y_deg: float,
    bounds: AngularBounds,
) -> np.ndarray:
    """
    Select actual rendered pixels whose Blender rays lie inside the ToF zone.
    """
    sx = fov_to_slope(blender_fov_x_deg)
    sy = fov_to_slope(blender_fov_y_deg)

    xs = ((np.arange(image_w, dtype=np.float64) + 0.5) / image_w) * 2.0 - 1.0
    ys = ((np.arange(image_h, dtype=np.float64) + 0.5) / image_h) * 2.0 - 1.0

    slope_x = xs[np.newaxis, :] * sx
    slope_y = ys[:, np.newaxis] * sy

    return (
        (slope_x >= bounds.x0)
        & (slope_x < bounds.x1)
        & (slope_y >= bounds.y0)
        & (slope_y < bounds.y1)
    )


def compute_contribution_map(
    depth_z: np.ndarray,
    normal: np.ndarray | None,
    ray_dirs: np.ndarray,
    zone_mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    block_size_l: int,
    detection_probability_rho: float,
    uniform_contribution: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        expected_detected_photons: [H,W]
        probability_map: [H,W]
        range_map: [H,W]
    """
    ray_z = np.maximum(ray_dirs[:, :, 2], 1e-8)
    ranges = depth_z / ray_z

    valid = (
        zone_mask
        & np.isfinite(depth_z)
        & np.isfinite(ranges)
        & (ranges > min_depth_m)
        & (ranges < max_depth_m)
    )

    weights = np.zeros(depth_z.shape, dtype=np.float64)

    if uniform_contribution:
        weights[valid] = 1.0
    else:
        distance_falloff = np.zeros_like(weights)
        distance_falloff[valid] = 1.0 / np.maximum(ranges[valid] ** 2, 1e-12)

        if normal is not None:
            cos_incidence = np.sum((-ray_dirs) * normal, axis=-1)
            cos_incidence = np.maximum(cos_incidence, 0.0)
            weights[valid] = cos_incidence[valid] * distance_falloff[valid]
        else:
            weights[valid] = distance_falloff[valid]

    total_weight = float(np.sum(weights))

    probability = np.zeros_like(weights)

    if total_weight > 0:
        probability = weights / total_weight

    expected_detected = (
        block_size_l
        * detection_probability_rho
        * probability
    )

    return expected_detected, probability, ranges


def map_extent(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)

    if xs.size == 0:
        return None

    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


def save_selected_contribution(
    output_path: Path,
    background: np.ndarray,
    expected_detected: np.ndarray,
    zone_mask: np.ndarray,
    selected_y: int,
    selected_x: int,
) -> None:
    extent = map_extent(zone_mask)

    if extent is None:
        raise RuntimeError(
            "The selected ToF zone has no overlap with the Blender render."
        )

    x0, x1, y0, y1 = extent

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    if background.ndim == 2:
        axes[0].imshow(background, origin="upper")
    else:
        axes[0].imshow(background, origin="upper")

    axes[0].add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=2.0,
        )
    )
    axes[0].set_title(f"Selected ToF zone ({selected_y},{selected_x})")
    axes[0].set_xlabel("Rendered pixel x")
    axes[0].set_ylabel("Rendered pixel y")

    contribution_crop = expected_detected[y0:y1, x0:x1]
    masked_crop = np.ma.masked_where(
        ~zone_mask[y0:y1, x0:x1],
        contribution_crop,
    )

    image_artist = axes[1].imshow(
        masked_crop,
        origin="upper",
        interpolation="nearest",
    )
    figure.colorbar(
        image_artist,
        ax=axes[1],
        label="Expected detected photons per rendered ray per block",
    )

    axes[1].set_title(
        "Integrated rendered-ray contribution\n"
        r"$E_i=L\rho\,p_i$"
    )
    axes[1].set_xlabel("Rendered-patch pixel x")
    axes[1].set_ylabel("Rendered-patch pixel y")

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_selected_ray_fan(
    output_path: Path,
    bounds: AngularBounds,
    blender_fov_x_deg: float,
    blender_fov_y_deg: float,
    tof_fov_x_deg: float,
    tof_fov_y_deg: float,
    selected_y: int,
    selected_x: int,
) -> None:
    """
    Draw horizontal and vertical cross-section ray fans.

    These are camera-coordinate diagrams; they do not include scene geometry.
    """
    center_x = 0.5 * (bounds.x0 + bounds.x1)
    center_y = 0.5 * (bounds.y0 + bounds.y1)

    z_end = 1.0

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Horizontal x-z fan.
    for slope, label in [
        (bounds.x0, "Zone left edge"),
        (center_x, "Zone center"),
        (bounds.x1, "Zone right edge"),
    ]:
        axes[0].plot([0.0, slope * z_end], [0.0, z_end], label=label)

    blender_x = fov_to_slope(blender_fov_x_deg)
    tof_x = fov_to_slope(tof_fov_x_deg)

    axes[0].plot([0.0, -blender_x], [0.0, z_end], linestyle=":", label="Blender FoV edge")
    axes[0].plot([0.0, blender_x], [0.0, z_end], linestyle=":")
    axes[0].plot([0.0, -tof_x], [0.0, z_end], linestyle="--", label="ToF FoV edge")
    axes[0].plot([0.0, tof_x], [0.0, z_end], linestyle="--")

    axes[0].set_xlabel("Camera x / arbitrary scale")
    axes[0].set_ylabel("Forward z / arbitrary scale")
    axes[0].set_title("Horizontal ray fan")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    # Vertical y-z fan.
    for slope, label in [
        (bounds.y0, "Zone top edge"),
        (center_y, "Zone center"),
        (bounds.y1, "Zone bottom edge"),
    ]:
        axes[1].plot([0.0, slope * z_end], [0.0, z_end], label=label)

    blender_y = fov_to_slope(blender_fov_y_deg)
    tof_y = fov_to_slope(tof_fov_y_deg)

    axes[1].plot([0.0, -blender_y], [0.0, z_end], linestyle=":", label="Blender FoV edge")
    axes[1].plot([0.0, blender_y], [0.0, z_end], linestyle=":")
    axes[1].plot([0.0, -tof_y], [0.0, z_end], linestyle="--", label="ToF FoV edge")
    axes[1].plot([0.0, tof_y], [0.0, z_end], linestyle="--")

    axes[1].set_xlabel("Camera y / arbitrary scale")
    axes[1].set_ylabel("Forward z / arbitrary scale")
    axes[1].set_title("Vertical ray fan")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    figure.suptitle(
        f"Selected ToF-zone rays: y={selected_y}, x={selected_x}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    output_path: Path,
    blender_fov_x_deg: float,
    blender_fov_y_deg: float,
    sensor,
    selected_y: int,
    selected_x: int,
    zone_mask: np.ndarray,
    valid_contribution: np.ndarray,
) -> None:
    horizontal, vertical, total = angular_overlap_fraction(
        blender_fov_x_deg,
        blender_fov_y_deg,
        sensor.camera_fov_x_deg,
        sensor.camera_fov_y_deg,
    )

    rendered_rays_in_zone = int(np.count_nonzero(zone_mask))
    contributing_rays = int(np.count_nonzero(valid_contribution > 0.0))
    expected_detections = float(np.sum(valid_contribution))

    lines = [
        "Blender / ToF FoV overlap summary",
        "=================================",
        "",
        f"Sensor preset: {sensor.name}",
        f"Blender FoV: {blender_fov_x_deg:.6f} deg x {blender_fov_y_deg:.6f} deg",
        (
            f"ToF FoV: {sensor.camera_fov_x_deg:.6f} deg x "
            f"{sensor.camera_fov_y_deg:.6f} deg"
        ),
        f"ToF grid: {sensor.tof_h} x {sensor.tof_w}",
        "",
        f"Horizontal ToF coverage represented by Blender: {100.0 * horizontal:.3f}%",
        f"Vertical ToF coverage represented by Blender: {100.0 * vertical:.3f}%",
        f"Approximate 2D projective coverage: {100.0 * total:.3f}%",
        "",
        f"Selected ToF zone: y={selected_y}, x={selected_x}",
        f"Rendered rays/pixels in angular overlap: {rendered_rays_in_zone}",
        f"Valid contributing rendered rays: {contributing_rays}",
        f"Expected detections in one block: {expected_detections:.6f}",
        (
            "Configured expected detections per valid ToF pixel/block: "
            f"{sensor.block_size_L * sensor.detection_probability_rho:.6f}"
        ),
        "",
        "Interpretation:",
        (
            "- Expected detections are distributed among valid rendered rays "
            "according to normalized integration weights."
        ),
        (
            "- A zone that is only partly inside the Blender FoV has fewer "
            "available rendered rays, but the current normalized model still "
            "allocates the configured expected detections among those rays."
        ),
        (
            "- Missing angular coverage is not simulated; it should not be "
            "mistaken for zero-photon scene content."
        ),
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sensor = get_sensor_preset(args.sensor)

    depth = load_depth(args.depth)
    normal = load_normal(args.normal)

    if normal is not None and normal.shape[:2] != depth.shape:
        raise ValueError(
            f"Depth shape {depth.shape} and normal shape {normal.shape[:2]} differ."
        )

    background, background_name = load_background(args.rgb, depth)

    if background.shape[:2] != depth.shape:
        raise ValueError(
            f"Background shape {background.shape[:2]} and depth shape {depth.shape} differ."
        )

    pixel_x = sensor.tof_w // 2 if args.pixel_x is None else args.pixel_x

    if not (0 <= pixel_x < sensor.tof_w):
        raise ValueError(
            f"Selected column {pixel_x} is outside "
            f"the valid range 0 to {sensor.tof_w - 1}."
        )

    selected_rows = {
        "top": 0,
        "bottom": sensor.tof_h - 1,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_h, image_w = depth.shape

    ray_dirs = build_blender_ray_dirs(
        image_h,
        image_w,
        args.blender_fov_x,
        args.blender_fov_y,
    )
    
    save_global_fov_overlap(
        output_dir / "fov_overlap.png",
        args.blender_fov_x,
        args.blender_fov_y,
        sensor.camera_fov_x_deg,
        sensor.camera_fov_y_deg,
    )

    for row_name, pixel_y in selected_rows.items():
        bounds = tof_zone_bounds(
            pixel_y,
            pixel_x,
            sensor.tof_h,
            sensor.tof_w,
            sensor.camera_fov_x_deg,
            sensor.camera_fov_y_deg,
        )

        zone_mask = selected_zone_mask(
            image_h,
            image_w,
            args.blender_fov_x,
            args.blender_fov_y,
            bounds,
        )
        
        save_tof_zone_overlay(
            output_dir / f"{row_name}_zone_overlay.png",
            background,
            background_name,
            sensor.tof_h,
            sensor.tof_w,
            sensor.camera_fov_x_deg,
            sensor.camera_fov_y_deg,
            args.blender_fov_x,
            args.blender_fov_y,
            pixel_y,
            pixel_x,
        )

        expected_detected, probability, ranges = compute_contribution_map(
            depth_z=depth,
            normal=normal,
            ray_dirs=ray_dirs,
            zone_mask=zone_mask,
            min_depth_m=sensor.min_valid_depth_m,
            max_depth_m=sensor.max_valid_depth_m,
            block_size_l=sensor.block_size_L,
            detection_probability_rho=sensor.detection_probability_rho,
            uniform_contribution=args.uniform_contribution,
        )

        if np.count_nonzero(zone_mask) == 0:
            print(
                f"Warning: {row_name} zone "
                f"({pixel_y},{pixel_x}) has no Blender overlap."
            )
            continue

        save_selected_contribution(
            output_dir / f"{row_name}_zone_contribution.png",
            background,
            expected_detected,
            zone_mask,
            pixel_y,
            pixel_x,
        )

        save_selected_ray_fan(
            output_dir / f"{row_name}_zone_ray_fan.png",
            bounds,
            args.blender_fov_x,
            args.blender_fov_y,
            sensor.camera_fov_x_deg,
            sensor.camera_fov_y_deg,
            pixel_y,
            pixel_x,
        )

        write_summary(
            output_dir / f"{row_name}_overlap_summary.txt",
            args.blender_fov_x,
            args.blender_fov_y,
            sensor,
            pixel_y,
            pixel_x,
            zone_mask,
            expected_detected,
        )

        print(
            f"{row_name.capitalize()} zone ({pixel_y},{pixel_x}): "
            f"{np.count_nonzero(zone_mask):,} rendered rays, "
            f"{np.count_nonzero(expected_detected > 0.0):,} contributing rays"
        )


if __name__ == "__main__":
    main()
