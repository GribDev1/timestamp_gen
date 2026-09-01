from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a VisionSIM normal EXR."
    )
    parser.add_argument("normal_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--arrow-stride",
        type=int,
        default=16,
        help="Display one normal arrow every N rendered pixels.",
    )
    return parser.parse_args()


def load_normals(path: Path):
    normals = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if normals is None:
        raise RuntimeError(f"Could not load normal EXR: {path}")

    if normals.ndim != 3 or normals.shape[2] < 3:
        raise RuntimeError(
            f"Expected at least three channels, got {normals.shape}"
        )

    normals = normals[:, :, :3].astype(np.float32)

    # Match timestamp_gen.py: OpenCV BGR -> RGB.
    normals = normals[:, :, ::-1]

    magnitude = np.linalg.norm(normals, axis=-1)
    valid = np.isfinite(magnitude) & (magnitude > 1e-8)

    normalized = np.zeros_like(normals)
    normalized[valid] = (
        normals[valid] / magnitude[valid, np.newaxis]
    )

    return normalized, valid


def main():
    args = parse_args()

    normals, valid = load_normals(args.normal_file)

    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2]

    # Convert [-1, 1] normal components into [0, 1] RGB.
    normal_rgb = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
    normal_rgb[~valid] = 0.0

    height, width = valid.shape
    stride = args.arrow_stride

    yy, xx = np.mgrid[
        stride // 2 : height : stride,
        stride // 2 : width : stride,
    ]

    arrow_valid = valid[yy, xx]
    arrow_x = xx[arrow_valid]
    arrow_y = yy[arrow_valid]
    arrow_u = nx[yy, xx][arrow_valid]
    arrow_v = -ny[yy, xx][arrow_valid]
    arrow_color = nz[yy, xx][arrow_valid]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        constrained_layout=True,
    )

    axes[0, 0].imshow(normal_rgb, origin="upper")
    axes[0, 0].set_title("RGB normal map")
    axes[0, 0].set_xlabel("Rendered pixel x")
    axes[0, 0].set_ylabel("Rendered pixel y")

    component_data = [
        (nx, r"$N_x$"),
        (ny, r"$N_y$"),
        (nz, r"$N_z$"),
    ]

    for axis, (component, title) in zip(
        [axes[0, 1], axes[0, 2], axes[1, 0]],
        component_data,
    ):
        masked = np.ma.masked_where(~valid, component)

        image = axis.imshow(
            masked,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            origin="upper",
        )

        axis.set_title(title)
        axis.set_xlabel("Rendered pixel x")
        axis.set_ylabel("Rendered pixel y")
        figure.colorbar(image, ax=axis, label="Normal component")

    arrows = axes[1, 1].quiver(
        arrow_x,
        arrow_y,
        arrow_u,
        arrow_v,
        arrow_color,
        cmap="coolwarm",
        clim=(-1.0, 1.0),
        angles="xy",
        scale_units="xy",
        scale=0.08,
        width=0.004,
    )

    axes[1, 1].set_xlim(0, width)
    axes[1, 1].set_ylim(height, 0)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title(
        f"Normal direction, every {stride} pixels"
    )
    axes[1, 1].set_xlabel("Rendered pixel x")
    axes[1, 1].set_ylabel("Rendered pixel y")
    figure.colorbar(arrows, ax=axes[1, 1], label=r"$N_z$")

    magnitude_image = axes[1, 2].imshow(
        valid.astype(np.float32),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    axes[1, 2].set_title("Valid normal mask")
    axes[1, 2].set_xlabel("Rendered pixel x")
    axes[1, 2].set_ylabel("Rendered pixel y")
    figure.colorbar(
        magnitude_image,
        ax=axes[1, 2],
        ticks=[0, 1],
    )

    figure.suptitle(args.normal_file.name, fontsize=15)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_file, dpi=200)
    plt.close(figure)

    print(f"Normal resolution: {width}x{height}")
    print(f"Valid normals: {np.count_nonzero(valid):,}")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()