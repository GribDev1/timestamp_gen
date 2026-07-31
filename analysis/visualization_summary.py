from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_depth_frame(tof_depths, frame_idx, output_path):
    depth = tof_depths[frame_idx]

    plt.figure(figsize=(8, 4))
    plt.imshow(depth, origin="upper")
    plt.colorbar(label="Depth estimate (m)")
    plt.title(f"Histogram depth estimate, block {frame_idx}")
    plt.xlabel("ToF pixel x")
    plt.ylabel("ToF pixel y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_valid_fraction_frame(all_I, frame_idx, output_path):
    I = all_I[frame_idx]

    plt.figure(figsize=(8, 4))
    plt.imshow(I, origin="upper", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Valid detection fraction")
    plt.title(f"Valid detection fraction, block {frame_idx}")
    plt.xlabel("ToF pixel x")
    plt.ylabel("ToF pixel y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_mean_depth_over_time(tof_depths, output_path):
    mean_depth = np.nanmean(tof_depths, axis=(1, 2))

    plt.figure(figsize=(8, 4))
    plt.plot(mean_depth)
    plt.xlabel("Timestamp block index")
    plt.ylabel("Mean depth estimate (m)")
    plt.title("Mean ToF depth over time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_valid_fraction_over_time(all_I, output_path):
    mean_I = np.nanmean(all_I, axis=(1, 2))

    plt.figure(figsize=(8, 4))
    plt.plot(mean_I)
    plt.xlabel("Timestamp block index")
    plt.ylabel("Mean valid detection fraction")
    plt.title("Mean valid detection fraction over time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_ttc_frame(ttc, frame_idx, output_path, max_ttc_s):
    plt.figure(figsize=(8, 4))
    plt.imshow(
        ttc[frame_idx],
        origin="upper",
        vmin=0.0,
        vmax=max_ttc_s,
    )
    plt.colorbar(label="Time-to-contact (s)")
    plt.title(f"Time-to-contact, block {frame_idx}")
    plt.xlabel("ToF pixel x")
    plt.ylabel("ToF pixel y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary_over_time(
    tof_depths: np.ndarray,
    all_I: np.ndarray,
    block_times_s: np.ndarray,
    output_path: Path,
) -> None:
    mean_depth = np.nanmean(tof_depths, axis=(1, 2))
    mean_valid_fraction = np.nanmean(all_I, axis=(1, 2))
    time_ms = block_times_s * 1e3

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    axes[0].plot(time_ms, mean_depth)
    axes[0].set_ylabel("Mean depth (m)")
    axes[0].set_title("Timestamp dataset summary")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_ms, mean_valid_fraction)
    axes[1].set_xlabel("Simulation time (ms)")
    axes[1].set_ylabel("Valid detection fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_block_dashboard(
    tof_depths: np.ndarray,
    all_I: np.ndarray,
    all_histograms: np.ndarray,
    hist_bin_centers_depth_m: np.ndarray,
    ttc: np.ndarray,
    frame_idx: int,
    pixel_y: int,
    pixel_x: int,
    max_ttc_s: float,
    output_path: Path,
) -> None:
    depth = tof_depths[frame_idx]
    valid_fraction = all_I[frame_idx]
    ttc_frame = ttc[frame_idx]

    histogram = all_histograms[
        frame_idx,
        pixel_y,
        pixel_x,
        :,
    ]

    bin_width = float(
        np.mean(np.diff(hist_bin_centers_depth_m))
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    depth_image = axes[0, 0].imshow(
        depth,
        origin="upper",
    )
    figure.colorbar(
        depth_image,
        ax=axes[0, 0],
        label="Depth estimate (m)",
    )
    axes[0, 0].set_title("Histogram depth")
    axes[0, 0].set_xlabel("ToF pixel x")
    axes[0, 0].set_ylabel("ToF pixel y")

    valid_image = axes[0, 1].imshow(
        valid_fraction,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(
        valid_image,
        ax=axes[0, 1],
        label="Valid detection fraction",
    )
    axes[0, 1].set_title("Valid detection fraction")
    axes[0, 1].set_xlabel("ToF pixel x")
    axes[0, 1].set_ylabel("ToF pixel y")

    ttc_image = axes[1, 0].imshow(
        ttc_frame,
        origin="upper",
        vmin=0.0,
        vmax=max_ttc_s,
    )
    figure.colorbar(
        ttc_image,
        ax=axes[1, 0],
        label="Time-to-contact (s)",
    )
    axes[1, 0].set_title("Time-to-contact")
    axes[1, 0].set_xlabel("ToF pixel x")
    axes[1, 0].set_ylabel("ToF pixel y")

    axes[1, 1].bar(
        hist_bin_centers_depth_m,
        histogram,
        width=bin_width,
        align="center",
    )
    axes[1, 1].set_xlabel("Depth bin center (m)")
    axes[1, 1].set_ylabel("Detected count")
    axes[1, 1].set_title(
        f"Pixel histogram y={pixel_y}, x={pixel_x}"
    )

    figure.suptitle(
        f"Timestamp summary, block {frame_idx}"
    )

    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)