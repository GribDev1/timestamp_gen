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


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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