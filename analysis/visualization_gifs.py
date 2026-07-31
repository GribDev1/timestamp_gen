from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def save_depth_gif(tof_depths, output_path, fps=12, gif_stride=1):
    """
    Save animated GIF of histogram-derived depth estimates over time.
    """
    frame_indices = np.arange(
        0,
        tof_depths.shape[0],
        gif_stride,
    )

    num_frames = len(frame_indices)

    finite_depths = tof_depths[np.isfinite(tof_depths)]
    if finite_depths.size == 0:
        print("Skipping depth GIF: no finite depth values.")
        return

    vmin = np.nanpercentile(finite_depths, 1)
    vmax = np.nanpercentile(finite_depths, 99)
    first_source_idx = frame_indices[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        tof_depths[first_source_idx],
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Depth estimate (m)")

    title = ax.set_title(f"Histogram depth estimate, block {first_source_idx}")
    ax.set_xlabel("ToF pixel x")
    ax.set_ylabel("ToF pixel y")

    def update(frame_idx):
        source_idx = frame_indices[frame_idx]
        im.set_data(tof_depths[source_idx])
        title.set_text(f"Histogram depth estimate, block {source_idx}")
        return im, title

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_valid_fraction_gif(all_I, output_path, fps=12, gif_stride=1):
    """
    Save animated GIF of valid detection fraction over time.
    """
    frame_indices = np.arange(
        0,
        all_I.shape[0],
        gif_stride,
    )

    num_frames = len(frame_indices)
    first_source_idx = frame_indices[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        all_I[first_source_idx],
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Valid detection fraction")

    title = ax.set_title(f"Valid detection fraction, block {first_source_idx}")
    ax.set_xlabel("ToF pixel x")
    ax.set_ylabel("ToF pixel y")

    def update(frame_idx):
        source_idx = frame_indices[frame_idx]

        im.set_data(all_I[source_idx])
        title.set_text(
            f"Valid detection fraction, block {source_idx}"
        )

        return im, title

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_pixel_histogram_gif(
    all_histograms,
    hist_bin_centers_depth_m,
    pixel_y,
    pixel_x,
    output_path,
    fps=12,
    gif_stride=1,
):
    """
    Save animated GIF of one pixel histogram over time.

    By default, use the center pixel selected in main().
    """
    frame_indices = np.arange(
        0,
        all_histograms.shape[0],
        gif_stride,
    )

    num_frames = len(frame_indices)
    bin_width = np.mean(np.diff(hist_bin_centers_depth_m))
    hist_depth_min = hist_bin_centers_depth_m[0] - 0.5 * bin_width
    hist_depth_max = hist_bin_centers_depth_m[-1] + 0.5 * bin_width
    first_source_idx = frame_indices[0]

    max_count = np.max(all_histograms[:, pixel_y, pixel_x, :])
    if max_count <= 0:
        max_count = 1

    fig, ax = plt.subplots(figsize=(8, 4))

    hist0 = all_histograms[first_source_idx, pixel_y, pixel_x, :]
    bars = ax.bar(
        hist_bin_centers_depth_m,
        hist0,
        width=bin_width,
    )

    ax.set_xlim(hist_depth_min, hist_depth_max)
    ax.set_ylim(0, max_count * 1.1)

    ax.set_xlabel("Depth bin center (m)")
    ax.set_ylabel("Detected count")
    title = ax.set_title(f"Center pixel histogram, block {first_source_idx}, y={pixel_y}, x={pixel_x}")

    def update(frame_idx):
        source_idx = frame_indices[frame_idx]

        hist = all_histograms[
            source_idx,
            pixel_y,
            pixel_x,
            :,
        ]

        for bar, height in zip(bars, hist):
            bar.set_height(height)

        title.set_text(
            f"Center pixel histogram, block {source_idx}, "
            f"y={pixel_y}, x={pixel_x}"
        )

        return (*bars, title)

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def validate_gif_options(
    fps: int,
    gif_stride: int,
) -> None:
    if fps <= 0:
        raise ValueError("GIF FPS must be greater than zero.")

    if gif_stride <= 0:
        raise ValueError(
            "GIF stride must be greater than zero."
        )