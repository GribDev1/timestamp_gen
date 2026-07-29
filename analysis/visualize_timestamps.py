"""
Visualize timestamp generator outputs.

This script loads timestamp_precomputed.npz and creates diagnostic figures:
    - selected depth frames
    - valid detection fraction frames
    - mean depth over time
    - mean valid detection fraction over time
    - one example pixel histogram

Example:
    python visualize_timestamps.py --input timestamp_output/timestamp_precomputed.npz --output-dir timestamp_output/figures
"""

from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize precomputed timestamp generator outputs."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("timestamp_output/timestamp_precomputed.npz"),
        help="Path to timestamp_precomputed.npz.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/timestamp_output/figures"),
        help="Directory where figures will be saved.",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame/block index to visualize. Default: 0",
    )

    parser.add_argument(
        "--pixel-y",
        type=int,
        default=None,
        help="Pixel y index for histogram. Default: center row.",
    )

    parser.add_argument(
        "--pixel-x",
        type=int,
        default=None,
        help="Pixel x index for histogram. Default: center column.",
    )
    
    parser.add_argument(
        "--make-gifs",
        action="store_true",
        help="Create animated GIFs over all timestamp blocks.",
    )

    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Frames per second for output GIFs. Default: 12",
    )
    
    parser.add_argument(
        "--start-time-ms",
        type=float,
        default=None,
        help="Optional beginning of timestamp plot window in milliseconds.",
    )

    parser.add_argument(
        "--end-time-ms",
        type=float,
        default=None,
        help="Optional end of timestamp plot window in milliseconds.",
    )

    parser.add_argument(
        "--timestamp-marker-size",
        type=float,
        default=2.0,
        help="Scatter marker size for timestamp-versus-time plot. Default: 2.0",
    )

    parser.add_argument(
        "--ttc-window-ms",
        type=float,
        default=10.0,
        help=(
            "Causal smoothing/differencing window for time-to-contact in "
            "milliseconds. Default: 10.0"
        ),
    )

    parser.add_argument(
        "--ttc-min-closing-speed",
        type=float,
        default=0.10,
        help=(
            "Minimum positive closing speed in m/s required for a valid "
            "time-to-contact estimate. Default: 0.10"
        ),
    )

    parser.add_argument(
        "--ttc-max-s",
        type=float,
        default=10.0,
        help=(
            "Maximum TTC value retained and displayed, in seconds. "
            "Larger values are stored as NaN. Default: 10.0"
        ),
    )

    parser.add_argument(
        "--ttc-warning-s",
        type=float,
        default=1.0,
        help=(
            "Warning threshold drawn on the selected-pixel TTC plot, "
            "in seconds. Default: 1.0"
        ),
    )

    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def save_pixel_histogram(
    all_histograms,
    hist_bin_centers_depth_m,
    frame_idx,
    pixel_y,
    pixel_x,
    output_path,
):
    hist = all_histograms[frame_idx, pixel_y, pixel_x]

    bin_width = np.mean(np.diff(hist_bin_centers_depth_m))
    hist_depth_min = hist_bin_centers_depth_m[0] - 0.5 * bin_width
    hist_depth_max = hist_bin_centers_depth_m[-1] + 0.5 * bin_width

    plt.figure(figsize=(8, 4))
    plt.bar(
        hist_bin_centers_depth_m,
        hist,
        width=bin_width,
        align="center",
    )

    plt.xlim(hist_depth_min, hist_depth_max)

    plt.xlabel("Depth bin center (m)")
    plt.ylabel("Detected count")
    plt.title(f"Pixel histogram, block {frame_idx}, y={pixel_y}, x={pixel_x}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    
def save_depth_gif(tof_depths, output_path, fps=12):
    """
    Save animated GIF of histogram-derived depth estimates over time.
    """
    num_frames = tof_depths.shape[0]

    finite_depths = tof_depths[np.isfinite(tof_depths)]
    if finite_depths.size == 0:
        print("Skipping depth GIF: no finite depth values.")
        return

    vmin = np.nanpercentile(finite_depths, 1)
    vmax = np.nanpercentile(finite_depths, 99)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        tof_depths[0],
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Depth estimate (m)")

    title = ax.set_title("Histogram depth estimate, block 0")
    ax.set_xlabel("ToF pixel x")
    ax.set_ylabel("ToF pixel y")

    def update(frame_idx):
        im.set_data(tof_depths[frame_idx])
        title.set_text(f"Histogram depth estimate, block {frame_idx}")
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


def save_valid_fraction_gif(all_I, output_path, fps=12):
    """
    Save animated GIF of valid detection fraction over time.
    """
    num_frames = all_I.shape[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        all_I[0],
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Valid detection fraction")

    title = ax.set_title("Valid detection fraction, block 0")
    ax.set_xlabel("ToF pixel x")
    ax.set_ylabel("ToF pixel y")

    def update(frame_idx):
        im.set_data(all_I[frame_idx])
        title.set_text(f"Valid detection fraction, block {frame_idx}")
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


def save_center_pixel_histogram_gif(
    all_histograms,
    hist_bin_centers_depth_m,
    pixel_y,
    pixel_x,
    output_path,
    fps=12,
):
    """
    Save animated GIF of one pixel histogram over time.

    By default, use the center pixel selected in main().
    """
    num_frames = all_histograms.shape[0]
    bin_width = np.mean(np.diff(hist_bin_centers_depth_m))
    hist_depth_min = hist_bin_centers_depth_m[0] - 0.5 * bin_width
    hist_depth_max = hist_bin_centers_depth_m[-1] + 0.5 * bin_width

    max_count = np.max(all_histograms[:, pixel_y, pixel_x, :])
    if max_count <= 0:
        max_count = 1

    fig, ax = plt.subplots(figsize=(8, 4))

    hist0 = all_histograms[0, pixel_y, pixel_x, :]
    bars = ax.bar(
        hist_bin_centers_depth_m,
        hist0,
        width=bin_width,
    )

    ax.set_xlim(hist_depth_min, hist_depth_max)
    ax.set_ylim(0, max_count * 1.1)

    ax.set_xlabel("Depth bin center (m)")
    ax.set_ylabel("Detected count")
    title = ax.set_title(f"Center pixel histogram, block 0, y={pixel_y}, x={pixel_x}")

    def update(frame_idx):
        hist = all_histograms[frame_idx, pixel_y, pixel_x, :]

        for bar, height in zip(bars, hist):
            bar.set_height(height)

        title.set_text(
            f"Center pixel histogram, block {frame_idx}, y={pixel_y}, x={pixel_x}"
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
    
    

def build_block_times_s(data, metadata, num_blocks):
    """
    Return one representative acquisition time per timestamp block.

    Newer datasets may store block_start_time_s and block_end_time_s in
    timestamp_precomputed.npz. Older datasets fall back to metadata["dt_s"].
    """
    if "block_start_time_s" in data.files and "block_end_time_s" in data.files:
        starts = np.asarray(data["block_start_time_s"], dtype=np.float64)
        ends = np.asarray(data["block_end_time_s"], dtype=np.float64)

        if starts.shape[0] != num_blocks or ends.shape[0] != num_blocks:
            raise ValueError(
                "Saved block timing arrays do not match tof_depths length."
            )

        return 0.5 * (starts + ends)

    dt_s = float(metadata["dt_s"])
    return (np.arange(num_blocks, dtype=np.float64) + 0.5) * dt_s


def causal_nanmean(values, window_blocks):
    """
    Causal trailing mean along axis 0 while ignoring NaNs.

    values shape: [time, ...]
    """
    if window_blocks <= 1:
        return values.astype(np.float32, copy=True)

    finite = np.isfinite(values)
    sums = np.cumsum(np.where(finite, values, 0.0), axis=0, dtype=np.float64)
    counts = np.cumsum(finite.astype(np.int32), axis=0, dtype=np.int64)

    sums = np.concatenate([np.zeros_like(sums[:1]), sums], axis=0)
    counts = np.concatenate([np.zeros_like(counts[:1]), counts], axis=0)

    end_idx = np.arange(1, values.shape[0] + 1)
    start_idx = np.maximum(0, end_idx - window_blocks)

    window_sums = sums[end_idx] - sums[start_idx]
    window_counts = counts[end_idx] - counts[start_idx]

    result = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(
        window_sums,
        window_counts,
        out=result,
        where=window_counts > 0,
    )
    return result


def compute_time_to_contact(
    tof_depths,
    block_times_s,
    window_ms=10.0,
    min_closing_speed_mps=0.10,
    max_ttc_s=10.0,
):
    """
    Compute causal radial velocity, closing speed, and time-to-contact.

    TTC is defined as:
        TTC = depth / closing_speed

    where closing_speed = -d(depth)/dt. A valid TTC is produced only when
    the range is decreasing faster than min_closing_speed_mps.
    """
    if window_ms <= 0:
        raise ValueError("--ttc-window-ms must be positive.")
    if min_closing_speed_mps < 0:
        raise ValueError("--ttc-min-closing-speed must be nonnegative.")
    if max_ttc_s <= 0:
        raise ValueError("--ttc-max-s must be positive.")
    if tof_depths.shape[0] != block_times_s.shape[0]:
        raise ValueError("block_times_s length must match tof_depths.")
    if block_times_s.size < 2:
        raise ValueError("At least two timestamp blocks are required for TTC.")

    median_dt_s = float(np.median(np.diff(block_times_s)))
    if not np.isfinite(median_dt_s) or median_dt_s <= 0:
        raise ValueError("Block times must be finite and strictly increasing.")

    window_blocks = max(1, int(round((window_ms * 1e-3) / median_dt_s)))
    smoothed_depths = causal_nanmean(tof_depths, window_blocks)

    radial_velocity = np.full_like(smoothed_depths, np.nan, dtype=np.float32)

    # Compare each causal mean with the causal mean one window earlier.
    lag = window_blocks
    if lag < smoothed_depths.shape[0]:
        dt = block_times_s[lag:] - block_times_s[:-lag]
        valid_dt = np.isfinite(dt) & (dt > 0)

        depth_delta = smoothed_depths[lag:] - smoothed_depths[:-lag]
        velocity = np.full_like(depth_delta, np.nan, dtype=np.float32)
        np.divide(
            depth_delta,
            dt[:, None, None],
            out=velocity,
            where=valid_dt[:, None, None],
        )
        radial_velocity[lag:] = velocity

    closing_speed = -radial_velocity

    valid_ttc = (
        np.isfinite(smoothed_depths)
        & np.isfinite(closing_speed)
        & (smoothed_depths > 0)
        & (closing_speed >= min_closing_speed_mps)
    )

    ttc = np.full_like(smoothed_depths, np.nan, dtype=np.float32)
    np.divide(
        smoothed_depths,
        closing_speed,
        out=ttc,
        where=valid_ttc,
    )

    ttc[(ttc <= 0) | (ttc > max_ttc_s)] = np.nan

    return {
        "smoothed_depths_m": smoothed_depths,
        "radial_velocity_mps": radial_velocity,
        "closing_speed_mps": closing_speed,
        "time_to_contact_s": ttc,
        "window_blocks": window_blocks,
        "median_dt_s": median_dt_s,
    }


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


def save_pixel_ttc_over_time(
    block_times_s,
    ttc_results,
    pixel_y,
    pixel_x,
    output_path,
    warning_s,
    max_ttc_s,
):
    ttc = ttc_results["time_to_contact_s"][:, pixel_y, pixel_x]

    plt.figure(figsize=(10, 5))
    plt.plot(block_times_s * 1e3, ttc, linewidth=1.0)
    plt.axhline(
        warning_s,
        linestyle="--",
        linewidth=1.0,
        label=f"Warning threshold = {warning_s:g} s",
    )
    plt.ylim(0.0, max_ttc_s)
    plt.xlabel("Simulation time (ms)")
    plt.ylabel("Time-to-contact (s)")
    plt.title(f"Time-to-contact over time, ToF pixel y={pixel_y}, x={pixel_x}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_pixel_ttc_csv(
    block_times_s,
    ttc_results,
    pixel_y,
    pixel_x,
    output_path,
):
    num_blocks = block_times_s.shape[0]
    block_index = np.arange(num_blocks, dtype=np.int64)

    table = np.column_stack(
        [
            block_index,
            block_times_s,
            ttc_results["smoothed_depths_m"][:, pixel_y, pixel_x],
            ttc_results["radial_velocity_mps"][:, pixel_y, pixel_x],
            ttc_results["closing_speed_mps"][:, pixel_y, pixel_x],
            ttc_results["time_to_contact_s"][:, pixel_y, pixel_x],
        ]
    )

    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=(
            "block_index,time_s,smoothed_depth_m,radial_velocity_mps,"
            "closing_speed_mps,time_to_contact_s"
        ),
        comments="",
        fmt=["%d", "%.9f", "%.6f", "%.6f", "%.6f", "%.6f"],
    )


def save_timestamps_vs_time(
    dataset_dir,
    metadata,
    pixel_y,
    pixel_x,
    output_path,
    start_time_ms=None,
    end_time_ms=None,
    marker_size=2.0,
):
    """
    Plot raw detected photon timestamps versus real simulation time.

    Horizontal axis:
        Real pulse emission time within the simulated scene.

    Vertical axis:
        Round-trip photon timestamp/delay.

    simulation_time_s stored in each frame is the END time of that block.
    """

    frames_dir = Path(dataset_dir) / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npz"))

    if not frame_files:
        raise RuntimeError(
            f"No raw timestamp frame files found in {frames_dir}"
        )

    laser_rate_hz = float(metadata["laser_rate_hz"])
    block_size_L = int(metadata["block_size_L"])
    block_duration_s = block_size_L / laser_rate_hz

    start_time_s = (
        start_time_ms * 1e-3
        if start_time_ms is not None
        else None
    )

    end_time_s = (
        end_time_ms * 1e-3
        if end_time_ms is not None
        else None
    )
    
    all_pulse_times_s = []
    all_timestamps_s = []

    dt_s = float(metadata["dt_s"])
    num_files = len(frame_files)

    first_idx = 0
    last_idx = num_files

    if start_time_s is not None:
        # Include one earlier block in case its physical pulse interval
        # overlaps the requested starting time.
        first_idx = max(
            0,
            int(np.floor(start_time_s / dt_s)) - 1,
        )

    if end_time_s is not None:
        last_idx = min(
            num_files,
            int(np.ceil(end_time_s / dt_s)) + 1,
        )

    selected_frame_files = frame_files[first_idx:last_idx]

    print(
        f"Reading {len(selected_frame_files):,} of "
        f"{num_files:,} timestamp block files."
    )
    
    pulse_offset_s = np.arange(block_size_L, dtype=np.float64) / laser_rate_hz

    for local_idx, frame_path in enumerate(
        selected_frame_files,
        start=first_idx,
    ):
        frame_number = local_idx + 1

        block_end_time_s = frame_number * dt_s
        block_start_time_s = block_end_time_s - block_duration_s

        with np.load(frame_path) as frame:
            timestamps_s = frame[
                "timestamps_noisy_s"
            ][:, pixel_y, pixel_x]

        valid = np.isfinite(timestamps_s)

        if not np.any(valid):
            continue

        pulse_times_s = block_start_time_s + pulse_offset_s

        if start_time_s is not None:
            valid &= pulse_times_s >= start_time_s

        if end_time_s is not None:
            valid &= pulse_times_s <= end_time_s

        if not np.any(valid):
            continue

        all_pulse_times_s.append(pulse_times_s[valid])
        all_timestamps_s.append(timestamps_s[valid])

    if not all_pulse_times_s:
        raise RuntimeError(
            "No detected timestamps were found in the selected "
            "pixel and time range."
        )

    pulse_times_s = np.concatenate(all_pulse_times_s)
    timestamps_s = np.concatenate(all_timestamps_s)

    plt.figure(figsize=(10, 5))

    plt.scatter(
        pulse_times_s * 1e3,
        timestamps_s * 1e9,
        s=marker_size,
        alpha=0.5,
        linewidths=0,
    )

    plt.xlabel("Simulation time (ms)")
    plt.ylabel("Photon round-trip timestamp (ns)")
    plt.title(
        "Detected timestamps versus simulation time\n"
        f"ToF pixel y={pixel_y}, x={pixel_x}"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(
        f"Timestamp plot contains "
        f"{timestamps_s.size:,} detected photons."
    )


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input)
    
    dataset_dir = args.input.parent
    metadata = load_metadata(dataset_dir)

    tof_depths = data["tof_depths"]
    all_I = data["all_I"]
    all_histograms = data["all_histograms"]
    hist_bin_centers_depth_m = data["hist_bin_centers_depth_m"]

    num_frames, tof_h, tof_w = tof_depths.shape

    block_times_s = build_block_times_s(
        data=data,
        metadata=metadata,
        num_blocks=num_frames,
    )

    ttc_results = compute_time_to_contact(
        tof_depths=tof_depths,
        block_times_s=block_times_s,
        window_ms=args.ttc_window_ms,
        min_closing_speed_mps=args.ttc_min_closing_speed,
        max_ttc_s=args.ttc_max_s,
    )

    frame_idx = args.frame

    if frame_idx < 0 or frame_idx >= num_frames:
        raise ValueError(
            f"Frame index {frame_idx} is out of range. "
            f"Valid range: 0 to {num_frames - 1}"
        )

    pixel_y = args.pixel_y if args.pixel_y is not None else tof_h // 2
    pixel_x = args.pixel_x if args.pixel_x is not None else tof_w // 2

    if not (0 <= pixel_y < tof_h and 0 <= pixel_x < tof_w):
        raise ValueError(
            f"Pixel ({pixel_y}, {pixel_x}) is outside ToF grid {tof_h}x{tof_w}"
        )

    print(f"Loaded: {args.input}")
    print(f"tof_depths shape: {tof_depths.shape}")
    print(f"all_I shape: {all_I.shape}")
    print(f"all_histograms shape: {all_histograms.shape}")
    print(f"Visualizing block {frame_idx}")
    print(f"Histogram pixel: y={pixel_y}, x={pixel_x}")
    print(
        "TTC window: "
        f"{args.ttc_window_ms:.3f} ms "
        f"({ttc_results['window_blocks']} blocks)"
    )

    save_depth_frame(
        tof_depths,
        frame_idx,
        args.output_dir / f"depth_block_{frame_idx:04d}.png",
    )

    save_valid_fraction_frame(
        all_I,
        frame_idx,
        args.output_dir / f"valid_fraction_block_{frame_idx:04d}.png",
    )

    save_mean_depth_over_time(
        tof_depths,
        args.output_dir / "mean_depth_over_time.png",
    )

    save_valid_fraction_over_time(
        all_I,
        args.output_dir / "valid_fraction_over_time.png",
    )

    save_pixel_histogram(
        all_histograms,
        hist_bin_centers_depth_m,
        frame_idx,
        pixel_y,
        pixel_x,
        args.output_dir / f"histogram_block_{frame_idx:04d}_y{pixel_y}_x{pixel_x}.png",
    )
    
    save_timestamps_vs_time(
        dataset_dir=dataset_dir,
        metadata=metadata,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
        output_path=(
            args.output_dir
            / f"timestamps_vs_time_y{pixel_y}_x{pixel_x}.png"
        ),
        start_time_ms=args.start_time_ms,
        end_time_ms=args.end_time_ms,
        marker_size=args.timestamp_marker_size,
    )

    np.savez(
        args.output_dir / "time_to_contact.npz",
        block_times_s=block_times_s,
        smoothed_depths_m=ttc_results["smoothed_depths_m"],
        radial_velocity_mps=ttc_results["radial_velocity_mps"],
        closing_speed_mps=ttc_results["closing_speed_mps"],
        time_to_contact_s=ttc_results["time_to_contact_s"],
        ttc_window_ms=np.array(args.ttc_window_ms, dtype=np.float64),
        ttc_window_blocks=np.array(ttc_results["window_blocks"], dtype=np.int32),
        min_closing_speed_mps=np.array(
            args.ttc_min_closing_speed, dtype=np.float64
        ),
        max_ttc_s=np.array(args.ttc_max_s, dtype=np.float64),
    )

    save_ttc_frame(
        ttc=ttc_results["time_to_contact_s"],
        frame_idx=frame_idx,
        output_path=args.output_dir / f"ttc_block_{frame_idx:04d}.png",
        max_ttc_s=args.ttc_max_s,
    )

    save_pixel_ttc_over_time(
        block_times_s=block_times_s,
        ttc_results=ttc_results,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
        output_path=(
            args.output_dir
            / f"ttc_over_time_y{pixel_y}_x{pixel_x}.png"
        ),
        warning_s=args.ttc_warning_s,
        max_ttc_s=args.ttc_max_s,
    )

    save_pixel_ttc_csv(
        block_times_s=block_times_s,
        ttc_results=ttc_results,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
        output_path=(
            args.output_dir
            / f"ttc_y{pixel_y}_x{pixel_x}.csv"
        ),
    )
    
    if args.make_gifs:
        print("Saving GIFs...")

        save_depth_gif(
            tof_depths,
            args.output_dir / "depth_over_time.gif",
            fps=args.gif_fps,
        )

        save_valid_fraction_gif(
            all_I,
            args.output_dir / "valid_fraction_over_time.gif",
            fps=args.gif_fps,
        )

        save_center_pixel_histogram_gif(
            all_histograms,
            hist_bin_centers_depth_m,
            pixel_y,
            pixel_x,
            args.output_dir / f"center_histogram_y{pixel_y}_x{pixel_x}_over_time.gif",
            fps=args.gif_fps,
        )

    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()