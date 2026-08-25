from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    marker_size=0.1,
    y_min=0.7,
    y_max=26.7,
):
    """
    Plot raw detected photon timestamps versus real simulation time.

    Horizontal axis:
        Real pulse emission time within the simulated scene.

    Vertical axis:
        Photon round-trip timestamp in nanoseconds.

    The y-axis is normalized to the sensor's configured detectable
    depth range by converting min/max valid depth into equivalent
    round-trip timestamp limits.

    simulation_time_s stored in each frame is the END time of that block.
    """

    frames_dir = Path(dataset_dir) / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npz"))

    if not frame_files:
        raise RuntimeError(
            f"No raw timestamp frame files found in {frames_dir}"
        )
        
    if y_min_ns >= y_max_ns:
        raise ValueError(
            f"Timestamp y-axis minimum ({y_min_ns} ns) must be "
            f"less than maximum ({y_max_ns} ns)."
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
        # Include one earlier block in case its pulse interval
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

    selected_frame_files = frame_files[
        first_idx:last_idx
    ]

    print(
        f"Reading {len(selected_frame_files):,} of "
        f"{num_files:,} timestamp block files."
    )

    pulse_offset_s = (
        np.arange(
            block_size_L,
            dtype=np.float64,
        )
        / laser_rate_hz
    )

    for local_idx, frame_path in enumerate(
        selected_frame_files,
        start=first_idx,
    ):
        frame_number = local_idx + 1

        block_end_time_s = (
            frame_number * dt_s
        )

        block_start_time_s = (
            block_end_time_s
            - block_duration_s
        )

        with np.load(frame_path) as frame:
            timestamps_s = frame[
                "timestamps_noisy_s"
            ][:, pixel_y, pixel_x]

        valid = np.isfinite(timestamps_s)

        if not np.any(valid):
            continue

        pulse_times_s = (
            block_start_time_s
            + pulse_offset_s
        )

        if start_time_s is not None:
            valid &= (
                pulse_times_s >= start_time_s
            )

        if end_time_s is not None:
            valid &= (
                pulse_times_s <= end_time_s
            )

        if not np.any(valid):
            continue

        all_pulse_times_s.append(
            pulse_times_s[valid]
        )

        all_timestamps_s.append(
            timestamps_s[valid]
        )

    if not all_pulse_times_s:
        raise RuntimeError(
            "No detected timestamps were found in the selected "
            "pixel and time range."
        )

    pulse_times_s = np.concatenate(
        all_pulse_times_s
    )

    timestamps_s = np.concatenate(
        all_timestamps_s
    )

    timestamps_ns = timestamps_s * 1e9

    plt.figure(figsize=(10, 5))

    plt.scatter(
        pulse_times_s * 1e3,
        timestamps_ns,
        s=marker_size,
        alpha=0.35,
        linewidths=0,
    )

    plt.xlabel("Simulation time (ms)")
    plt.ylabel(
        "Photon round-trip timestamp (ns)"
    )

    plt.title(
        "Detected timestamps versus simulation time\n"
        f"ToF pixel y={pixel_y}, x={pixel_x}"
    )
    
    plt.ylim(y_min_ns, y_max_ns)

    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()

    print(
        f"Timestamp plot contains "
        f"{timestamps_s.size:,} detected photons."
    )


def save_pixel_outputs(
    all_histograms: np.ndarray,
    hist_bin_centers_depth_m: np.ndarray,
    block_times_s: np.ndarray,
    ttc_results: dict[str, np.ndarray],
    frame_idx: int,
    pixel_y: int,
    pixel_x: int,
    output_dir: Path,
    warning_s: float,
    max_ttc_s: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pixel_histogram(
        all_histograms=all_histograms,
        hist_bin_centers_depth_m=hist_bin_centers_depth_m,
        frame_idx=frame_idx,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
        output_path=(
            output_dir
            / f"histogram_block_{frame_idx:06d}.png"
        ),
    )

    save_pixel_ttc_over_time(
        block_times_s=block_times_s,
        ttc_results=ttc_results,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
        output_path=output_dir / "ttc_over_time.png",
        warning_s=warning_s,
        max_ttc_s=max_ttc_s,
    )

    save_pixel_ttc_csv(
        block_times_s=block_times_s,
        ttc_results=ttc_results,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
        output_path=output_dir / "ttc.csv",
    )