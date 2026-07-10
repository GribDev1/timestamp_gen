"""
Minimal working example for reading AND interpreting timestamp_gen output.

This script demonstrates:
1. Reading metadata.
2. Loading precomputed histogram-depth data.
3. Extracting a center-pixel depth trace.
4. Computing simple frame/block differencing.
5. Estimating velocity from histogram depth.
6. Reading raw timestamps and converting them back to depth.
7. Inspecting a mini-histogram for possible multi-surface structure.

Run:
    python interpret_timestamp_dataset.py outputs/examples/flat_moving_ex
"""

from pathlib import Path
import json
import argparse

import numpy as np


def load_metadata(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_precomputed(dataset_dir: Path):
    precomputed_path = dataset_dir / "timestamp_precomputed.npz"

    data = np.load(precomputed_path)

    return {
        "tof_depths": data["tof_depths"],
        "all_I": data["all_I"],
        "all_histograms": data["all_histograms"],
        "hist_bin_centers_depth_m": data["hist_bin_centers_depth_m"],
    }


def load_raw_block(dataset_dir: Path, block_idx: int):
    """
    Load one raw timestamp block.

    block_idx is zero-based:
        block_idx = 0  -> frame_000001.npz
        block_idx = 40 -> frame_000041.npz
        block_idx = 53 -> frame_000054.npz
    """
    frames_dir = dataset_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npz"))

    if not frame_files:
        raise RuntimeError(f"No frame files found in {frames_dir}")

    if block_idx < 0 or block_idx >= len(frame_files):
        raise ValueError(
            f"block_idx {block_idx} is out of range. "
            f"Valid range: 0 to {len(frame_files) - 1}"
        )

    return frame_files[block_idx], np.load(frame_files[block_idx])


def summarize_metadata(metadata):
    print("=== Metadata summary ===")

    tof_h = metadata["tof_h"]
    tof_w = metadata["tof_w"]
    L = metadata["block_size_L"]
    laser_rate_hz = metadata["laser_rate_hz"]
    rho = metadata["detection_probability_rho"]
    fps = metadata["fps"]
    dt_s = metadata["dt_s"]

    block_duration_s = L / laser_rate_hz
    expected_detections = L * rho

    print(f"ToF grid: {tof_h} x {tof_w}")
    print(f"Saved block/update rate: {fps:.2f} Hz")
    print(f"Saved block dt: {dt_s * 1e3:.4f} ms")
    print(f"Laser-pulse block duration: {block_duration_s * 1e6:.2f} us")
    print(f"Expected detections / pixel / block: {expected_detections:.2f}")
    print()


def analyze_center_pixel_depth(pre, metadata):
    tof_depths = pre["tof_depths"]
    all_I = pre["all_I"]

    dt_s = metadata["dt_s"]

    num_blocks, tof_h, tof_w = tof_depths.shape

    y = tof_h // 2
    x = tof_w // 2

    depth_trace = tof_depths[:, y, x]
    valid_fraction_trace = all_I[:, y, x]

    # Simple finite difference of histogram depth estimates.
    depth_diff = np.diff(depth_trace)

    # Approximate longitudinal velocity.
    # Negative velocity means object moving toward the sensor if depth decreases.
    velocity_estimate = depth_diff / dt_s

    print("=== Center pixel histogram-depth interpretation ===")
    print(f"Center pixel: y={y}, x={x}")
    print(f"Number of timestamp blocks: {num_blocks}")
    print(f"Mean histogram depth: {np.nanmean(depth_trace):.4f} m")
    print(f"Depth min/max: {np.nanmin(depth_trace):.4f} / {np.nanmax(depth_trace):.4f} m")
    print(f"Mean valid detection fraction: {np.nanmean(valid_fraction_trace):.4f}")

    if np.any(np.isfinite(velocity_estimate)):
        print(f"Mean finite-difference velocity: {np.nanmean(velocity_estimate):.4f} m/s")
        print(f"Velocity min/max: {np.nanmin(velocity_estimate):.4f} / {np.nanmax(velocity_estimate):.4f} m/s")

    print()
    print("First 10 center-pixel depth values:")
    print(np.array2string(depth_trace[:10], precision=4, suppress_small=True))

    print()
    print("First 10 center-pixel valid detection fractions:")
    print(np.array2string(valid_fraction_trace[:10], precision=4, suppress_small=True))

    print()
    print("First 10 finite-difference velocity estimates:")
    print(np.array2string(velocity_estimate[:10], precision=4, suppress_small=True))

    print()

    return y, x, depth_trace, velocity_estimate


def analyze_center_histogram(pre, block_idx, y, x):
    all_histograms = pre["all_histograms"]
    bin_centers = pre["hist_bin_centers_depth_m"]

    hist = all_histograms[block_idx, y, x, :]

    total_counts = int(np.sum(hist))

    print("=== Center pixel mini-histogram interpretation ===")
    print(f"Block index: {block_idx}")
    print(f"Total histogram counts: {total_counts}")

    if total_counts == 0:
        print("No valid detections in this histogram.")
        print()
        return

    peak_idx = int(np.argmax(hist))
    peak_depth = float(bin_centers[peak_idx])
    peak_count = int(hist[peak_idx])

    print(f"Peak bin depth: {peak_depth:.4f} m")
    print(f"Peak bin count: {peak_count}")

    # Simple multi-peak heuristic:
    # Find bins with at least half the peak count.
    strong_bins = np.where(hist >= 0.5 * peak_count)[0]
    strong_depths = bin_centers[strong_bins]

    print(f"Strong bins >= 50% of peak count: {strong_bins.tolist()}")
    print("Strong-bin depths:")
    print(np.array2string(strong_depths, precision=4, suppress_small=True))

    if strong_bins.size >= 2:
        depth_span = float(np.max(strong_depths) - np.min(strong_depths))
        print(f"Strong-bin depth span: {depth_span:.4f} m")

        if depth_span > 0.5:
            print("Interpretation: possible broad or multi-surface return.")
        else:
            print("Interpretation: likely single surface or coarse-bin spread.")
    else:
        print("Interpretation: likely single dominant surface.")

    print()


def analyze_raw_timestamps(dataset_dir, metadata, y, x, block_idx=40, interp_steps=4):
    frame_path, frame = load_raw_block(dataset_dir, block_idx)

    timestamps_noisy_s = frame["timestamps_noisy_s"]
    sampled_depths_m = frame["sampled_depths_m"]
    detection_mask = frame["detection_mask"]

    c_light = metadata["c_light"]

    raw_ts = timestamps_noisy_s[:, y, x]
    valid_ts = raw_ts[np.isfinite(raw_ts)]

    raw_sampled_depths = sampled_depths_m[:, y, x]
    raw_detected_depths = 0.5 * c_light * valid_ts

    time_ms = block_idx * metadata["dt_s"] * 1e3

    render_pair_start = (block_idx // interp_steps) + 1
    render_pair_end = render_pair_start + 1
    interp_idx = block_idx % interp_steps

    print("=== Raw timestamp interpretation ===")
    print(f"Zero-based block index: {block_idx}")
    print(f"Saved file: {frame_path.name}")
    print(f"Approx. block time: {time_ms:.3f} ms")
    print(
        f"Approx. rendered frame transition: "
        f"{render_pair_start} -> {render_pair_end}"
    )
    print(f"Interpolation sub-step: {interp_idx} / {interp_steps - 1}")
    print()
    print(f"timestamps_noisy_s[:, y, x] length: {raw_ts.size}")
    print(f"Detected photons: {valid_ts.size} / {raw_ts.size}")
    print(f"Detection fraction from mask: {np.mean(detection_mask[:, y, x]):.4f}")

    print()
    print("Raw sampled geometric depth:")
    print(f"Mean: {np.nanmean(raw_sampled_depths):.4f} m")
    print(
        f"Min/max: "
        f"{np.nanmin(raw_sampled_depths):.4f} / "
        f"{np.nanmax(raw_sampled_depths):.4f} m"
    )

    if valid_ts.size > 0:
        print()
        print("Detected timestamp depths:")
        print(f"Mean: {np.mean(raw_detected_depths):.4f} m")
        print(
            f"Min/max: "
            f"{np.min(raw_detected_depths):.4f} / "
            f"{np.max(raw_detected_depths):.4f} m"
        )

        print()
        print("First few valid timestamps converted to depth:")
        print(
            np.array2string(
                raw_detected_depths[:10],
                precision=4,
                suppress_small=True,
            )
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Read and interpret timestamp_gen dataset."
    )

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to timestamp_gen dataset directory.",
    )

    parser.add_argument(
        "--block",
        type=int,
        default=40,
        help=(
            "Zero-based timestamp block index to inspect. "
            "Default: 40, which corresponds to frame_000041.npz."
        ),
    )

    parser.add_argument(
        "--interp-steps",
        type=int,
        default=4,
        help=(
            "Number of interpolation steps used when generating timestamps. "
            "Default: 4."
        ),
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    block_idx = args.block
    interp_steps = args.interp_steps

    metadata = load_metadata(dataset_dir)
    pre = load_precomputed(dataset_dir)

    summarize_metadata(metadata)

    y, x, depth_trace, velocity_estimate = analyze_center_pixel_depth(
        pre=pre,
        metadata=metadata,
    )

    print("=== Selected block mapping ===")
    render_pair_start = (block_idx // interp_steps) + 1
    render_pair_end = render_pair_start + 1
    interp_idx = block_idx % interp_steps
    saved_file_number = block_idx + 1
    time_ms = block_idx * metadata["dt_s"] * 1e3

    print(f"Zero-based block index: {block_idx}")
    print(f"Saved file: frame_{saved_file_number:06d}.npz")
    print(f"Approx. block time: {time_ms:.3f} ms")
    print(f"Approx. rendered frame transition: {render_pair_start} -> {render_pair_end}")
    print(f"Interpolation sub-step: {interp_idx} / {interp_steps - 1}")
    print()

    analyze_center_histogram(
        pre=pre,
        block_idx=block_idx,
        y=y,
        x=x,
    )

    analyze_raw_timestamps(
        dataset_dir=dataset_dir,
        metadata=metadata,
        y=y,
        x=x,
        block_idx=block_idx,
        interp_steps=interp_steps,
    )


if __name__ == "__main__":
    main()