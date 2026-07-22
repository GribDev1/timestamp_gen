"""
Compute simple per-pixel time-to-contact from timestamp_precomputed.npz.

Uses only:
    - tof_depths
    - metadata.json

Math:
    radial_velocity = (depth[k] - depth[k-window]) / (time[k] - time[k-window])
    closing_speed = -radial_velocity
    TTC = depth[k] / closing_speed

TTC is valid only when closing_speed > min_closing_speed.

Example:
    python compute_ttc.py ^
        --input outputs/examples/wall_approach/timestamp_precomputed.npz ^
        --output outputs/examples/wall_approach/ttc_simple.npz ^
        --window-ms 10
"""

from pathlib import Path
import argparse
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute simple per-pixel time-to-contact from ToF depth."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to timestamp_precomputed.npz.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save TTC results as .npz.",
    )

    parser.add_argument(
        "--window-ms",
        type=float,
        default=10.0,
        help="Depth-difference window in milliseconds. Default: 10.",
    )

    parser.add_argument(
        "--min-closing-speed",
        type=float,
        default=0.1,
        help="Minimum positive closing speed in m/s. Default: 0.1.",
    )

    parser.add_argument(
        "--max-ttc-s",
        type=float,
        default=10.0,
        help="Maximum TTC kept in seconds. Default: 10.",
    )

    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_times_s(data, metadata, num_blocks):
    """
    Return one time value per timestamp block.

    Prefer saved block timing. Otherwise use metadata["dt_s"].
    """
    if "block_start_time_s" in data and "block_end_time_s" in data:
        block_start = np.asarray(data["block_start_time_s"], dtype=np.float64)
        block_end = np.asarray(data["block_end_time_s"], dtype=np.float64)

        if len(block_start) != num_blocks or len(block_end) != num_blocks:
            raise ValueError("Block timing arrays do not match tof_depths.")

        return 0.5 * (block_start + block_end)

    dt_s = float(metadata["dt_s"])
    return np.arange(num_blocks, dtype=np.float64) * dt_s


def compute_ttc(
    depths_m,
    times_s,
    window_ms,
    min_closing_speed,
    max_ttc_s,
):
    """
    Compute radial velocity, closing speed, and TTC.

    depths_m shape:
        [num_blocks, tof_h, tof_w]
    """
    if window_ms <= 0:
        raise ValueError("window_ms must be positive.")

    median_dt_s = float(np.median(np.diff(times_s)))
    window_blocks = max(
        1,
        int(round((window_ms * 1e-3) / median_dt_s)),
    )

    radial_velocity_mps = np.full_like(depths_m, np.nan, dtype=np.float32)
    closing_speed_mps = np.full_like(depths_m, np.nan, dtype=np.float32)
    ttc_s = np.full_like(depths_m, np.nan, dtype=np.float32)

    current_depth = depths_m[window_blocks:]
    previous_depth = depths_m[:-window_blocks]

    delta_t_s = (
        times_s[window_blocks:] - times_s[:-window_blocks]
    )[:, None, None]

    valid = (
        np.isfinite(current_depth)
        & np.isfinite(previous_depth)
        & (delta_t_s > 0)
    )

    velocity = np.full_like(current_depth, np.nan, dtype=np.float64)
    velocity[valid] = (
        current_depth[valid] - previous_depth[valid]
    ) / np.broadcast_to(delta_t_s, current_depth.shape)[valid]

    closing = -velocity

    valid_ttc = (
        valid
        & np.isfinite(closing)
        & (closing > min_closing_speed)
    )

    ttc = np.full_like(current_depth, np.nan, dtype=np.float64)
    ttc[valid_ttc] = current_depth[valid_ttc] / closing[valid_ttc]

    valid_ttc &= (
        np.isfinite(ttc)
        & (ttc > 0)
        & (ttc <= max_ttc_s)
    )

    ttc[~valid_ttc] = np.nan
    closing[~valid] = np.nan

    radial_velocity_mps[window_blocks:] = velocity.astype(np.float32)
    closing_speed_mps[window_blocks:] = closing.astype(np.float32)
    ttc_s[window_blocks:] = ttc.astype(np.float32)

    return radial_velocity_mps, closing_speed_mps, ttc_s, window_blocks


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    dataset_dir = args.input.parent
    metadata = load_metadata(dataset_dir)

    with np.load(args.input) as data:
        tof_depths = np.asarray(data["tof_depths"], dtype=np.float32)
        times_s = get_times_s(data, metadata, tof_depths.shape[0])

    (
        radial_velocity_mps,
        closing_speed_mps,
        ttc_s,
        window_blocks,
    ) = compute_ttc(
        depths_m=tof_depths,
        times_s=times_s,
        window_ms=args.window_ms,
        min_closing_speed=args.min_closing_speed,
        max_ttc_s=args.max_ttc_s,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        times_s=times_s,
        depths_m=tof_depths,
        radial_velocity_mps=radial_velocity_mps,
        closing_speed_mps=closing_speed_mps,
        ttc_s=ttc_s,
        window_ms=np.float32(args.window_ms),
        window_blocks=np.int32(window_blocks),
        min_closing_speed_mps=np.float32(args.min_closing_speed),
        max_ttc_s=np.float32(args.max_ttc_s),
    )

    valid_ttc_count = int(np.count_nonzero(np.isfinite(ttc_s)))

    print(f"Loaded: {args.input}")
    print(f"Depth shape: {tof_depths.shape}")
    print(f"Window: {args.window_ms:.3f} ms = {window_blocks} blocks")
    print(f"Valid TTC values: {valid_ttc_count:,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()