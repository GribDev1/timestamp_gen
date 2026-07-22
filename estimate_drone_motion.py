"""
Predicting Drone Motion:

The goal of this program is to predict the position and orientation of a drone by processing timestamps.

Workflow:
    1. Load timestamp_precomputed.npz and metadata.json.
    2. Keep only measurements up to a selected cutoff.
    3. Robustly combine selected ToF pixels into one depth trace.
    4. Fit a constant-velocity model over a recent causal fitting window.
    5. Predict the next 10 rendered frames by default.
    6. Save the predictions to CSV for later comparison with ground truth.
    
Example:
    python estimate_drone_motion.py ^
        --input outputs/examples/wall_approach/timestamp_precomputed.npz ^
        --output outputs/examples/wall_approach/predictions/prediction.csv ^
        --observe-end-ms 500 ^
        --fit-window-ms 100 ^
        --predict-render-frames 10 ^
        --render-fps 240
"""

from pathlib import Path
import argparse
import csv
import json

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict short-horizon 1D drone motion from ToF depth."
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
        help="Output CSV path.",
    )

    cutoff = parser.add_mutually_exclusive_group()
    cutoff.add_argument(
        "--observe-blocks",
        type=int,
        default=None,
        help="Use only the first N timestamp blocks.",
    )
    cutoff.add_argument(
        "--observe-end-block",
        type=int,
        default=None,
        help="Last observed timestamp block index, inclusive.",
    )
    cutoff.add_argument(
        "--observe-end-ms",
        type=float,
        default=None,
        help="Last observed simulation time in milliseconds.",
    )

    parser.add_argument(
        "--observe-start-ms",
        type=float,
        default=None,
        help="Optional first observation time in milliseconds.",
    )
    parser.add_argument(
        "--fit-window-ms",
        type=float,
        default=100.0,
        help="Recent causal interval used for the velocity fit. Default: 100 ms.",
    )
    parser.add_argument(
        "--predict-render-frames",
        type=int,
        default=10,
        help="Number of future rendered frames to predict. Default: 10.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=240.0,
        help="Rendered scene frame rate. Default: 240 Hz.",
    )

    parser.add_argument(
        "--pixel-y",
        type=int,
        default=None,
        help="Use one ToF pixel row. Must be paired with --pixel-x.",
    )
    parser.add_argument(
        "--pixel-x",
        type=int,
        default=None,
        help="Use one ToF pixel column. Must be paired with --pixel-y.",
    )

    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help=(
            "Reject pixel samples whose valid detection fraction is below this "
            "value. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--depth-min-m",
        type=float,
        default=None,
        help="Optional minimum accepted depth.",
    )
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=None,
        help="Optional maximum accepted depth.",
    )

    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_block_times(data: np.lib.npyio.NpzFile, metadata: dict, n: int) -> np.ndarray:
    """
    Return one representative acquisition time per timestamp block.

    Prefer saved block timing. Otherwise, fall back to metadata dt_s.
    """
    if "block_start_time_s" in data and "block_end_time_s" in data:
        starts = np.asarray(data["block_start_time_s"], dtype=np.float64)
        ends = np.asarray(data["block_end_time_s"], dtype=np.float64)

        if starts.shape[0] != n or ends.shape[0] != n:
            raise ValueError("Saved block timing arrays do not match tof_depths.")

        return 0.5 * (starts + ends)

    dt_s = float(metadata["dt_s"])
    return np.arange(n, dtype=np.float64) * dt_s


def select_depth_trace(
    tof_depths: np.ndarray,
    all_I: np.ndarray | None,
    pixel_y: int | None,
    pixel_x: int | None,
    min_valid_fraction: float,
    depth_min_m: float | None,
    depth_max_m: float | None,
) -> np.ndarray:
    """
    Build one robust radial-depth trace.

    If one pixel is selected, use that pixel.
    Otherwise, use the spatial median across all ToF pixels.
    """
    if (pixel_y is None) != (pixel_x is None):
        raise ValueError("--pixel-y and --pixel-x must be used together.")

    depths = np.asarray(tof_depths, dtype=np.float64).copy()

    valid = np.isfinite(depths)

    if depth_min_m is not None:
        valid &= depths >= depth_min_m

    if depth_max_m is not None:
        valid &= depths <= depth_max_m

    if all_I is not None and min_valid_fraction > 0:
        valid &= np.asarray(all_I) >= min_valid_fraction

    depths[~valid] = np.nan

    if pixel_y is not None and pixel_x is not None:
        _, h, w = depths.shape
        if not (0 <= pixel_y < h and 0 <= pixel_x < w):
            raise ValueError(
                f"Pixel ({pixel_y}, {pixel_x}) is outside ToF grid {h}x{w}."
            )
        return depths[:, pixel_y, pixel_x]

    return np.nanmedian(depths, axis=(1, 2))


def resolve_observation_mask(
    times_s: np.ndarray,
    observe_blocks: int | None,
    observe_end_block: int | None,
    observe_start_ms: float | None,
    observe_end_ms: float | None,
) -> np.ndarray:
    n = times_s.size
    mask = np.ones(n, dtype=bool)

    if observe_blocks is not None:
        if observe_blocks < 2:
            raise ValueError("--observe-blocks must be at least 2.")
        mask &= np.arange(n) < min(observe_blocks, n)

    elif observe_end_block is not None:
        if observe_end_block < 1:
            raise ValueError("--observe-end-block must be at least 1.")
        mask &= np.arange(n) <= min(observe_end_block, n - 1)

    elif observe_end_ms is not None:
        mask &= times_s <= observe_end_ms * 1e-3

    if observe_start_ms is not None:
        mask &= times_s >= observe_start_ms * 1e-3

    return mask


def fit_constant_velocity(
    times_s: np.ndarray,
    depths_m: np.ndarray,
    fit_window_ms: float,
) -> tuple[float, float, float, int, float]:
    """
    Fit depth(t) = intercept + radial_velocity * t.

    Returns:
        cutoff_time_s
        cutoff_depth_m
        radial_velocity_mps
        number_of_fit_samples
        fit_rmse_m
    """
    finite = np.isfinite(times_s) & np.isfinite(depths_m)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError("Not enough valid observed depth samples.")

    t_valid = times_s[finite]
    d_valid = depths_m[finite]

    cutoff_time_s = float(t_valid[-1])
    fit_start_s = cutoff_time_s - fit_window_ms * 1e-3

    fit_mask = finite & (times_s >= fit_start_s) & (times_s <= cutoff_time_s)

    if np.count_nonzero(fit_mask) < 2:
        raise RuntimeError(
            "The fitting window contains fewer than two valid depth samples."
        )

    t_fit = times_s[fit_mask]
    d_fit = depths_m[fit_mask]

    # Center time for improved numerical conditioning.
    t_centered = t_fit - cutoff_time_s
    slope, intercept_at_cutoff = np.polyfit(t_centered, d_fit, deg=1)

    fitted = intercept_at_cutoff + slope * t_centered
    rmse = float(np.sqrt(np.mean((d_fit - fitted) ** 2)))

    return (
        cutoff_time_s,
        float(intercept_at_cutoff),
        float(slope),
        int(t_fit.size),
        rmse,
    )


def write_prediction_csv(
    output_path: Path,
    cutoff_time_s: float,
    cutoff_depth_m: float,
    radial_velocity_mps: float,
    render_fps: float,
    predict_render_frames: int,
    blocks_per_render_frame: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    closing_speed_mps = -radial_velocity_mps

    rows = []
    for future_frame in range(1, predict_render_frames + 1):
        horizon_s = future_frame / render_fps
        prediction_time_s = cutoff_time_s + horizon_s
        predicted_depth_m = cutoff_depth_m + radial_velocity_mps * horizon_s

        ttc_s = np.nan
        if closing_speed_mps > 0:
            ttc_s = predicted_depth_m / closing_speed_mps

        rows.append(
            {
                "future_render_frame": future_frame,
                "equivalent_block_offset": future_frame * blocks_per_render_frame,
                "prediction_horizon_s": horizon_s,
                "prediction_time_s": prediction_time_s,
                "predicted_depth_m": predicted_depth_m,
                "predicted_radial_velocity_mps": radial_velocity_mps,
                "predicted_closing_speed_mps": closing_speed_mps,
                "predicted_ttc_s": ttc_s,
            }
        )

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.render_fps <= 0:
        raise ValueError("--render-fps must be positive.")

    if args.predict_render_frames <= 0:
        raise ValueError("--predict-render-frames must be positive.")

    if args.fit_window_ms <= 0:
        raise ValueError("--fit-window-ms must be positive.")

    dataset_dir = args.input.parent
    metadata = load_metadata(dataset_dir)

    with np.load(args.input) as data:
        tof_depths = np.asarray(data["tof_depths"])
        all_I = np.asarray(data["all_I"]) if "all_I" in data else None
        times_s = load_block_times(data, metadata, tof_depths.shape[0])

    depth_trace_m = select_depth_trace(
        tof_depths=tof_depths,
        all_I=all_I,
        pixel_y=args.pixel_y,
        pixel_x=args.pixel_x,
        min_valid_fraction=args.min_valid_fraction,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
    )

    observed = resolve_observation_mask(
        times_s=times_s,
        observe_blocks=args.observe_blocks,
        observe_end_block=args.observe_end_block,
        observe_start_ms=args.observe_start_ms,
        observe_end_ms=args.observe_end_ms,
    )

    if np.count_nonzero(observed) < 2:
        raise RuntimeError("The selected observation interval is too short.")

    observed_times_s = times_s[observed]
    observed_depths_m = depth_trace_m[observed]

    (
        cutoff_time_s,
        cutoff_depth_m,
        radial_velocity_mps,
        fit_sample_count,
        fit_rmse_m,
    ) = fit_constant_velocity(
        times_s=observed_times_s,
        depths_m=observed_depths_m,
        fit_window_ms=args.fit_window_ms,
    )

    laser_rate_hz = float(metadata["laser_rate_hz"])
    block_size_L = int(metadata["block_size_L"])
    block_rate_hz = laser_rate_hz / block_size_L
    blocks_per_render_frame = block_rate_hz / args.render_fps

    write_prediction_csv(
        output_path=args.output,
        cutoff_time_s=cutoff_time_s,
        cutoff_depth_m=cutoff_depth_m,
        radial_velocity_mps=radial_velocity_mps,
        render_fps=args.render_fps,
        predict_render_frames=args.predict_render_frames,
        blocks_per_render_frame=blocks_per_render_frame,
    )

    print(f"Input: {args.input}")
    print(f"Observed through: {cutoff_time_s * 1e3:.3f} ms")
    print(f"Estimated depth at cutoff: {cutoff_depth_m:.6f} m")
    print(f"Estimated radial velocity: {radial_velocity_mps:.6f} m/s")
    print(f"Estimated closing speed: {-radial_velocity_mps:.6f} m/s")
    print(f"Fit samples: {fit_sample_count:,}")
    print(f"Fit RMSE: {fit_rmse_m:.6f} m")
    print(f"Timestamp block rate: {block_rate_hz:.6f} Hz")
    print(
        "Timestamp blocks per rendered frame: "
        f"{blocks_per_render_frame:.9f}"
    )
    print(
        f"Prediction horizon: {args.predict_render_frames} render frames = "
        f"{args.predict_render_frames / args.render_fps * 1e3:.3f} ms"
    )
    print(f"Saved prediction: {args.output}")


if __name__ == "__main__":
    main()