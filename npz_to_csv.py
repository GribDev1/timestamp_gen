"""
Convert timestamp_gen NPZ outputs into visualization-friendly CSV files.

Recommended usage:
    # Export compact block-level data for one pixel
    python npz_to_csv.py outputs/examples/drone_flyby \
        --mode pixel-summary --pixel-y 4 --pixel-x 4

    # Export every detected photon for one pixel
    python npz_to_csv.py outputs/examples/drone_flyby \
        --mode pixel-photons --pixel-y 4 --pixel-x 4

    # Export block-level depth and valid-fraction values for every pixel
    python npz_to_csv.py outputs/examples/drone_flyby \
        --mode all-pixel-summary

Notes:
    - Avoid exporting every pulse for every pixel unless the dataset is very small.
    - CSV is much larger and slower than NPZ. Use CSV mainly for selected pixels,
      selected time windows, or compact block-level summaries.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert timestamp_gen NPZ outputs to CSV"
    )
    
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory containing metadata.json, timestamp_precomputed.npz, and frames/."
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="",
    )
    
    parser.add_argument(
        "--mode",
        choices=[
            "pixel-summary",
            "pixel-photons",
            "all-pixel-summary",
            "frame-photons",
        ],
        default="pixel-summary",
        help=(
            "pixel-summary: one row per block for one pixel; "
            "pixel-photons: one row per detected photon for one pixel; "
            "all-pixel-summary: one row per block per pixel; "
            "frame-photons: detected photons from one saved frame."
        ),
    )
    
    parser.add_argument("--pixel-y", type=int, default=None)
    parser.add_argument("--pixel-x", type=int, default=None)
    
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="One-based saved frame number for frame-photon mode.",
    )
    
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Minimum block end time in seconds.",
    )
    
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Maximum block end time in seconds.",
    )
    
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Minimum one-based saved frame number.",
    )
    
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Maximum one-based saved frame number.",
    )
    
    parser.add_argument(
        "--include-missed",
        action="store_true",
        help="Include missed detections in photon exports.",
    )
    
    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
    
def load_precomputed(dataset_dir: Path) -> np.lib.npyio.NpzFile:
    path = dataset_dir / "timestamp_precomputed.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing precomputed file: {path}")
    
    return np.load(path)


def get_block_times_s(pre: np.lib.npyio.NpzFile, metadata: dict) -> np.ndarray:
    num_blocks = pre["tof_depths"].shape[0]
    
    if "tof_block_times_s" in pre.files:
        return np.asarray(pre["tof_block_times_s"], dtype=np.float64)
    
    # Backward-compatible fallback. Times are block-end times.
    dt_s = float(metadata["dt_s"])
    return np.arange(1, num_blocks + 1, dtype=np.float64) * dt_s


def validate_pixel(metadata: dict, pixel_y: int | None, pixel_x: int | None) -> tuple[int, int]:
    tof_h = int(metadata["tof_h"])
    tof_w = int(metadata["tof_w"])
    
    y = tof_h // 2 if pixel_y is None else pixel_y
    x = tof_w // 2 if pixel_x is None else pixel_x
    
    if not (0 <= y < tof_h and 0 <= x < tof_w):
        raise ValueError(f"Pixel ({y}, {x}) is outside ToF grid {tof_h}x{tof_w}.")
    
    return y, x


def block_selected(
    frame_number: int,
    block_end_time_s: float,
    args: argparse.Namespace,
) -> bool:
    if args.start_frame is not None and frame_number < args.start_frame:
        return False
    if args.end_frame is not None and frame_number > args.end_frame:
        return False
    if args.start_time is not None and block_end_time_s < args.start_time:
        return False
    if args.end_time is not None and block_end_time_s > args.end_time:
        return False
    return True


def frame_files(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "frames").glob("frame_*.npz"))
    if not files:
        raise RuntimeError(f"No frame NPZ files found in {dataset_dir / 'frames'}")
    return files


def frame_time_s(frame: np.lib.npyio.NpzFile, frame_number: int, metadata: dict) -> float:
    if "simulation_time_s" in frame.files:
        return float(frame["simulation_time_s"])

    return frame_number * float(metadata["dt_s"])


def export_pixel_summary(
    dataset_dir: Path,
    output_dir: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> Path:
    pre = load_precomputed(dataset_dir)
    y, x = validate_pixel(metadata, args.pixel_y, args.pixel_x)
    
    tof_depths = pre["tof_depths"]
    all_I = pre["all_I"]
    times_s = get_block_times_s(pre, metadata)
    
    output_path = output_dir / f"pixel_summary_y{y}_x{x}.csv"
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_number",
            "block_end_time_s",
            "block_end_time_ms",
            "pixel_y",
            "pixel_x",
            "depth_estimate_m",
            "valid_detection_fraction",
        ])

        for block_idx, block_end_time_s in enumerate(times_s):
            frame_number = block_idx + 1

            if not block_selected(frame_number, float(block_end_time_s), args):
                continue

            writer.writerow([
                frame_number,
                f"{block_end_time_s:.12g}",
                f"{block_end_time_s * 1e3:.12g}",
                y,
                x,
                float(tof_depths[block_idx, y, x]),
                float(all_I[block_idx, y, x]),
            ])

    return output_path

def export_all_pixel_summary(
    dataset_dir: Path,
    output_dir: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> Path:
    pre = load_precomputed(dataset_dir)
    
    tof_depths = pre["tof_depths"]
    all_I = pre["all_I"]
    times_s = get_block_times_s(pre, metadata)
    
    _, tof_h, tof_w = tof_depths.shape
    output_path = output_dir / "all_pixel_summary.csv"
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_number",
            "block_end_time_s",
            "block_end_time_ms",
            "pixel_y",
            "pixel_x",
            "depth_estimate_m",
            "valid_detection_fraction",
        ])

        for block_idx, block_end_time_s in enumerate(times_s):
            frame_number = block_idx + 1

            if not block_selected(frame_number, float(block_end_time_s), args):
                continue

            for y in range(tof_h):
                for x in range(tof_w):
                    writer.writerow([
                        frame_number,
                        f"{block_end_time_s:.12g}",
                        f"{block_end_time_s * 1e3:.12g}",
                        y,
                        x,
                        float(tof_depths[block_idx, y, x]),
                        float(all_I[block_idx, y, x]),
                    ])

    return output_path


def iter_selected_frames(
    dataset_dir: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> Iterator[tuple[Path, np.lib.npyio.NpzFile, int, float]]:
    for path in frame_files(dataset_dir):
        frame = np.load(path)
        frame_number = int(frame["frame_number"])
        block_end_time_s = frame_time_s(frame, frame_number, metadata)
        
        if block_selected(frame_number, block_end_time_s, args):
            yield path, frame, frame_number, block_end_time_s
        else:
            frame.close()
            
    
def write_photon_rows(
    writer: csv.writer,
    frame: np.lib.npyio.NpzFile,
    frame_number: int,
    block_end_time_s: float,
    metadata: dict,
    y: int,
    x: int,
    include_missed: bool,
) -> int:
    timestamps = frame["timestamps_noisy_s"][:, y, x]
    clean = frame["timestamps_clean_s"][:, y, x]
    sampled_depths = frame["sampled_depths_m"][:, y, x]
    detection_mask = frame["detection_mask"][:, y, x].astype(bool)

    laser_rate_hz = float(metadata["laser_rate_hz"])
    block_size_L = int(metadata["block_size_L"])
    c_light = float(metadata["c_light"])
    block_duration_s = block_size_L / laser_rate_hz
    block_start_time_s = block_end_time_s - block_duration_s

    rows_written = 0

    for pulse_idx in range(block_size_L):
        detected = bool(detection_mask[pulse_idx])

        if not include_missed and not detected:
            continue

        pulse_time_s = block_start_time_s + pulse_idx / laser_rate_hz
        tau_noisy_s = float(timestamps[pulse_idx])
        tau_clean_s = float(clean[pulse_idx])
        sampled_depth_m = float(sampled_depths[pulse_idx])

        detected_depth_m = (
            0.5 * c_light * tau_noisy_s
            if np.isfinite(tau_noisy_s)
            else np.nan
        )

        writer.writerow([
            frame_number,
            f"{block_end_time_s:.12g}",
            pulse_idx,
            f"{pulse_time_s:.12g}",
            f"{pulse_time_s * 1e3:.12g}",
            y,
            x,
            int(detected),
            tau_clean_s,
            tau_noisy_s,
            tau_noisy_s * 1e9 if np.isfinite(tau_noisy_s) else np.nan,
            sampled_depth_m,
            detected_depth_m,
        ])
        rows_written += 1

    return rows_written


def photon_header() -> list[str]:
    return [
        "frame_number",
        "block_end_time_s",
        "pulse_index",
        "pulse_time_s",
        "pulse_time_ms",
        "pixel_y",
        "pixel_x",
        "detected",
        "timestamp_clean_s",
        "timestamp_noisy_s",
        "timestamp_noisy_ns",
        "sampled_depth_m",
        "detected_depth_m",
    ]
    
    
def export_pixel_photons(
    dataset_dir: Path,
    output_dir: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> Path:
    y, x = validate_pixel(metadata, args.pixel_y, args.pixel_x)
    output_path = output_dir / f"pixel_photons_y{y}_x{x}.csv"
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(photon_header())
        
        for _, frame, frame_number, block_end_time_s in iter_selected_frames(
            dataset_dir, metadata, args
        ):
            write_photon_rows(
                writer=writer,
                frame=frame,
                block_end_time_s=block_end_time_s,
                metadata=metadata,
                y=y,
                x=x,
                include_missed=args.include_missed,
            )
            frame.close()
            
    return output_path

def export_frame_photons(
    dataset_dir: Path,
    output_dir: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> Path:
    if args.frame is None:
        raise ValueError("--frame is required for frame-photons mode.")

    path = dataset_dir / "frames" / f"frame_{args.frame:06d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame file not found: {path}")

    frame = np.load(path)
    frame_number = int(frame["frame_number"])
    block_end_time_s = frame_time_s(frame, frame_number, metadata)

    tof_h = int(metadata["tof_h"])
    tof_w = int(metadata["tof_w"])
    output_path = output_dir / f"frame_{frame_number:06d}_photons.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(photon_header())

        for y in range(tof_h):
            for x in range(tof_w):
                write_photon_rows(
                    writer=writer,
                    frame=frame,
                    frame_number=frame_number,
                    block_end_time_s=block_end_time_s,
                    metadata=metadata,
                    y=y,
                    x=x,
                    include_missed=args.include_missed,
                )

    frame.close()
    return output_path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset_dir / "csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(dataset_dir)

    if args.mode == "pixel-summary":
        output_path = export_pixel_summary(
            dataset_dir, output_dir, metadata, args
        )
    elif args.mode == "pixel-photons":
        output_path = export_pixel_photons(
            dataset_dir, output_dir, metadata, args
        )
    elif args.mode == "all-pixel-summary":
        output_path = export_all_pixel_summary(
            dataset_dir, output_dir, metadata, args
        )
    elif args.mode == "frame-photons":
        output_path = export_frame_photons(
            dataset_dir, output_dir, metadata, args
        )
    else:
        raise RuntimeError(f"Unsupported mode: {args.mode}")

    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()