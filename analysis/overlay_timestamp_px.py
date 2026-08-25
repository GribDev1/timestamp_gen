from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_pixel(value: str) -> tuple[int, int]:
    """
    Convert a command-line pixel such as "4,4" into (4, 4).
    """
    try:
        pixel_y, pixel_x = value.split(",")
        return int(pixel_y), int(pixel_x)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid pixel '{value}'. Use y,x, for example 4,4."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay timestamp-versus-time plots from multiple ToF pixels."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "Timestamp dataset directory containing metadata.json "
            "and frames/."
        ),
    )

    parser.add_argument(
        "--pixel",
        type=parse_pixel,
        action="append",
        required=True,
        help=(
            "Pixel to plot, written as y,x. Repeat this argument "
            "to overlay multiple pixels."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )

    parser.add_argument(
        "--start-time-ms",
        type=float,
        default=None,
        help="Optional beginning of the plotted time window.",
    )

    parser.add_argument(
        "--end-time-ms",
        type=float,
        default=None,
        help="Optional end of the plotted time window.",
    )

    parser.add_argument(
        "--y-min-ns",
        type=float,
        default=0.7,
        help="Lower y-axis limit in nanoseconds. Default: 0.7",
    )

    parser.add_argument(
        "--y-max-ns",
        type=float,
        default=26.7,
        help="Upper y-axis limit in nanoseconds. Default: 26.7",
    )

    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.1,
        help="Scatter marker area. Default: 0.1",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Marker opacity between 0 and 1. Default: 0.5",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution. Default: 300",
    )

    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(
    args: argparse.Namespace,
    metadata: dict,
) -> None:
    if args.y_min_ns >= args.y_max_ns:
        raise ValueError(
            "--y-min-ns must be less than --y-max-ns."
        )

    if (
        args.start_time_ms is not None
        and args.end_time_ms is not None
        and args.start_time_ms >= args.end_time_ms
    ):
        raise ValueError(
            "--start-time-ms must be less than --end-time-ms."
        )

    if args.marker_size <= 0:
        raise ValueError("--marker-size must be positive.")

    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must be greater than 0 and at most 1.")

    tof_height = int(metadata["tof_h"])
    tof_width = int(metadata["tof_w"])

    for pixel_y, pixel_x in args.pixel:
        if not (
            0 <= pixel_y < tof_height
            and 0 <= pixel_x < tof_width
        ):
            raise ValueError(
                f"Pixel ({pixel_y}, {pixel_x}) is outside the "
                f"{tof_height}x{tof_width} sensor."
            )


def load_overlaid_timestamps(
    dataset_dir: Path,
    metadata: dict,
    pixels: list[tuple[int, int]],
    start_time_ms: float | None,
    end_time_ms: float | None,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    frames_dir = dataset_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npz"))

    if not frame_files:
        raise RuntimeError(
            f"No timestamp frame files found in {frames_dir}"
        )

    laser_rate_hz = float(metadata["laser_rate_hz"])
    block_size = int(metadata["block_size_L"])
    dt_s = float(metadata["dt_s"])

    block_duration_s = block_size / laser_rate_hz

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

    first_index = 0
    last_index = len(frame_files)

    if start_time_s is not None:
        first_index = max(
            0,
            int(np.floor(start_time_s / dt_s)) - 1,
        )

    if end_time_s is not None:
        last_index = min(
            len(frame_files),
            int(np.ceil(end_time_s / dt_s)) + 1,
        )

    selected_files = frame_files[first_index:last_index]

    print(
        f"Reading {len(selected_files):,} of "
        f"{len(frame_files):,} timestamp frame files."
    )

    pulse_offsets_s = (
        np.arange(block_size, dtype=np.float64)
        / laser_rate_hz
    )

    pixel_times = {
        pixel: []
        for pixel in pixels
    }

    pixel_timestamps = {
        pixel: []
        for pixel in pixels
    }

    for local_index, frame_path in enumerate(
        selected_files,
        start=first_index,
    ):
        frame_number = local_index + 1
        block_end_time_s = frame_number * dt_s
        block_start_time_s = (
            block_end_time_s - block_duration_s
        )

        pulse_times_s = (
            block_start_time_s + pulse_offsets_s
        )

        time_valid = np.ones(
            block_size,
            dtype=bool,
        )

        if start_time_s is not None:
            time_valid &= pulse_times_s >= start_time_s

        if end_time_s is not None:
            time_valid &= pulse_times_s <= end_time_s

        if not np.any(time_valid):
            continue

        with np.load(frame_path) as frame:
            noisy = frame["timestamps_noisy_s"]

            for pixel in pixels:
                pixel_y, pixel_x = pixel
                timestamps_s = noisy[
                    :,
                    pixel_y,
                    pixel_x,
                ]

                valid = (
                    time_valid
                    & np.isfinite(timestamps_s)
                )

                if not np.any(valid):
                    continue

                pixel_times[pixel].append(
                    pulse_times_s[valid]
                )

                pixel_timestamps[pixel].append(
                    timestamps_s[valid]
                )

    results = {}

    for pixel in pixels:
        if pixel_times[pixel]:
            results[pixel] = (
                np.concatenate(pixel_times[pixel]),
                np.concatenate(pixel_timestamps[pixel]),
            )
        else:
            results[pixel] = (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

    return results


def save_overlay_plot(
    results: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray],
    ],
    output_path: Path,
    y_min_ns: float,
    y_max_ns: float,
    marker_size: float,
    alpha: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(14, 7),
    )

    colors = plt.get_cmap("tab10")

    total_timestamps = 0

    for color_index, (
        pixel,
        values,
    ) in enumerate(results.items()):
        pulse_times_s, timestamps_s = values
        pixel_y, pixel_x = pixel

        if timestamps_s.size == 0:
            print(
                f"Pixel y={pixel_y}, x={pixel_x}: "
                "no detected timestamps."
            )
            continue

        total_timestamps += timestamps_s.size

        axis.scatter(
            pulse_times_s * 1e3,
            timestamps_s * 1e9,
            s=marker_size,
            alpha=alpha,
            linewidths=0,
            color=colors(color_index % 10),
            label=(
                f"y={pixel_y}, x={pixel_x} "
                f"({timestamps_s.size:,})"
            ),
        )

        print(
            f"Pixel y={pixel_y}, x={pixel_x}: "
            f"{timestamps_s.size:,} timestamps."
        )

    if total_timestamps == 0:
        raise RuntimeError(
            "None of the selected pixels contained timestamps "
            "in the requested time window."
        )

    axis.set_xlabel("Simulation time (ms)")
    axis.set_ylabel(
        "Photon round-trip timestamp (ns)"
    )

    axis.set_title(
        "Detected timestamps from multiple ToF pixels"
    )

    axis.set_ylim(
        y_min_ns,
        y_max_ns,
    )

    axis.grid(alpha=0.25)
    axis.legend(
        markerscale=20,
        loc="best",
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
    )
    plt.close(figure)

    print(
        f"Saved {total_timestamps:,} overlaid timestamps to "
        f"{output_path}"
    )


def main() -> None:
    args = parse_args()

    # Remove duplicate pixels while retaining their entered order.
    pixels = list(dict.fromkeys(args.pixel))

    metadata = load_metadata(args.dataset_dir)
    validate_inputs(args, metadata)

    results = load_overlaid_timestamps(
        dataset_dir=args.dataset_dir,
        metadata=metadata,
        pixels=pixels,
        start_time_ms=args.start_time_ms,
        end_time_ms=args.end_time_ms,
    )

    save_overlay_plot(
        results=results,
        output_path=args.output,
        y_min_ns=args.y_min_ns,
        y_max_ns=args.y_max_ns,
        marker_size=args.marker_size,
        alpha=args.alpha,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()