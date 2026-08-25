from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.ttc_utils import compute_time_to_contact
from analysis.visualization_gifs import (
    save_depth_gif,
    save_pixel_histogram_gif,
    save_valid_fraction_gif,
    validate_gif_options,
)
from analysis.visualization_io import (
    build_block_times_s,
    load_metadata,
    load_timestamp_data,
    load_ttc_results,
    save_ttc_results,
)
from analysis.visualization_pixels import (
    save_pixel_outputs,
    save_timestamps_vs_time,
)
from analysis.visualization_summary import (
    save_block_dashboard,
    save_summary_over_time,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize timestamp generator outputs."
    )

    parser.add_argument(
        "--mode",
        choices=(
            "summary",
            "pixel",
            "timestamps",
            "depth-gif",
            "valid-gif",
            "histogram-gif",
        ),
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ttc-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pixel-y",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pixel-x",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--gif-stride",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--ttc-window-ms",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--ttc-min-closing-speed",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--ttc-max-s",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--ttc-warning-s",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--start-time-ms",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--end-time-ms",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--timestamp-marker-size",
        type=float,
        default=2.0,
    )
    
    parser.add_argument(
        "--timestamp-y-min-ns",
        type=float,
        default=0.7,
        help="Lower timestamp-axis limit in nanoseconds. Default: 0.7",
    )

    parser.add_argument(
        "--timestamp-y-max-ns",
        type=float,
        default=26.7,
        help="Upper timestamp-axis limit in nanoseconds. Default: 26.7",
    )

    return parser.parse_args()


def validate_pixel(
    pixel_y: int | None,
    pixel_x: int | None,
    tof_h: int,
    tof_w: int,
) -> tuple[int, int]:
    if pixel_y is None or pixel_x is None:
        raise ValueError(
            "This mode requires --pixel-y and --pixel-x."
        )

    if not (0 <= pixel_y < tof_h):
        raise ValueError(
            f"pixel-y must be between 0 and {tof_h - 1}."
        )

    if not (0 <= pixel_x < tof_w):
        raise ValueError(
            f"pixel-x must be between 0 and {tof_w - 1}."
        )

    return pixel_y, pixel_x


def main() -> None:
    args = parse_args()

    dataset_dir = args.input.parent
    metadata_path = dataset_dir / "metadata.json"

    metadata = (
        load_metadata(dataset_dir)
        if metadata_path.exists()
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "timestamps":
        if metadata is None:
            raise FileNotFoundError(
                "Timestamp-versus-time mode requires metadata.json."
            )

        tof_h = int(metadata["tof_h"])
        tof_w = int(metadata["tof_w"])

        pixel_y, pixel_x = validate_pixel(
            args.pixel_y,
            args.pixel_x,
            tof_h,
            tof_w,
        )

        output_path = (
            args.output_dir
            / f"timestamps_vs_time_y{pixel_y}_x{pixel_x}.png"
        )

        save_timestamps_vs_time(
            dataset_dir=dataset_dir,
            metadata=metadata,
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            output_path=output_path,
            start_time_ms=args.start_time_ms,
            end_time_ms=args.end_time_ms,
            marker_size=args.timestamp_marker_size,
            y_min_ns=args.timestamp_y_min_ns,
            y_max_ns=args.timestamp_y_max_ns,
        )

        print(f"Completed visualization mode: {args.mode}")
        print(f"Output directory: {args.output_dir}")
        return

    data = load_timestamp_data(args.input)

    tof_depths = data["tof_depths"]
    all_I = data["all_I"]
    all_histograms = data["all_histograms"]
    hist_bin_centers_depth_m = data[
        "hist_bin_centers_depth_m"
    ]

    num_blocks, tof_h, tof_w = tof_depths.shape

    if not (0 <= args.frame < num_blocks):
        raise ValueError(
            f"Frame {args.frame} is outside valid range "
            f"0 to {num_blocks - 1}."
        )

    block_times_s = build_block_times_s(
        data=data,
        metadata=metadata,
        num_blocks=num_blocks,
    )

    if args.mode == "summary":
        ttc_results = compute_time_to_contact(
            tof_depths=tof_depths,
            block_times_s=block_times_s,
            window_ms=args.ttc_window_ms,
            min_closing_speed_mps=(
                args.ttc_min_closing_speed
            ),
            max_ttc_s=args.ttc_max_s,
        )

        ttc_path = (
            args.ttc_file
            if args.ttc_file is not None
            else args.output_dir / "time_to_contact.npz"
        )

        save_ttc_results(
            output_path=ttc_path,
            block_times_s=block_times_s,
            ttc_results=ttc_results,
            window_ms=args.ttc_window_ms,
            min_closing_speed_mps=(
                args.ttc_min_closing_speed
            ),
            max_ttc_s=args.ttc_max_s,
        )

        save_summary_over_time(
            tof_depths=tof_depths,
            all_I=all_I,
            block_times_s=block_times_s,
            output_path=(
                args.output_dir / "summary_over_time.png"
            ),
        )

        dashboard_y = (
            args.pixel_y
            if args.pixel_y is not None
            else tof_h // 2
        )
        dashboard_x = (
            args.pixel_x
            if args.pixel_x is not None
            else tof_w // 2
        )

        save_block_dashboard(
            tof_depths=tof_depths,
            all_I=all_I,
            all_histograms=all_histograms,
            hist_bin_centers_depth_m=(
                hist_bin_centers_depth_m
            ),
            ttc=ttc_results["time_to_contact_s"],
            frame_idx=args.frame,
            pixel_y=dashboard_y,
            pixel_x=dashboard_x,
            max_ttc_s=args.ttc_max_s,
            output_path=(
                args.output_dir
                / f"dashboard_block_{args.frame:06d}.png"
            ),
        )

    elif args.mode == "pixel":
        pixel_y, pixel_x = validate_pixel(
            args.pixel_y,
            args.pixel_x,
            tof_h,
            tof_w,
        )

        if args.ttc_file is None:
            raise ValueError(
                "Pixel mode requires --ttc-file."
            )

        ttc_results = load_ttc_results(args.ttc_file)

        save_pixel_outputs(
            all_histograms=all_histograms,
            hist_bin_centers_depth_m=(
                hist_bin_centers_depth_m
            ),
            block_times_s=block_times_s,
            ttc_results=ttc_results,
            frame_idx=args.frame,
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            output_dir=args.output_dir,
            warning_s=args.ttc_warning_s,
            max_ttc_s=args.ttc_max_s,
        )

    elif args.mode == "depth-gif":
        validate_gif_options(
            args.gif_fps,
            args.gif_stride,
        )

        save_depth_gif(
            tof_depths=tof_depths,
            output_path=(
                args.output_dir / "depth_over_time.gif"
            ),
            fps=args.gif_fps,
            gif_stride=args.gif_stride,
        )

    elif args.mode == "valid-gif":
        validate_gif_options(
            args.gif_fps,
            args.gif_stride,
        )

        save_valid_fraction_gif(
            all_I=all_I,
            output_path=(
                args.output_dir
                / "valid_fraction_over_time.gif"
            ),
            fps=args.gif_fps,
            gif_stride=args.gif_stride,
        )

    elif args.mode == "histogram-gif":
        validate_gif_options(
            args.gif_fps,
            args.gif_stride,
        )

        pixel_y, pixel_x = validate_pixel(
            args.pixel_y,
            args.pixel_x,
            tof_h,
            tof_w,
        )

        save_pixel_histogram_gif(
            all_histograms=all_histograms,
            hist_bin_centers_depth_m=(
                hist_bin_centers_depth_m
            ),
            pixel_y=pixel_y,
            pixel_x=pixel_x,
            output_path=(
                args.output_dir
                / f"histogram_y{pixel_y}_x{pixel_x}.gif"
            ),
            fps=args.gif_fps,
            gif_stride=args.gif_stride,
        )

    print(f"Completed visualization mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()