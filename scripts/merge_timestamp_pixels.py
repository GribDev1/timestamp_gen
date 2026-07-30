from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-pixel timestamp files into one sensor-grid dataset."
    )

    parser.add_argument(
        "--pixel-dir",
        type=Path,
        required=True,
        help="Directory containing pixel_y*_x*.npz files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Final merged timestamp_precomputed.npz path.",
    )

    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="ToF sensor height in pixels.",
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="ToF sensor width in pixels.",
    )

    return parser.parse_args()


def load_pixel(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pixel file: {path}")

    with np.load(path) as data:
        required_keys = {
            "pixel_y",
            "pixel_x",
            "tof_depths",
            "all_I",
            "all_histograms",
            "tof_block_times_s",
            "hist_bin_centers_tau",
            "hist_bin_centers_depth_m",
        }

        missing = required_keys.difference(data.files)

        if missing:
            raise KeyError(
                f"{path} is missing required arrays: {sorted(missing)}"
            )

        return {
            key: np.array(data[key], copy=True)
            for key in required_keys
        }


def main() -> None:
    args = parse_args()

    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be greater than zero.")

    if not args.pixel_dir.is_dir():
        raise FileNotFoundError(
            f"Pixel directory not found: {args.pixel_dir}"
        )

    first_path = args.pixel_dir / "pixel_y0_x0.npz"
    first = load_pixel(first_path)

    num_blocks = first["tof_depths"].shape[0]

    if first["tof_depths"].ndim != 1:
        raise ValueError(
            f"{first_path}: tof_depths must have shape [blocks], "
            f"got {first['tof_depths'].shape}"
        )

    if first["all_I"].shape != (num_blocks,):
        raise ValueError(
            f"{first_path}: all_I must have shape ({num_blocks},), "
            f"got {first['all_I'].shape}"
        )

    if first["all_histograms"].ndim != 2:
        raise ValueError(
            f"{first_path}: all_histograms must have shape [blocks, bins], "
            f"got {first['all_histograms'].shape}"
        )

    if first["all_histograms"].shape[0] != num_blocks:
        raise ValueError(
            f"{first_path}: histogram block count does not match tof_depths."
        )

    num_bins = first["all_histograms"].shape[1]

    block_times = first["tof_block_times_s"]
    bin_centers_tau = first["hist_bin_centers_tau"]
    bin_centers_depth_m = first["hist_bin_centers_depth_m"]

    if block_times.shape != (num_blocks,):
        raise ValueError(
            f"{first_path}: tof_block_times_s must have shape "
            f"({num_blocks},), got {block_times.shape}"
        )

    if bin_centers_tau.shape != (num_bins,):
        raise ValueError(
            f"{first_path}: hist_bin_centers_tau must have shape "
            f"({num_bins},), got {bin_centers_tau.shape}"
        )

    if bin_centers_depth_m.shape != (num_bins,):
        raise ValueError(
            f"{first_path}: hist_bin_centers_depth_m must have shape "
            f"({num_bins},), got {bin_centers_depth_m.shape}"
        )

    tof_depths = np.full(
        (num_blocks, args.height, args.width),
        np.nan,
        dtype=np.float32,
    )

    all_I = np.full(
        (num_blocks, args.height, args.width),
        np.nan,
        dtype=np.float32,
    )

    all_histograms = np.zeros(
        (num_blocks, args.height, args.width, num_bins),
        dtype=np.uint16,
    )

    expected_pixel_count = args.height * args.width
    loaded_pixel_count = 0

    for pixel_y in range(args.height):
        for pixel_x in range(args.width):
            path = (
                args.pixel_dir
                / f"pixel_y{pixel_y}_x{pixel_x}.npz"
            )

            pixel = load_pixel(path)

            stored_y = int(np.asarray(pixel["pixel_y"]).item())
            stored_x = int(np.asarray(pixel["pixel_x"]).item())

            if stored_y != pixel_y or stored_x != pixel_x:
                raise ValueError(
                    f"{path}: filename indicates y={pixel_y}, x={pixel_x}, "
                    f"but stored coordinates are y={stored_y}, x={stored_x}"
                )

            depths = pixel["tof_depths"]
            valid_fraction = pixel["all_I"]
            histograms = pixel["all_histograms"]
            times = pixel["tof_block_times_s"]

            if depths.shape != (num_blocks,):
                raise ValueError(
                    f"{path}: tof_depths shape mismatch: {depths.shape}"
                )

            if valid_fraction.shape != (num_blocks,):
                raise ValueError(
                    f"{path}: all_I shape mismatch: {valid_fraction.shape}"
                )

            if histograms.shape != (num_blocks, num_bins):
                raise ValueError(
                    f"{path}: all_histograms shape mismatch: "
                    f"{histograms.shape}"
                )

            if times.shape != (num_blocks,):
                raise ValueError(
                    f"{path}: tof_block_times_s shape mismatch: {times.shape}"
                )

            if not np.array_equal(times, block_times):
                raise ValueError(
                    f"{path}: timestamp block times do not match pixel_y0_x0."
                )

            if not np.array_equal(
                pixel["hist_bin_centers_tau"],
                bin_centers_tau,
            ):
                raise ValueError(
                    f"{path}: histogram timestamp bin centers do not match."
                )

            if not np.array_equal(
                pixel["hist_bin_centers_depth_m"],
                bin_centers_depth_m,
            ):
                raise ValueError(
                    f"{path}: histogram depth bin centers do not match."
                )

            tof_depths[:, pixel_y, pixel_x] = depths
            all_I[:, pixel_y, pixel_x] = valid_fraction
            all_histograms[:, pixel_y, pixel_x, :] = histograms

            loaded_pixel_count += 1

            print(
                f"Loaded {loaded_pixel_count}/{expected_pixel_count}: "
                f"y={pixel_y}, x={pixel_x}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        tof_depths=tof_depths,
        all_I=all_I,
        all_histograms=all_histograms,
        tof_block_times_s=block_times.astype(np.float64),
        hist_bin_centers_tau=bin_centers_tau.astype(np.float32),
        hist_bin_centers_depth_m=bin_centers_depth_m.astype(np.float32),
    )

    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(
            f"Merged output was not created correctly: {args.output}"
        )

    print()
    print("Merge completed successfully.")
    print(f"Output:         {args.output}")
    print(f"Pixels merged:  {loaded_pixel_count}")
    print(f"Blocks:         {num_blocks}")
    print(f"Histogram bins: {num_bins}")
    print(f"tof_depths:     {tof_depths.shape}")
    print(f"all_I:          {all_I.shape}")
    print(f"all_histograms: {all_histograms.shape}")


if __name__ == "__main__":
    main()