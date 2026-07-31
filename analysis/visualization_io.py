from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_metadata(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_timestamp_data(input_path: Path) -> dict[str, np.ndarray]:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Timestamp input file not found: {input_path}"
        )

    required_keys = {
        "tof_depths",
        "all_I",
        "all_histograms",
        "hist_bin_centers_depth_m",
    }

    with np.load(input_path) as data:
        missing = required_keys.difference(data.files)

        if missing:
            raise KeyError(
                f"{input_path} is missing arrays: {sorted(missing)}"
            )

        result = {
            key: np.array(data[key], copy=True)
            for key in required_keys
        }

        # Copy optional timing arrays when available.
        for key in (
            "tof_block_times_s",
            "block_start_time_s",
            "block_end_time_s",
        ):
            if key in data.files:
                result[key] = np.array(data[key], copy=True)

    tof_depths = result["tof_depths"]
    all_I = result["all_I"]
    all_histograms = result["all_histograms"]

    if tof_depths.ndim != 3:
        raise ValueError(
            "tof_depths must have shape [blocks, height, width], "
            f"got {tof_depths.shape}"
        )

    if all_I.shape != tof_depths.shape:
        raise ValueError(
            f"all_I shape {all_I.shape} does not match "
            f"tof_depths shape {tof_depths.shape}"
        )

    expected_hist_prefix = tof_depths.shape

    if (
        all_histograms.ndim != 4
        or all_histograms.shape[:3] != expected_hist_prefix
    ):
        raise ValueError(
            "all_histograms must have shape "
            "[blocks, height, width, bins], "
            f"got {all_histograms.shape}"
        )

    return result


def build_block_times_s(
    data: dict[str, np.ndarray],
    metadata: dict | None,
    num_blocks: int,
) -> np.ndarray:
    if "tof_block_times_s" in data:
        times = np.asarray(
            data["tof_block_times_s"],
            dtype=np.float64,
        )

        if times.shape != (num_blocks,):
            raise ValueError(
                "tof_block_times_s length does not match tof_depths."
            )

        return times

    if (
        "block_start_time_s" in data
        and "block_end_time_s" in data
    ):
        starts = np.asarray(
            data["block_start_time_s"],
            dtype=np.float64,
        )
        ends = np.asarray(
            data["block_end_time_s"],
            dtype=np.float64,
        )

        if starts.shape != (num_blocks,) or ends.shape != (num_blocks,):
            raise ValueError(
                "Saved block timing arrays do not match tof_depths."
            )

        return 0.5 * (starts + ends)

    if metadata is None:
        raise FileNotFoundError(
            "No saved block timing array was found and "
            "metadata.json is unavailable."
        )

    dt_s = float(metadata["dt_s"])

    return (
        np.arange(num_blocks, dtype=np.float64) + 0.5
    ) * dt_s


def save_ttc_results(
    output_path: Path,
    block_times_s: np.ndarray,
    ttc_results: dict[str, np.ndarray | int | float],
    window_ms: float,
    min_closing_speed_mps: float,
    max_ttc_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        block_times_s=block_times_s,
        smoothed_depths_m=ttc_results["smoothed_depths_m"],
        radial_velocity_mps=ttc_results["radial_velocity_mps"],
        closing_speed_mps=ttc_results["closing_speed_mps"],
        time_to_contact_s=ttc_results["time_to_contact_s"],
        ttc_window_ms=np.array(window_ms, dtype=np.float64),
        ttc_window_blocks=np.array(
            ttc_results["window_blocks"],
            dtype=np.int32,
        ),
        min_closing_speed_mps=np.array(
            min_closing_speed_mps,
            dtype=np.float64,
        ),
        max_ttc_s=np.array(max_ttc_s, dtype=np.float64),
    )


def load_ttc_results(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"TTC file not found: {path}")

    required_keys = {
        "block_times_s",
        "smoothed_depths_m",
        "radial_velocity_mps",
        "closing_speed_mps",
        "time_to_contact_s",
    }

    with np.load(path) as data:
        missing = required_keys.difference(data.files)

        if missing:
            raise KeyError(
                f"{path} is missing TTC arrays: {sorted(missing)}"
            )

        return {
            key: np.array(data[key], copy=True)
            for key in required_keys
        }