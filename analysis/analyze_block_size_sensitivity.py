from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_dataset(value: str) -> tuple[int, Path]:
    try:
        block_size_text, path_text = value.split(",", 1)
        block_size = int(block_size_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Dataset must have the form BLOCK_SIZE,DATASET_DIR"
        ) from exc

    if block_size <= 0:
        raise argparse.ArgumentTypeError("Block size must be positive")

    return block_size, Path(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare depth-estimation accuracy across timestamp block sizes. "
            "Reference depth is the median ideal sampled range in each "
            "pixel-block."
        )
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset,
        action="append",
        required=True,
        metavar="L,DIR",
        help=(
            "Block size and timestamp dataset directory. Repeat once per run, "
            "for example --dataset 64,outputs/test_L64."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-reference-spread-m",
        type=float,
        default=0.05,
        help=(
            "Keep only pixel-blocks whose ideal sampled ranges span no more "
            "than this value. This excludes multi-surface boundaries. "
            "Default: 0.05 m."
        ),
    )
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument(
        "--end-block",
        type=int,
        default=None,
        help="Inclusive zero-based ending block. Default: complete dataset.",
    )
    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict:
    path = dataset_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def analyze_dataset(
    requested_l: int,
    dataset_dir: Path,
    max_reference_spread_m: float,
    start_block: int,
    end_block: int | None,
) -> dict:
    metadata = load_metadata(dataset_dir)
    actual_l = int(metadata["block_size_L"])
    if actual_l != requested_l:
        raise ValueError(
            f"{dataset_dir}: requested L={requested_l}, but metadata records "
            f"L={actual_l}"
        )

    laser_rate_hz = float(metadata["laser_rate_hz"])
    rho = float(metadata["detection_probability_rho"])
    precomputed_path = dataset_dir / "timestamp_precomputed.npz"
    frames_dir = dataset_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.npz"))

    if not precomputed_path.is_file():
        raise FileNotFoundError(f"Missing precomputed data: {precomputed_path}")
    if not frame_files:
        raise RuntimeError(f"No frame files found in {frames_dir}")

    with np.load(precomputed_path) as precomputed:
        estimated_depths = np.asarray(precomputed["tof_depths"])

    total_blocks = min(len(frame_files), estimated_depths.shape[0])
    first = max(0, start_block)
    last = total_blocks - 1 if end_block is None else min(end_block, total_blocks - 1)
    if first > last:
        raise ValueError(
            f"{dataset_dir}: invalid block interval {first}-{last}; "
            f"dataset has {total_blocks} aligned blocks"
        )

    geometry_samples = 0
    valid_estimates = 0
    detected_photons = 0
    errors: list[np.ndarray] = []

    for block_index in range(first, last + 1):
        with np.load(frame_files[block_index]) as frame:
            sampled = np.asarray(frame["sampled_depths_m"], dtype=np.float64)
            noisy = np.asarray(frame["timestamps_noisy_s"])

        finite_sampled = np.isfinite(sampled)
        sample_count = finite_sampled.sum(axis=0)
        safe_sampled = np.where(finite_sampled, sampled, np.nan)

        with np.errstate(all="ignore"):
            reference = np.nanmedian(safe_sampled, axis=0)
            spread = np.nanmax(safe_sampled, axis=0) - np.nanmin(
                safe_sampled, axis=0
            )

        eligible = (
            (sample_count > 0)
            & np.isfinite(reference)
            & np.isfinite(spread)
            & (spread <= max_reference_spread_m)
        )
        estimate = np.asarray(estimated_depths[block_index], dtype=np.float64)
        paired = eligible & np.isfinite(estimate)

        geometry_samples += int(eligible.sum())
        valid_estimates += int(paired.sum())
        detected_photons += int(np.isfinite(noisy[:, eligible]).sum())
        if np.any(paired):
            errors.append((estimate[paired] - reference[paired]).reshape(-1))

    if geometry_samples == 0:
        raise RuntimeError(
            f"{dataset_dir}: no eligible single-surface pixel-blocks were found"
        )

    error = np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
    if error.size == 0:
        mae = float("nan")
        rmse = float("nan")
    else:
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))

    expected_valid_fraction = 1.0 - (1.0 - rho) ** actual_l
    return {
        "block_size_L": actual_l,
        "block_duration_us": actual_l / laser_rate_hz * 1e6,
        "expected_detections": actual_l * rho,
        "expected_valid_estimate_fraction": expected_valid_fraction,
        "measured_valid_estimate_fraction": valid_estimates / geometry_samples,
        "depth_mae_m": mae,
        "depth_rmse_m": rmse,
        "eligible_pixel_blocks": geometry_samples,
        "valid_depth_estimates": valid_estimates,
        "measured_detections_per_eligible_block": (
            detected_photons / geometry_samples
        ),
        "dataset_dir": str(dataset_dir),
    }


def save_csv(rows: list[dict], output_path: Path) -> None:
    fields = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_latex(rows: list[dict], output_path: Path) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"    \centering",
        r"    \caption{Effect of timestamp-block size on photon availability and depth-estimation accuracy.}",
        r"    \label{tab:block_size_sensitivity}",
        r"    \begin{tabular}{rrrrrr}",
        r"        \toprule",
        "        $L$ & Duration & Expected & Valid & MAE & RMSE \\\\",
        "        & ($\\mu$s) & detections & estimates & (m) & (m) \\\\",
        r"        \midrule",
    ]
    for row in rows:
        lines.append(
            "        "
            f"{row['block_size_L']} & "
            f"{row['block_duration_us']:.1f} & "
            f"{row['expected_detections']:.1f} & "
            f"{row['measured_valid_estimate_fraction']:.5f} & "
            f"{row['depth_mae_m']:.5f} & "
            f"{row['depth_rmse_m']:.5f} \\\\" 
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(rows: list[dict], output_path: Path) -> None:
    block_sizes = np.array([row["block_size_L"] for row in rows])
    mae = np.array([row["depth_mae_m"] for row in rows])
    rmse = np.array([row["depth_rmse_m"] for row in rows])
    measured_valid = np.array(
        [row["measured_valid_estimate_fraction"] for row in rows]
    )
    expected_valid = np.array(
        [row["expected_valid_estimate_fraction"] for row in rows]
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    axes[0].plot(block_sizes, mae, "o-", label="MAE")
    axes[0].plot(block_sizes, rmse, "s-", label="RMSE")
    axes[0].set_xlabel("Pulses per block, L")
    axes[0].set_ylabel("Depth error (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(block_sizes, expected_valid, "o--", label="Expected")
    axes[1].plot(block_sizes, measured_valid, "s-", label="Measured")
    axes[1].set_xlabel("Pulses per block, L")
    axes[1].set_ylabel("Valid-estimate fraction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_reference_spread_m < 0:
        raise ValueError("--max-reference-spread-m must be nonnegative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        analyze_dataset(
            requested_l=block_size,
            dataset_dir=dataset_dir,
            max_reference_spread_m=args.max_reference_spread_m,
            start_block=args.start_block,
            end_block=args.end_block,
        )
        for block_size, dataset_dir in args.dataset
    ]
    rows.sort(key=lambda row: row["block_size_L"])

    save_csv(rows, args.output_dir / "block_size_sensitivity.csv")
    save_latex(rows, args.output_dir / "block_size_sensitivity.tex")
    save_figure(rows, args.output_dir / "block_size_sensitivity.png")

    print("Block-size sensitivity results")
    print(f"Single-surface spread threshold: {args.max_reference_spread_m:.6f} m")
    for row in rows:
        print(
            f"L={row['block_size_L']}: "
            f"valid={row['measured_valid_estimate_fraction']:.6f}, "
            f"MAE={row['depth_mae_m']:.6f} m, "
            f"RMSE={row['depth_rmse_m']:.6f} m"
        )
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()