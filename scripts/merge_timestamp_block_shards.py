from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge block-range timestamp precomputed shards."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-blocks",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    shard_dir = args.output_dir / "precomputed_shards"

    shard_paths = sorted(
        shard_dir.glob("blocks_*.npz")
    )

    if not shard_paths:
        raise RuntimeError(
            f"No precomputed shards found in {shard_dir}"
        )

    records = []

    reference_tau = None
    reference_depth = None

    for path in shard_paths:
        with np.load(path) as data:
            start_block = int(data["start_block"])
            end_block = int(data["end_block"])

            tof_depths = data["tof_depths"].copy()
            all_I = data["all_I"].copy()
            all_histograms = (
                data["all_histograms"].copy()
            )
            block_times = (
                data["tof_block_times_s"].copy()
            )

            tau_centers = (
                data["hist_bin_centers_tau"].copy()
            )
            depth_centers = (
                data[
                    "hist_bin_centers_depth_m"
                ].copy()
            )

        expected_count = end_block - start_block

        if tof_depths.shape[0] != expected_count:
            raise RuntimeError(
                f"{path.name}: expected "
                f"{expected_count} rows but found "
                f"{tof_depths.shape[0]}"
            )

        if all_I.shape[0] != expected_count:
            raise RuntimeError(
                f"{path.name}: all_I length mismatch"
            )

        if all_histograms.shape[0] != expected_count:
            raise RuntimeError(
                f"{path.name}: histogram length mismatch"
            )

        if block_times.shape[0] != expected_count:
            raise RuntimeError(
                f"{path.name}: time length mismatch"
            )

        if reference_tau is None:
            reference_tau = tau_centers
            reference_depth = depth_centers
        else:
            if not np.array_equal(
                reference_tau,
                tau_centers,
            ):
                raise RuntimeError(
                    f"{path.name}: timestamp bins differ"
                )

            if not np.array_equal(
                reference_depth,
                depth_centers,
            ):
                raise RuntimeError(
                    f"{path.name}: depth bins differ"
                )

        records.append(
            (
                start_block,
                end_block,
                tof_depths,
                all_I,
                all_histograms,
                block_times,
                path,
            )
        )

    records.sort(key=lambda item: item[0])

    expected_start = 0

    for start_block, end_block, *_ in records:
        if start_block != expected_start:
            raise RuntimeError(
                "Missing or overlapping block range: "
                f"expected block {expected_start}, "
                f"found shard beginning at {start_block}"
            )

        expected_start = end_block

    if (
        args.expected_blocks is not None
        and expected_start != args.expected_blocks
    ):
        raise RuntimeError(
            f"Merged range ends at {expected_start}, "
            f"but expected {args.expected_blocks} blocks"
        )

    tof_depths = np.concatenate(
        [record[2] for record in records],
        axis=0,
    )
    all_I = np.concatenate(
        [record[3] for record in records],
        axis=0,
    )
    all_histograms = np.concatenate(
        [record[4] for record in records],
        axis=0,
    )
    tof_block_times_s = np.concatenate(
        [record[5] for record in records],
        axis=0,
    )

    output_path = (
        args.output_dir
        / "timestamp_precomputed.npz"
    )

    temporary_path = (
        args.output_dir
        / "timestamp_precomputed.tmp.npz"
    )

    np.savez(
        temporary_path,
        tof_depths=tof_depths,
        all_I=all_I,
        all_histograms=all_histograms,
        tof_block_times_s=tof_block_times_s,
        hist_bin_centers_tau=reference_tau,
        hist_bin_centers_depth_m=reference_depth,
    )

    temporary_path.replace(output_path)

    print(f"Merged {len(records)} shards.")
    print(f"Total blocks: {tof_depths.shape[0]}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()