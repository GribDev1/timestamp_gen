"""
MWE for reading timestamp_gen output.

Run:
    python read_timestamp_dataset.py outputs/examples/flat_moving_ex
"""

from pathlib import Path
import json
import sys

import numpy as np


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python read_timestamp_dataset.py <dataset_dir>"
        )
        
    dataset_dir = Path(sys.argv[1])
    
    metadata_path = dataset_dir / "metadata.json"
    precomputed_path = dataset_dir / "timestamp_precomputed.npz"
    frames_dir = dataset_dir / "frames"
            
    # Read metadata
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    tof_h = metadata["tof_h"]
    tof_w = metadata["tof_w"]
    L = metadata["block_size_L"]
    laser_rate_hz = metadata["laser_rate_hz"]
    rho = metadata["detection_probability_rho"]

    block_duration_s = L / laser_rate_hz
    expected_detections = L * rho

    print()
    print("Derived:")
    print(f"ToF grid: {tof_h} x {tof_w}")
    print(f"Block duration: {block_duration_s * 1e6:.2f} us")
    print(f"Expected detections / pixel / block: {expected_detections:.2f}")
    
    # Read precomputed histograms/depths
    pre = np.load(precomputed_path)

    tof_depths = pre["tof_depths"]
    all_I = pre["all_I"]
    all_histograms = pre["all_histograms"]
    hist_bin_centers_depth_m = pre["hist_bin_centers_depth_m"]

    print()
    print("Precomputed arrays:")
    print(f"tof_depths shape: {tof_depths.shape}")
    print(f"all_I shape: {all_I.shape}")
    print(f"all_histograms shape: {all_histograms.shape}")
    print(f"hist_bin_centers_depth_m shape: {hist_bin_centers_depth_m.shape}")
    
    # Example interpretation:
    # tof_depths[t, y, x] is the histogram-derived depth estimate
    # at timestamp block t and ToF pixel (y, x).
    center_y = tof_depths.shape[1] // 2
    center_x = tof_depths.shape[2] // 2

    center_depth_trace = tof_depths[:, center_y, center_x]
    center_valid_trace = all_I[:, center_y, center_x]

    print()
    print("Center pixel summary:")
    print(f"Center pixel: y={center_y}, x={center_x}")
    print(f"Mean depth: {np.nanmean(center_depth_trace):.4f} m")
    print(f"Depth min/max: {np.nanmin(center_depth_trace):.4f} / {np.nanmax(center_depth_trace):.4f} m")
    print(f"Mean valid detection fraction: {np.nanmean(center_valid_trace):.4f}")
    
    # Read one raw timestamp frame
    frame_files = sorted(frames_dir.glob("frame_*.npz"))
    if not frame_files:
        raise RuntimeError(f"No frame files found in {frames_dir}")

    first_frame = np.load(frame_files[0])

    sampled_depths_m = first_frame["sampled_depths_m"]
    timestamps_clean_s = first_frame["timestamps_clean_s"]
    timestamps_noisy_s = first_frame["timestamps_noisy_s"]
    detection_mask = first_frame["detection_mask"]
    
    raw_center_timestamps = timestamps_noisy_s[:, center_y, center_x]
    valid_ts = raw_center_timestamps[np.isfinite(raw_center_timestamps)]

    print()
    print("=== Center pixel raw timestamps, first block ===")
    print(f"File: {frame_files[0].name}")
    print(f"Detected photons: {valid_ts.size} / {raw_center_timestamps.size}")

    if valid_ts.size > 0:
        c_light = metadata["c_light"]
        depths_from_ts = 0.5 * c_light * valid_ts
        print(f"Mean raw timestamp depth: {np.mean(depths_from_ts):.4f} m")
        print(f"Raw timestamp depth min/max: {np.min(depths_from_ts):.4f} / {np.max(depths_from_ts):.4f} m")


if __name__ == "__main__":
    main()