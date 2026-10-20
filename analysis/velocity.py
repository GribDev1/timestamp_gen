import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--pixel-y", type=int, default=4)
    parser.add_argument("--pixel-x", type=int, default=4)
    args = parser.parse_args()

    with np.load(args.input_file) as data:
        depths = data["tof_depths"]
        times = data["tof_block_times_s"]

    y, x = args.pixel_y, args.pixel_x

    if depths.ndim != 3:
        raise ValueError("Expected depths shaped [blocks, height, width].")

    if not (0 <= y < depths.shape[1] and 0 <= x < depths.shape[2]):
        raise ValueError("Selected pixel is outside the sensor grid.")

    if len(times) != len(depths) or len(times) < 2:
        raise ValueError("Need matching depth/time arrays with at least two blocks.")

    dt = np.diff(times)

    if not np.all(np.isfinite(times)) or np.any(dt <= 0):
        raise ValueError("Times must be finite and strictly increasing.")

    pixel_depths = depths[:, y, x]
    
    # Negative radial velocity means the surface is approaching.
    radial_velocity = np.diff(pixel_depths) / dt
    closing_speed = -radial_velocity
    
    # Velocity belongs to an interval between two depth measurements.
    velocity_times = 0.5 * (times[1:] + times[:-1])
    
    print(f"Pixel: (y={y}, x={x})")
    print("Time (s)     Depth change (m)    Radial velocity (m/s)")
    
    for i in range(min(10, len(radial_velocity))):
        print(
            f"{velocity_times[i]:.6f}   "
            f"{pixel_depths[i+1] - pixel_depths[i]:+.6f}    "
            f"{radial_velocity[i]:+.3f}"
        )
        
    valid = np.isfinite(radial_velocity)
    print(f"\nValid velocity estimates: {valid.sum()}/{len(valid)}")
    
    
if __name__ == "__main__":
    main()