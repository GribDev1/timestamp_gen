import numpy as np

with np.load("outputs/drone_flyby/timestamp_precomputed.npz") as data:
    ranges = data["tof_depths"].astype(np.float64)
    times = data["tof_block_times_s"].astype(np.float64)

dt = np.diff(times)

if np.any(dt <= 0):
    raise ValueError("Block times must be strictly increasing.")

velocity = np.diff(ranges, axis=0) / dt[:, None, None]
velocity_times = (times[:-1] + times[1:]) / 2