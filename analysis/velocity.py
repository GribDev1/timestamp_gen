"""Simple radial-velocity estimate from timestamp_precomputed.npz.

Idea: one block is only 25.6 us long, so the range barely changes between
neighbouring blocks and the noise swamps it. Instead, take a WINDOW of many
blocks, fit a straight line to range-vs-time, and use the line's slope as the
velocity. Slide the window along the recording and repeat for every pixel.

Negative velocity = approaching the sensor, positive = moving away.

Usage: python velocity.py path/to/timestamp_precomputed.npz
"""
import sys
import numpy as np

# Settings
WINDOW = 500        # blocks per fit (500 blocks * 25.6 us = 12.8 ms)
STEP = 20           # move the window forward this many blocks each time
MAX_SPEED = 20.0    # m/s; faster fits are treated as errors and dropped
MIN_POINTS = 250    # need at least this many valid samples in a window

# 1) Load the data
with np.load(sys.argv[1]) as data:
    ranges = data["tof_depths"].astype(np.float64) # [blocks, 8, 8], NaN = no
    times = data["tof_block_times_s"].astype(np.float64) # [blocks], seconds

num_blocks, H, W = ranges.shape
print(f"{num_blocks} blocks, {H}x{W} pixels, {times[-1] - times[0]:.3f} s of data")

# 2) Slide a window along time and fit a line for each pixel
window_starts = list(range(0, num_blocks - WINDOW + 1, STEP))
velocity = np.full((len(window_starts), H, W), np.nan) # result, NaN if not usable
window_times = np.zeros(len(window_starts))

for w, start in enumerate(window_starts):
    t = times[start:start + WINDOW]
    window_times[w] = t.mean()
    
    for y in range(H):
        for x in range(W):
            r = ranges[start:start + WINDOW, y, x]
            ok = np.isfinite(r) # ignore blocks with no readings
            if ok.sum() < MIN_POINTS:
                continue # too few readings to trust a fit
            
            # polyfit degree 1 -> [slope, intercept]; slope = meters per second
            slope = np.polyfit(t[ok] - t[ok].mean(), r[ok], 1)[0]
            
            if abs(slope) <= MAX_SPEED: # drop impossible speeds
                velocity[w, y, x] = slope
                

# 3) Report
print(f"Window = {WINDOW} blocks ({WINDOW * 25.6e-3:.1f} ms), "
      f"{np.isfinite(velocity).mean() * 100:.0f}% of pixel-windows usable")

np.set_printoptions(precision=1, suppress=True, linewidth=120)
print("\nMedian velocity per pixel (m/s):")
print(np.nanmedian(velocity, axis=0))

y, x = H // 2 - 1, W // 2 # a pixel near the center
print(f"\nPixel ({y}, {x}) over time:")
print("  t (ms)   v (m/s)")
for w in np.linspace(0, len(window_starts) - 1, 10).astype(int):
        print(f"   {window_times[w] * 1e3:6.1f}   {velocity[w, x, y]:+7.2f}") 