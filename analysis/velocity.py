"""Simple radial-velocity estimate from timestamp_precomputed.npz.

Idea: one block is only 25.6 us long, so the range barely changes between
neighbouring blocks and the noise swamps it. Instead, take a WINDOW of many
blocks, fit a straight line to range-vs-time, and use the line's slope as the
velocity. Slide the window along the recording and repeat for every pixel.

Negative velocity = approaching the sensor, positive = moving away.

Usage: python analysis/velocity.py <test_name>
"""
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg") # draw off-screen

import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Settings
RENDER_FPS = 240.0 # Blender frame rate
FIRST_FRAME = 1 # Blender first frame
OUTPUTS_DIR = Path("outputs") # folder that holds one sub-folder per test
WINDOW = 500 # blocks per fit (500 blocks * 25.6 us = 12.8 ms)
STEP = 250 # move the window forward this many blocks each time
MAX_SPEED = 20.0 # m/s; faster fits are treated as errors and dropped
MIN_VALID_FRAC = 0.5 # need at least this fraction of valid samples in a window
COLOR_LIMIT = 15.0 # vel color scale from -15 to +15 m/s
GIF_FPS = 15 # playback speed of GIF

# 1) Load the data
if len(sys.argv) < 2:
    sys.exit("Usage: python analysis/velocity.py <test_name>   (e.g. drone_flyby)")

test_name = sys.argv[1]
npz_path = OUTPUTS_DIR / test_name / "timestamp_precomputed.npz"
if not npz_path.exists():
    available = sorted(p.name for p in OUTPUTS_DIR.glob("*/") if (p / "timestamp_precomputed.npz").exists())
    sys.exit(f"Not found: {npz_path}\nTests with a timestamp_precomputed.npz: {available}")

with np.load(npz_path) as data:
    ranges = data["tof_depths"].astype(np.float64) # [blocks, 8, 8], NaN = no
    times = data["tof_block_times_s"].astype(np.float64) # [blocks], seconds

num_blocks, H, W = ranges.shape
print(f"{num_blocks} blocks, {H}x{W} pixels, {times[-1] - times[0]:.3f} s of data")

# 2) Slide a window along time and fit a line for each pixel
window_starts = list(range(0, num_blocks - WINDOW + 1, STEP))
velocity = np.full((len(window_starts), H, W), np.nan) # result, NaN if not usable
mean_range = np.full((len(window_starts), H, W), np.nan)
window_times = np.zeros(len(window_starts))

print(f"Computing {len(window_starts)} windows...")
for w, start in enumerate(window_starts):
    t = times[start:start + WINDOW]
    window_times[w] = t.mean()
    
    for y in range(H):
        for x in range(W):
            r = ranges[start:start + WINDOW, y, x]
            ok = np.isfinite(r) # ignore blocks with no readings
            if ok.sum() < MIN_VALID_FRAC * WINDOW:
                continue # too few readings to trust a fit
            
            mean_range[w, y, x] = r[ok].mean()
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
print("  t (ms)   frame   v (m/s)")

block_duration = np.median(np.diff(times))
for w in np.linspace(0, len(window_starts) - 1, 10).astype(int):
    scene_time = window_times[w] - block_duration
    frame = scene_time * RENDER_FPS + FIRST_FRAME
    
    print(f"   {window_times[w] * 1e3:6.1f}   {frame:6.1f}   {velocity[w, y, x]:+7.2f}") 
    
# 4) Draw one image per window
vel_cmap = plt.get_cmap("coolwarm").copy()
vel_cmap.set_bad("lightgrey")
rng_cmap = plt.get_cmap("viridis_r").copy()
rng_cmap.set_bad("lightgrey")
range_max = np.nanmax(mean_range)

fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(9, 4.6))
im_r = ax_r.imshow(np.ma.masked_invalid(mean_range[0]), cmap=rng_cmap, vmin=0, vmax=range_max)
im_v = ax_v.imshow(np.ma.masked_invalid(velocity[0]), cmap=vel_cmap, vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT)
ax_r.set_title("Mean range (m)")
ax_v.set_title("Velocity (m/s) blue = approaching")
fig.colorbar(im_r, ax=ax_r, fraction=0.046)
fig.colorbar(im_v, ax=ax_v, fraction=0.046)
for ax in (ax_r, ax_v):
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
text_r = [[ax_r.text(x, y, "", ha="center", va="center", fontsize=6) for x in range(W)] for y in range(H)]
text_v = [[ax_v.text(x, y, "", ha="center", va="center", fontsize=6) for x in range(W)] for y in range(H)]
suptitle = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.93))

images = []
for w in range(len(window_starts)):
    im_r.set_data(np.ma.masked_invalid(mean_range[w]))
    im_v.set_data(np.ma.masked_invalid(velocity[w]))
    for y in range(H):
        for x in range(W):
            text_r[y][x].set_text("" if np.isnan(mean_range[w, y, x]) else f"{mean_range[w, y, x]:.1f}")
            text_v[y][x].set_text("" if np.isnan(velocity[w, y, x]) else f"{velocity[w, y, x]:.1f}")
    frame = (window_times[w] - block_duration) * RENDER_FPS + FIRST_FRAME
    suptitle.set_text(f"{test_name}   t = {window_times[w] * 1e3:.0f} ms   Blender frame {frame:.1f}")
    fig.canvas.draw()
    images.append(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    
# 5) Save the GIF
out_path = npz_path.parent / "plots" / "velocity.gif"
out_path.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(out_path, images, duration=1000 / GIF_FPS, loop=0)
print(f"Saved {len(images)} frames to {out_path}")