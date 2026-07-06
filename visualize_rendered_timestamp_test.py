import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from timestamp_gen import (
    simulate_timestamp_block,
    timestamp_to_depth,
    TOF_H,
    TOF_W,
)

TEST_NAME = "occlusion"
DISPLAY_NAME = TEST_NAME.replace("_", " ") + " Plane"

render_path = f"tests/rendered_tests/test_{TEST_NAME}"
RENDER_DIR = Path(render_path)
DEPTH_DIR = RENDER_DIR / "depths"
NORMAL_DIR = RENDER_DIR / "normals"
OUTPUT_DIR = Path("tests/test_visualizations/") / TEST_NAME
OUTPUT_DIR.mkdir(exist_ok=True)

HIST_DIR = OUTPUT_DIR / "histograms"
CLEAN_HIST_DIR = HIST_DIR / "clean"
NOISY_HIST_DIR = HIST_DIR / "noisy"

CLEAN_HIST_DIR.mkdir(parents=True, exist_ok=True)
NOISY_HIST_DIR.mkdir(parents=True, exist_ok=True)


def load_depth_file(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Could not load depth file: {path}")

    depth = depth.astype(np.float32)

    # If EXR has multiple channels, inspect each one before choosing channel 0
    if depth.ndim == 3:
        for ch in range(depth.shape[2]):
            channel = depth[:, :, ch]
            finite = channel[np.isfinite(channel)]

        depth = depth[:, :, 0]

    MAX_VALID_DEPTH_M = 20.0

    depth = np.where(
        np.isfinite(depth) & (depth > 0.01) & (depth < MAX_VALID_DEPTH_M),
        depth,
        np.nan
    )

    return depth


def load_normal_file(path: Path) -> np.ndarray:
    normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if normal is None:
        raise RuntimeError(f"Could not load normal file: {path}")

    normal = normal.astype(np.float32)

    if normal.ndim == 3 and normal.shape[2] >= 3:
        normal = normal[:, :, :3]
        normal = normal[:, :, ::-1]  # BGR -> RGB

    return normal


def clean_mean_depth_from_block(block):
    clean_depths = timestamp_to_depth(block.timestamps_clean_s)
    return np.nanmean(clean_depths, axis=0)


def noisy_mean_depth_from_block(block):
    noisy_depths = timestamp_to_depth(block.timestamps_noisy_s)
    return np.nanmean(noisy_depths, axis=0)


def visualize_frame(frame_i, depth_file, normal_file):
    depth = load_depth_file(depth_file)
    normals = load_normal_file(normal_file)

    block = simulate_timestamp_block(
        depth=depth,
        normals=normals,
        frame_number=frame_i + 1,
        source_depth_file=str(depth_file),
        source_normal_file=str(normal_file),
    )

    clean_mean_depth = clean_mean_depth_from_block(block)
    noisy_mean_depth = noisy_mean_depth_from_block(block)

    valid_counts = np.isfinite(block.timestamps_noisy_s).sum(axis=0)
    
    depth_error_mm = 1e3 * (noisy_mean_depth - clean_mean_depth)
    
    # Noise-only visualization in micrometers
    err_lim = np.nanpercentile(np.abs(depth_error_mm), 99)
    if not np.isfinite(err_lim) or err_lim == 0:
        err_lim = 1.0

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    im0 = axes[0].imshow(depth)
    axes[0].set_title("Rendered depth EXR")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(clean_mean_depth, vmin=np.nanmin(clean_mean_depth), vmax=np.nanmax(clean_mean_depth))
    axes[1].set_title("Clean timestamp depth")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(noisy_mean_depth, vmin=np.nanmin(clean_mean_depth), vmax=np.nanmax(clean_mean_depth))
    axes[2].set_title("Noisy timestamp depth")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(valid_counts)
    axes[3].set_title("Detected photons/pixel")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
        
    im4 = axes[4].imshow(depth_error_mm, vmin=-err_lim, vmax=err_lim)
    axes[4].set_title("Noise Error (mm)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{TEST_NAME}_frame_{frame_i + 1:03d}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")
    
    # Plot a few representative per-pixel histograms
    hist_pixels = [
        (TOF_W // 2, TOF_H // 2), # center
        (0, TOF_H // 2), # left edge
        (TOF_W - 1, TOF_H // 2), # right edge
        (TOF_W // 2, 0), # top edge
        (TOF_W // 2, TOF_H - 1), # bottom edge
        (0, 1), # top left corner
        (TOF_W - 1, 0), # top right corner
        (0, TOF_H - 1), # bottom left corner
        (TOF_W - 1, TOF_H - 1), # bottom right corner
    ]
    
    num_clean_hists = 0
    num_noisy_hists = 0
    
    for px, py in hist_pixels:
        clean_saved = plot_pixel_histogram(
            block, px, py, frame_i, (3.0, 12.0), bins=60, model="clean"
        )
        noisy_saved = plot_pixel_histogram(
            block, px, py, frame_i, (3.0, 12.0), bins=256, model="noisy"
        )
        if clean_saved:
            num_clean_hists += 1
        if noisy_saved:
            num_noisy_hists += 1
        
    print(f"Saved {num_clean_hists} clean histograms")
    print(f"Saved {num_noisy_hists} noisy histograms")


def plot_frame_mean_depths(depth_files, normal_files):
    frame_numbers = []
    clean_means = []
    noisy_means = []

    for frame_i, (depth_file, normal_file) in enumerate(zip(depth_files, normal_files)):
        depth = load_depth_file(depth_file)
        normals = load_normal_file(normal_file)

        block = simulate_timestamp_block(
            depth=depth,
            normals=normals,
            frame_number=frame_i + 1,
            source_depth_file=str(depth_file),
            source_normal_file=str(normal_file),
        )

        clean_mean = np.nanmean(clean_mean_depth_from_block(block))
        noisy_mean = np.nanmean(noisy_mean_depth_from_block(block))

        frame_numbers.append(frame_i + 1)
        clean_means.append(clean_mean)
        noisy_means.append(noisy_mean)
        
    clean_means_um = 1e6 * np.array(clean_means)
    noisy_means_um = 1e6 * np.array(noisy_means)

    plt.figure(figsize=(8, 5))
    plt.plot(frame_numbers, clean_means_um, marker="o", label="Clean timestamp depth")
    plt.plot(frame_numbers, noisy_means_um, marker="o", label="Noisy timestamp depth")
    #plt.ylim(4.9998, 5.0002)
    plt.xlabel("Frame")
    plt.ylabel("Mean depth (um)")
    plt.title(f"{DISPLAY_NAME.title()}: Mean Depth over Time")
    plt.legend()
    plt.grid(True)

    output_path = OUTPUT_DIR / f"{TEST_NAME}_mean_depth_over_time.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")
    
    errors_um = 1000000.0 * (np.array(noisy_means) - np.array(clean_means))

    plt.figure(figsize=(8, 5))
    plt.plot(frame_numbers, errors_um, marker="o")
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Frame")
    plt.ylabel("Noise error mean depth (um)")
    plt.title(f"{DISPLAY_NAME.title()}: Noise Error over Time")
    plt.grid(True)

    output_path = OUTPUT_DIR / f"{TEST_NAME}_mean_depth_error_um.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")
    

def plot_pixel_histogram(block, pixel_x, pixel_y, frame_i, hist_range, bins=40, model="noisy"):
    """
    Plots the depth histogram for one ToF pixel.
    
    pixel_x: ToF x index, 0 to TOF_W - 1
    pixel_y: ToF y index, 0 to TOF_H - 1
    model: "clean" or "noisy"
    """
    if model == "clean":
        timestamps = block.timestamps_clean_s[:, pixel_y, pixel_x]
        output_dir = CLEAN_HIST_DIR
    elif model == "noisy":
        timestamps = block.timestamps_noisy_s[:, pixel_y, pixel_x]
        output_dir = NOISY_HIST_DIR
    else:
        raise ValueError("model must be 'clean' or 'noisy'")
    
    depths = timestamp_to_depth(timestamps)
    depths = depths[np.isfinite(depths)]
    
    if depths.size == 0:
        print(f"No valid {model} photons for pixel x={pixel_x}, y={pixel_y}")
        return False
    
    plt.figure(figsize=(7, 4))
    plt.hist(depths, bins=bins, range=hist_range)
    
    plt.xlim(hist_range)
    plt.xlabel("Depth / range (m)")
    plt.ylabel("Photon count")
    plt.title(
        f"{DISPLAY_NAME}: Frame {frame_i + 1}, {model.title()} Histogram\n"
        f"Pixel x={pixel_x}, y={pixel_y}"
    )
    plt.grid(True)
    
    output_path = output_dir / (
        f"{TEST_NAME}_frame_{frame_i + 1:03d}_"
        f"hist_{model}_x{pixel_x:02d}_y{pixel_y:02d}.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    return True
    
    
def compute_histogram_depth_range(depth_files, pad_fraction=0.05):
    """
    Computes one stable x-axis range for histogram plots across this tests.
    
    Uses the rendered depth maps to find the global valid depth range,
    then expands it slightly with padding.
    """
    global_min = np.inf
    global_max = -np.inf
    
    for depth_file in depth_files:
        depth = load_depth_file(depth_file)
        valid = depth[np.isfinite(depth)]
        
        if valid.size == 0:
            continue
        
        global_min = min(global_min, np.nanmin(valid))
        global_max = max(global_max, np.nanmax(valid))
        
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        return (0.1, 20.0)
    
    span = global_max - global_min
    
    if span <= 0:
        pad = 0.05 * max(global_min, 1.0)
    else:
        pad = pad_fraction * span
        
    return (global_min - pad, global_max + pad)


def main():
    np.random.seed(0)

    depth_files = sorted(DEPTH_DIR.glob("*.exr"))
    normal_files = sorted(NORMAL_DIR.glob("*.exr"))

    if len(depth_files) == 0:
        raise RuntimeError(f"No depth EXR files found in {DEPTH_DIR}")

    if len(depth_files) != len(normal_files):
        raise RuntimeError("Depth/normal file count mismatch.")

    selected_frames = [
        0,
        len(depth_files) // 2,
        len(depth_files) - 1,
    ]

    for frame_i in selected_frames:
        visualize_frame(
            frame_i, 
            depth_files[frame_i], 
            normal_files[frame_i],
        )

    plot_frame_mean_depths(depth_files, normal_files)


if __name__ == "__main__":
    main()