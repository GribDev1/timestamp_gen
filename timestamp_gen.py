"""
    Timestamp Generation:

        This script converts rendered VisionSIM/Blender depth and normal maps into a
        precomputed timestamp dataset for the token-based ToF pipeline.

        For each rendered frame, the script samples high-resolution depth values inside
        each lower-resolution ToF pixel footprint. This preserves sub-pixel depth
        mixtures at object boundaries and occlusions. The sampled depths are converted
        to clean photon timestamps, then a Bernoulli detection model and Gaussian timing
        jitter are applied to create noisy single-photon timestamp measurements.

        The script also computes the block-rate data needed by token_process.py:
            - per-frame mini-histograms
            - block-rate depth estimates
            - valid detection fraction I
            - in-window stream S1
            - out-of-window stream S2+
            - histogram bin centers

    Running file:
        python timestamp_gen.py

    Running for performance testing:
        python -m cProfile -o timestamp_profile.prof timestamp_gen.py
        python -m pstats timestamp_profile.prof
        sort cumulative
        stats 40
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from tof_sensor import ToFSensor
from timestamp_dataset import TimestampBlock, TimestampDataset, TimestampMetadata

# =========================
# User settings
# =========================

RENDER_DIR = Path("full_render_320x160")
DEPTH_DIR = RENDER_DIR / "depths"
NORMAL_DIR = RENDER_DIR / "normals"

OUTPUT_DIR = Path("timestamp_output")

SENSOR = ToFSensor(
    name="generic_spad_sensor",
    tof_h=32,
    tof_w=64,
    laser_rate_hz=10e6,
    block_size_L=256,
    detection_probability_rho=0.05,
    timing_jitter_std_s=50e-12,
    min_valid_depth_m=0.01,
    max_valid_depth_m=20.0,
)

USE_INTERPOLATED_VISIBILITY_SWITCH = True

NUM_INTERPOLATION_STEPS = 4

RENDER_FPS = 240.0
RENDER_DT = 1.0 / RENDER_FPS

EFFECTIVE_FPS = (
    RENDER_FPS * NUM_INTERPOLATION_STEPS
    if USE_INTERPOLATED_VISIBILITY_SWITCH
    else RENDER_FPS
)
EFFECTIVE_DT = 1.0 / EFFECTIVE_FPS

USE_WEIGHTED_DEPTH_SAMPLING = True

RANDOM_SEED = 0

HIST_DEPTH_MIN_M = 0.1
HIST_DEPTH_MAX_M = 20.0
HIST_NUM_BINS = 256

USE_ADAPTIVE_TIME_GATING = True
ADAPTIVE_GATE_M = 1.0

DELTA_WINDOW = 0.15

MAX_SAME_SURFACE_SPEED_M_PER_S = 10.0
SWITCH_DIST_THRESH_M = MAX_SAME_SURFACE_SPEED_M_PER_S * RENDER_DT

NORMAL_COSINE_THRESH = 0.9

SAVE_FULL_TIMESTAMP_DATASET = True
SAVE_PRECOMPUTED_DATA = True


# =========================
# File loading
# =========================

def load_depth_file(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Could not load depth file: {path}")

    depth = depth.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[:, :, 0]
        
    depth = np.where(
        np.isfinite(depth)
        & (depth > SENSOR.min_valid_depth_m)
        & (depth < SENSOR.max_valid_depth_m),
        depth,
        np.nan
    )

    return depth


def load_normal_file(path: Path) -> np.ndarray:
    normal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if normal is None:
        raise RuntimeError(f"Could not load normal file: {path}")

    normal = normal.astype(np.float32)

    if normal.ndim != 3 or normal.shape[2] < 3:
        raise RuntimeError(f"Normal file does not look like a 3-channel normal map: {path}")

    normal = normal[:, :, :3]

    # OpenCV loads EXR channels as BGR, convert to RGB
    normal = normal[:, :, ::-1]

    return normal


def find_depth_files(depth_dir: Path):
    files = sorted(depth_dir.glob("*.exr"))

    if not files:
        raise RuntimeError(f"No EXR depth files found in {depth_dir}")

    return files


def find_normal_files(normal_dir: Path):
    files = sorted(normal_dir.glob("*.exr"))

    if not files:
        raise RuntimeError(f"No EXR normal files found in {normal_dir}")

    return files


# =========================
# Interpolation
# =========================


def normalize_vectors(v, eps=1e-12):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)
       
        
def simulate_timestamp_block_interpolated_pair(
    depth1: np.ndarray,
    depth2: np.ndarray,
    normals1: np.ndarray | None,
    normals2: np.ndarray | None,
    alpha: float,
    frame_number: int,
    source_depth_file: str,
    source_normal_file: str,
    ray_dirs_full: np.ndarray,
    tof_h: int=SENSOR.tof_h,
    tof_w: int=SENSOR.tof_w,
    L: int=SENSOR.block_size_L,
    rho: float=SENSOR.detection_probability_rho,
    jitter_std: float=SENSOR.timing_jitter_std_s,
    switch_dist_thresh_m: float=SWITCH_DIST_THRESH_M,
    normal_cosine_thresh: float=NORMAL_COSINE_THRESH,
) -> TimestampBlock:
    """
    Generate one timestamp block between two rendered frames.
    
    Same-surface rays:
        interpolate depth and normal.
        
    Visibility-change rays:
        hard switch at alpha = 0.5.
    """
    h, w = depth1.shape
    cell_h = h // tof_h
    cell_w = w // tof_w
    
    sampled_depths = np.full((L, tof_h, tof_w), np.nan, dtype=np.float32)
    timestamps_clean = np.full((L, tof_h, tof_w), np.nan, dtype=np.float32)
    timestamps_noisy = np.full((L, tof_h, tof_w), np.nan, dtype=np.float32)
    
    detection_mask_all = np.random.rand(L, tof_h, tof_w) < rho
    jitter_all = np.random.randn(L, tof_h, tof_w).astype(np.float32) * jitter_std
    
    before_mid = alpha < 0.5
    
    for y in range(tof_h):
        y0 = y * cell_h
        y1 = (y + 1) * cell_h if y < tof_h - 1 else h

        for x in range(tof_w):
            x0 = x * cell_w
            x1 = (x + 1) * cell_w if x < tof_w - 1 else w

            z1 = depth1[y0:y1, x0:x1].reshape(-1)
            z2 = depth2[y0:y1, x0:x1].reshape(-1)

            valid1 = (
                np.isfinite(z1)
                & (z1 > SENSOR.min_valid_depth_m)
                & (z1 < SENSOR.max_valid_depth_m)
            )
            valid2 = (
                np.isfinite(z2)
                & (z2 > SENSOR.min_valid_depth_m)
                & (z2 < SENSOR.max_valid_depth_m)
            )

            ray_dirs = ray_dirs_full[y0:y1, x0:x1, :].reshape(-1, 3)
            ray_z = np.maximum(ray_dirs[:, 2], 1e-8)

            dist1 = z1 / ray_z
            dist2 = z2 / ray_z

            same_surface = (
                valid1
                & valid2
                & (np.abs(dist2 - dist1) <= switch_dist_thresh_m)
            )

            n = None

            if normals1 is not None and normals2 is not None:
                n1 = normals1[y0:y1, x0:x1, :].reshape(-1, 3)
                n2 = normals2[y0:y1, x0:x1, :].reshape(-1, 3)

                n1_unit = normalize_vectors(n1)
                n2_unit = normalize_vectors(n2)

                normal_cos = np.sum(n1_unit * n2_unit, axis=-1)

                same_surface = same_surface & (normal_cos >= normal_cosine_thresh)

                n_interp = (1.0 - alpha) * n1 + alpha * n2
                n_interp = normalize_vectors(n_interp)

                n_switch = n1 if before_mid else n2
                n_switch = normalize_vectors(n_switch)

                n = np.where(same_surface[:, None], n_interp, n_switch)

            z_interp = (1.0 - alpha) * z1 + alpha * z2
            z_switch = z1 if before_mid else z2

            z = np.where(same_surface, z_interp, z_switch)

            hit_interp = valid1 & valid2
            hit_switch = valid1 if before_mid else valid2
            hit = np.where(same_surface, hit_interp, hit_switch)

            dist = z / ray_z

            valid = (
                hit
                & np.isfinite(z)
                & np.isfinite(dist)
                & (dist > SENSOR.min_valid_depth_m)
                & (dist < SENSOR.max_valid_depth_m)
            )

            if not np.any(valid):
                continue

            valid_ranges = dist[valid].astype(np.float32)

            weights = None

            if n is not None and USE_WEIGHTED_DEPTH_SAMPLING:
                valid_normals = normalize_vectors(n[valid])
                valid_ray_dirs = ray_dirs[valid]
                
                cos_incidence = np.sum((-valid_ray_dirs) * valid_normals, axis=-1)
                cos_incidence = np.maximum(cos_incidence, 0.0)

                distance_falloff = 1.0 / np.maximum(valid_ranges ** 2, 1e-6)

                weights = cos_incidence * distance_falloff
                weights = weights.astype(np.float64)

                if np.sum(weights) <= 0 or not np.all(np.isfinite(weights)):
                    weights = np.ones_like(valid_ranges, dtype=np.float64)

                weights = weights / np.sum(weights)

            if weights is not None:
                sampled_range = np.random.choice(
                    valid_ranges,
                    size=L,
                    replace=True,
                    p=weights,
                )
            else:
                sample_idx = np.random.randint(0, valid_ranges.size, size=L)
                sampled_range = valid_ranges[sample_idx]

            tau_clean = (2.0 * sampled_range / SENSOR.c_light).astype(np.float32)

            detection_mask = detection_mask_all[:, y, x]
            tau_noisy = tau_clean + jitter_all[:, y, x]

            sampled_depths[:, y, x] = sampled_range.astype(np.float32)
            timestamps_clean[:, y, x] = tau_clean
            timestamps_noisy[detection_mask, y, x] = tau_noisy[detection_mask]

    return TimestampBlock(
        frame_number=frame_number,
        sampled_depths_m=sampled_depths,
        timestamps_clean_s=timestamps_clean,
        detection_mask=detection_mask_all,
        timestamps_noisy_s=timestamps_noisy,
        source_depth_file=source_depth_file,
        source_normal_file=source_normal_file,
    )


# =========================
# Ray Angle
# =========================

def pixel_ray_cos(
    x: int,
    y: int,
    tof_w: int,
    tof_h: int,
    fov_x_deg: float = SENSOR.camera_fov_x_deg,
    fov_y_deg: float = SENSOR.camera_fov_y_deg,
) -> float:
    """
    Computes cos(theta) for the ray through one ToF pixel.

    theta is the angle between this pixel's camera ray and the camera's
    forward optical axis.

    Center pixel:
        cos(theta) ≈ 1

    Edge/corner pixels:
        cos(theta) < 1
        range = z_depth / cos(theta)
    """
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)

    # Pixel center coordinates normalized to [-1, 1]
    nx = ((x + 0.5) / tof_w) * 2.0 - 1.0
    ny = ((y + 0.5) / tof_h) * 2.0 - 1.0

    # Convert normalized image position to ray slope
    ray_x = nx * np.tan(fov_x / 2.0)
    ray_y = ny * np.tan(fov_y / 2.0)
    ray_z = 1.0

    ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)

    # Dot product with forward axis [0, 0, 1]
    return ray_z / ray_norm


# =========================
# Timestamp simulation
# =========================

def simulate_timestamp_block(
    depth: np.ndarray,
    normals: np.ndarray | None,
    frame_number: int,
    source_depth_file: str,
    source_normal_file: str,
    ray_cos_map: np.ndarray,
    tof_h: int = SENSOR.tof_h,
    tof_w: int = SENSOR.tof_w,
    L: int = SENSOR.block_size_L,
    rho: float = SENSOR.detection_probability_rho,
    jitter_std: float = SENSOR.timing_jitter_std_s,
) -> TimestampBlock:
    """
    Converts one rendered depth/normal frame into a TimestampBlock.

    Clean model:
        sampled_depth -> tau_clean = 2d/c

    Noisy model:
        Bernoulli detection mask + Gaussian timing jitter.
        Missed detections are stored as NaN.
    """
    h, w = depth.shape
    cell_h = h // tof_h
    cell_w = w // tof_w

    if cell_h <= 0 or cell_w <= 0:
        raise ValueError(f"Depth map {depth.shape} is too small for ToF grid {tof_h}x{tof_w}.")

    if normals is not None:
        if normals.shape[0] != h or normals.shape[1] != w:
            raise ValueError(f"Normal map shape {normals.shape} does not match depth shape {depth.shape}.")

    sampled_depths = np.full((L, tof_h, tof_w), np.nan, dtype=np.float32)
    timestamps_clean = np.full((L, tof_h, tof_w), np.nan, dtype=np.float32)
    timestamps_noisy = np.full((L, tof_h, tof_w), np.nan, dtype=np.float32)
    
    detection_mask_all = np.random.rand(L, tof_h, tof_w) < rho
    jitter_all = (np.random.randn(L, tof_h, tof_w).astype(np.float32) * jitter_std)

    for y in range(tof_h):
        y0 = y * cell_h
        y1 = (y + 1) * cell_h if y < tof_h - 1 else h

        for x in range(tof_w):
            x0 = x * cell_w
            x1 = (x + 1) * cell_w if x < tof_w - 1 else w

            depth_block = depth[y0:y1, x0:x1]
            valid = (
                np.isfinite(depth_block) 
                & (depth_block > SENSOR.min_valid_depth_m)
                & (depth_block < SENSOR.max_valid_depth_m)
            )

            if not np.any(valid):
                continue

            valid_depths = depth_block[valid].astype(np.float32)

            weights = None

            if normals is not None and USE_WEIGHTED_DEPTH_SAMPLING:
                normal_block = normals[y0:y1, x0:x1, :]
                valid_normals = normal_block[valid]

                nz = valid_normals[:, 2]
                facing = np.abs(nz)
                facing = np.where(np.isfinite(facing), facing, 0.0)

                distance_falloff = 1.0 / np.maximum(valid_depths ** 2, 1e-6)

                weights = facing * distance_falloff
                weights = weights.astype(np.float64)

                if np.sum(weights) <= 0 or not np.all(np.isfinite(weights)):
                    weights = np.ones_like(valid_depths, dtype=np.float64)

                weights = weights / np.sum(weights)

            # Sample one geometric depth for every pulse.
            # This gives a complete clean model before missed detections.
            if weights is not None:
                sampled = np.random.choice(valid_depths, size=L, replace=True, p=weights)
            else:
                sample_idx = np.random.randint(0, valid_depths.size, size=L)
                sampled = valid_depths[sample_idx]

            ray_cos = ray_cos_map[y, x]
            
            sampled_range = sampled / ray_cos
            tau_clean = (2.0 * sampled_range / SENSOR.c_light).astype(np.float32)

            detection_mask = detection_mask_all[:, y, x]
            tau_noisy = tau_clean + jitter_all[:, y, x]

            sampled_depths[:, y, x] = sampled_range.astype(np.float32)
            timestamps_clean[:, y, x] = tau_clean.astype(np.float32)
            timestamps_noisy[detection_mask, y, x] = tau_noisy[detection_mask].astype(np.float32)

    return TimestampBlock(
        frame_number=frame_number,
        sampled_depths_m=sampled_depths,
        timestamps_clean_s=timestamps_clean,
        detection_mask=detection_mask_all,
        timestamps_noisy_s=timestamps_noisy,
        source_depth_file=source_depth_file,
        source_normal_file=source_normal_file,
    )
    
    
# =========================
# Generating Histograms
# =========================

def block_depth_estimate_histogram(
    timestamps: np.ndarray,
    tau_edges: np.ndarray,
    bin_centers_tau: np.ndarray,
):
    """
    Estimate tau_hat from mini-histograms for all ToF pixels.

    timestamps: [L, TOF_H, TOF_W] in seconds, NaN for missed pulses.

    This version vectorizes histogram construction across all pixels.
    """
    L, H, W = timestamps.shape
    num_bins = len(bin_centers_tau)
    num_pixels = H * W

    histograms_flat = np.zeros((num_pixels, num_bins), dtype=np.uint16)
    tau_hat_flat = np.full(num_pixels, np.nan, dtype=np.float32)

    valid = np.isfinite(timestamps)

    if not np.any(valid):
        return (
            tau_hat_flat.reshape(H, W),
            histograms_flat.reshape(H, W, num_bins),
        )

    # Flatten timestamps and corresponding pixel ids.
    ts_flat = timestamps.reshape(L, num_pixels)
    valid_flat = valid.reshape(L, num_pixels)

    pulse_idx, pixel_idx = np.nonzero(valid_flat)
    ts_valid = ts_flat[pulse_idx, pixel_idx]

    # Convert timestamps to histogram bin indices.
    bin_idx = np.searchsorted(tau_edges, ts_valid, side="right") - 1

    in_range = (bin_idx >= 0) & (bin_idx < num_bins)
    pixel_idx = pixel_idx[in_range]
    bin_idx = bin_idx[in_range]

    if bin_idx.size == 0:
        return (
            tau_hat_flat.reshape(H, W),
            histograms_flat.reshape(H, W, num_bins),
        )

    # Combine pixel id and bin id into one flat index.
    flat_hist_idx = pixel_idx * num_bins + bin_idx

    counts_flat = np.bincount(
        flat_hist_idx,
        minlength=num_pixels * num_bins,
    )

    histograms_flat = counts_flat.reshape(num_pixels, num_bins).astype(np.uint16)

    # Peak bin per pixel.
    peak_bins = np.argmax(histograms_flat, axis=1)
    peak_counts = histograms_flat[np.arange(num_pixels), peak_bins]

    active_pixels = np.nonzero(peak_counts > 0)[0]

    # Centroid around peak. This loop is now only over pixels with detections.
    for p in active_pixels:
        peak_bin = peak_bins[p]

        lo = max(0, peak_bin - 2)
        hi = min(num_bins, peak_bin + 3)

        local_counts = histograms_flat[p, lo:hi].astype(np.float64)
        local_bins = bin_centers_tau[lo:hi].astype(np.float64)

        total = np.sum(local_counts)

        if total > 0:
            tau_hat_flat[p] = np.sum(local_counts * local_bins) / total

    return (
        tau_hat_flat.reshape(H, W),
        histograms_flat.reshape(H, W, num_bins),
    )


def coarse_tau_estimate_median(timestamps: np.ndarray):
    return np.nanmedian(timestamps, axis=0).astype(np.float32)


def adaptive_time_gate(timestamps, coarse_tau_hat, gate_tau):
    
    diff = np.abs(timestamps - coarse_tau_hat[np.newaxis])
    keep = np.isfinite(timestamps) & (diff <= gate_tau)

    gated = np.where(keep, timestamps, np.nan)
    return gated


def compute_pulse_streams(timestamps, tau_hat, delta_tau):
    """
    Classifies each pulse as in-window (S1) or out-of-window (S2+).

    timestamps: [L, tof_h, tof_w]  — NaN = missed pulse
    tau_hat:    [tof_h, tof_w]      — block-rate depth estimate (seconds)
    delta_m:    window half-width in meters

    Returns:
        S1:   float32 [tof_h, tof_w], fraction of pulses inside window
        S2p:  float32 [tof_h, tof_w], fraction of pulses outside window
    """
    diff = np.abs(timestamps - tau_hat[np.newaxis])   # [L, tof_h, tof_w]
    valid = np.isfinite(timestamps)

    # Missed pulses register as 0 in both streams (paper Section 3.4)
    in_window  = valid & (diff <= delta_tau)
    out_window = valid & (diff >  delta_tau)

    L = timestamps.shape[0]
    S1  = in_window.sum(axis=0).astype(np.float32)  / L
    S2p = out_window.sum(axis=0).astype(np.float32) / L

    return S1, S2p


def compute_valid_detection_fraction(timestamps: np.ndarray):
    """
    I[k] = 1 if a valid timestamp is recorded, else 0.
    
    Compress the L pulse-rate detections in one mini-histogram block
    into a per-pixel valid detection fraction in [0, 1]. This becomes I[l].
    """
    return np.isfinite(timestamps).sum(axis=0).astype(np.float32) / timestamps.shape[0]


# =========================
# Main
# =========================

def main():
    np.random.seed(RANDOM_SEED)

    depth_files = find_depth_files(DEPTH_DIR)
    normal_files = find_normal_files(NORMAL_DIR)

    if len(depth_files) != len(normal_files):
        raise RuntimeError(
            f"Depth/normal mismatch: {len(depth_files)} depth files, "
            f"{len(normal_files)} normal files."
        )

    print(f"Found {len(depth_files)} frames.")
    print("Generating timestamp dataset...")

    blocks = [] if SAVE_FULL_TIMESTAMP_DATASET else None
    tof_depths = []
    all_S1 = []
    all_S2p = []
    all_I = []
    all_histograms = []

    prev_tau_hat = None

    depth_edges = np.linspace(HIST_DEPTH_MIN_M, HIST_DEPTH_MAX_M, HIST_NUM_BINS + 1)
    tau_edges = SENSOR.depth_to_timestamp(depth_edges).astype(np.float32)
    hist_bin_centers_tau = (0.5 * (tau_edges[:-1] + tau_edges[1:])).astype(np.float32)
    hist_bin_centers_depth_m = SENSOR.timestamp_to_depth(hist_bin_centers_tau)

    first_depth = load_depth_file(depth_files[0])
    image_h, image_w = first_depth.shape
    
    ray_dirs_full = SENSOR.build_fullres_ray_dirs(image_h, image_w)
    ray_cos_map = SENSOR.build_ray_cos_map()

    gate_tau = SENSOR.depth_to_timestamp(ADAPTIVE_GATE_M)
    delta_tau = SENSOR.depth_to_timestamp(DELTA_WINDOW)

    def process_block(block):
        nonlocal prev_tau_hat

        if SAVE_FULL_TIMESTAMP_DATASET:
            blocks.append(block)

        timestamps_raw = block.timestamps_noisy_s
        coarse_tau_hat = coarse_tau_estimate_median(timestamps_raw)

        if USE_ADAPTIVE_TIME_GATING:
            timestamps = adaptive_time_gate(
                timestamps=timestamps_raw,
                coarse_tau_hat=coarse_tau_hat,
                gate_tau=gate_tau,
            )
        else:
            timestamps = timestamps_raw

        curr_tau_hat, histograms = block_depth_estimate_histogram(
            timestamps=timestamps,
            tau_edges=tau_edges,
            bin_centers_tau=hist_bin_centers_tau,
        )

        tof_depth_hist = SENSOR.timestamp_to_depth(curr_tau_hat)

        reference_tau_hat = curr_tau_hat if prev_tau_hat is None else prev_tau_hat

        S1, S2p = compute_pulse_streams(
            timestamps=timestamps,
            tau_hat=reference_tau_hat,
            delta_tau=delta_tau,
        )

        I = compute_valid_detection_fraction(timestamps)

        tof_depths.append(tof_depth_hist)
        all_S1.append(S1)
        all_S2p.append(S2p)
        all_I.append(I)
        all_histograms.append(histograms)

        prev_tau_hat = curr_tau_hat

    if USE_INTERPOLATED_VISIBILITY_SWITCH:
        frame_pairs = list(zip(
            depth_files[:-1],
            depth_files[1:],
            normal_files[:-1],
            normal_files[1:],
        ))

        total_blocks = len(frame_pairs) * NUM_INTERPOLATION_STEPS

        with tqdm(
            total=total_blocks,
            desc="Generating timestamp blocks",
            unit="block",
        ) as pbar:
            output_frame_number = 1

            for pair_i, (depth_path1, depth_path2, normal_path1, normal_path2) in enumerate(frame_pairs):
                depth1 = load_depth_file(depth_path1)
                depth2 = load_depth_file(depth_path2)

                normals1 = load_normal_file(normal_path1)
                normals2 = load_normal_file(normal_path2)

                for interp_i in range(NUM_INTERPOLATION_STEPS):
                    alpha = interp_i / float(NUM_INTERPOLATION_STEPS)

                    block = simulate_timestamp_block_interpolated_pair(
                        depth1=depth1,
                        depth2=depth2,
                        normals1=normals1,
                        normals2=normals2,
                        alpha=alpha,
                        frame_number=output_frame_number,
                        source_depth_file=f"{depth_path1} -> {depth_path2}, alpha={alpha:.3f}",
                        source_normal_file=f"{normal_path1} -> {normal_path2}, alpha={alpha:.3f}",
                        ray_dirs_full=ray_dirs_full,
                    )

                    process_block(block)

                    pbar.update(1)
                    output_frame_number += 1
    else:
        frame_items = list(zip(depth_files, normal_files))

        for frame_i, (depth_path, normal_path) in enumerate(
            tqdm(frame_items, desc="Generating timestamp blocks", unit="frame")
        ):
            frame_number = frame_i + 1

            depth = load_depth_file(depth_path)
            normals = load_normal_file(normal_path)

            block = simulate_timestamp_block(
                depth=depth,
                normals=normals,
                frame_number=frame_number,
                source_depth_file=str(depth_path),
                source_normal_file=str(normal_path),
                ray_cos_map=ray_cos_map,
            )

            process_block(block)
            
    if len(tof_depths) == 0:
        raise RuntimeError("No timestamp blocks were generated.")

    tof_depths = np.stack(tof_depths, axis=0)
    all_S1 = np.stack(all_S1, axis=0)
    all_S2p = np.stack(all_S2p, axis=0)
    all_I = np.stack(all_I, axis=0)
    all_histograms = np.stack(all_histograms, axis=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_PRECOMPUTED_DATA:
        precomputed_path = OUTPUT_DIR / "timestamp_precomputed.npz"

        np.savez(
            precomputed_path,
            tof_depths=tof_depths,
            all_S1=all_S1,
            all_S2p=all_S2p,
            all_I=all_I,
            all_histograms=all_histograms,
            hist_bin_centers_tau=hist_bin_centers_tau,
            hist_bin_centers_depth_m=hist_bin_centers_depth_m,
        )

        print(f"Saved precomputed timestamp/histogram data to: {precomputed_path}")

    if SAVE_FULL_TIMESTAMP_DATASET:
        metadata = TimestampMetadata(
            tof_h=SENSOR.tof_h,
            tof_w=SENSOR.tof_w,
            block_size_L=SENSOR.block_size_L,
            laser_rate_hz=SENSOR.laser_rate_hz,
            detection_probability_rho=SENSOR.detection_probability_rho,
            timing_jitter_std_s=SENSOR.timing_jitter_std_s,
            fps=EFFECTIVE_FPS,
            dt_s=EFFECTIVE_DT,
            c_light=SENSOR.c_light,
            min_valid_depth_m=SENSOR.min_valid_depth_m,
            max_valid_depth_m=SENSOR.max_valid_depth_m,
            use_weighted_depth_sampling=USE_WEIGHTED_DEPTH_SAMPLING,
        )

        dataset = TimestampDataset(blocks=blocks, metadata=metadata)
        dataset.save(OUTPUT_DIR)

    print("Done.")
    
    
if __name__ == "__main__":
    main()