r"""
Timestamp Generation:

    This script converts rendered VisionSIM/Blender depth and normal maps into
    simulated single-photon ToF timestamp datasets.

    For each rendered frame, the script samples high-resolution depth values
    inside each lower-resolution ToF pixel footprint. This preserves sub-pixel
    depth mixtures at object boundaries and occlusions. The sampled depths are
    converted to clean photon timestamps, then a Bernoulli detection model and
    Gaussian timing jitter are applied to create noisy single-photon timestamp
    measurements.

    Temporal interpolation estimates scene states between adjacent rendered
    frames at each ToF block time. Timestamp blocks are generated according to
    the sensor laser rate and block size.

    Same-surface geometry is linearly interpolated between rendered frames,
    while likely visibility changes use a hard switch near alpha = 0.5.

    The script can save:
        - full timestamp blocks
        - sampled ray/range depths
        - clean timestamps
        - noisy timestamps
        - detection masks
        - mini-histograms
        - histogram-based depth estimates
        - valid detection fraction

    Running:
        python timestamp_gen.py --sensor vl53l8ch --render-dir inputs\drone_flyby --output-dir outputs\examples\drone_flyby_lab

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
import argparse

from sensor_presets import get_sensor_preset
from timestamp_dataset import TimestampBlock, TimestampDataset, TimestampMetadata

# =========================
# User settings
# =========================

SENSOR = get_sensor_preset("generic_spad_sensor")

RENDER_FPS = 240.0
RENDER_DT = 1.0 / RENDER_FPS

USE_WEIGHTED_DEPTH_SAMPLING = True

RANDOM_SEED = 0

HIST_DEPTH_MIN_M = 0.1
HIST_DEPTH_MAX_M = 10.0
HIST_NUM_BINS = 32

USE_ADAPTIVE_TIME_GATING = False
ADAPTIVE_GATE_M = 1.0

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
# Timestamp simulation
# =========================


def normalize_vectors(v, eps=1e-12):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)
       
        
def simulate_timestamp_block(
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


def compute_valid_detection_fraction(timestamps: np.ndarray):
    """
    I[k] = 1 if a valid timestamp is recorded, else 0.
    
    Compress the L pulse-rate detections in one mini-histogram block
    into a per-pixel valid detection fraction in [0, 1]. This becomes I[l].
    """
    return np.isfinite(timestamps).sum(axis=0).astype(np.float32) / timestamps.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated single-photon ToF timestamp data from "
            "rendered depth and normal EXR files."
        )
    )

    parser.add_argument(
        "--sensor",
        default="generic_spad_sensor",
        help=(
            "Name of the ToF sensor preset to use from configs/sensors/{name}.yaml. "
            "Default: generic_spad_sensor"
        ),
    )

    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path("full_render_320x160"),
        help=(
            "Directory containing depths/ and normals/ subfolders. "
            "Default: full_render_320x160"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("timestamp_output"),
        help="Directory where timestamp outputs will be saved. Default: timestamp_output",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for reproducible timestamp simulation. Default: 0",
    )

    parser.add_argument(
        "--render-fps",
        type=float,
        default=240.0,
        help="Frame rate of the rendered input sequence. Default: 240.0",
    )

    parser.add_argument(
        "--hist-bins",
        type=int,
        default=32,
        help="Number of histogram bins for timestamp depth estimation. Default: 256",
    )

    parser.add_argument(
        "--hist-depth-min",
        type=float,
        default=0.1,
        help="Minimum histogram depth in meters. Default: 0.1",
    )

    parser.add_argument(
        "--hist-depth-max",
        type=float,
        default=20.0,
        help="Maximum histogram depth in meters. Default: 20.0",
    )

    parser.add_argument(
        "--adaptive-gate-m",
        type=float,
        default=1.0,
        help="Adaptive time-gate half-width in meters. Default: 1.0",
    )

    parser.add_argument(
        "--no-full-dataset",
        action="store_true",
        help="Do not save full per-frame timestamp dataset.",
    )

    parser.add_argument(
        "--no-precomputed",
        action="store_true",
        help="Do not save timestamp_precomputed.npz.",
    )

    return parser.parse_args()


# =========================
# Main
# =========================

def main():
    global SENSOR

    args = parse_args()

    SENSOR = get_sensor_preset(args.sensor)

    render_dir = args.render_dir
    depth_dir = render_dir / "depths"
    normal_dir = render_dir / "normals"
    output_dir = args.output_dir

    render_fps = args.render_fps
    
    if render_fps <= 0:
        raise ValueError("--render-fps must be greater than 0.")
    
    render_dt = 1.0 / render_fps

    block_duration_s = SENSOR.block_size_L / SENSOR.laser_rate_hz
    block_rate_hz = SENSOR.laser_rate_hz / SENSOR.block_size_L

    hist_depth_min_m = args.hist_depth_min
    hist_depth_max_m = args.hist_depth_max
    hist_num_bins = args.hist_bins

    adaptive_gate_m = args.adaptive_gate_m

    save_full_timestamp_dataset = not args.no_full_dataset
    save_precomputed_data = not args.no_precomputed

    switch_dist_thresh_m = MAX_SAME_SURFACE_SPEED_M_PER_S * render_dt
    
    print(f"Using sensor preset: {SENSOR.name}")
    print(f"Sensor grid: {SENSOR.tof_h} x {SENSOR.tof_w}")
    print(f"Laser rate: {SENSOR.laser_rate_hz:.3g} Hz")
    print(f"Block size: {SENSOR.block_size_L} pulses")
    print(
        "Expected detections/pixel/block: "
        f"{SENSOR.block_size_L * SENSOR.detection_probability_rho:.2f}"
    )

    np.random.seed(args.random_seed)

    depth_files = find_depth_files(depth_dir)
    normal_files = find_normal_files(normal_dir)
    
    if len(depth_files) < 2:
        raise RuntimeError(
            "At least 2 rendered frames are required to define a scene timeline."
        )

    if len(depth_files) != len(normal_files):
        raise RuntimeError(
            f"Depth/normal mismatch: {len(depth_files)} depth files, "
            f"{len(normal_files)} normal files."
        )

    print(f"Found {len(depth_files)} frames.")
    print("Generating timestamp dataset...")
    
    num_render_intervals = len(depth_files) - 1

    scene_duration_s = num_render_intervals / render_fps
    
    blocks_per_render_interval = block_rate_hz / render_fps

    expected_blocks = int(
        np.floor(scene_duration_s / block_duration_s)
    )

    samples_per_block_per_pixel = SENSOR.block_size_L
    samples_per_block_all_pixels = (
        SENSOR.block_size_L * SENSOR.tof_h * SENSOR.tof_w
    )
    total_samples_per_pixel = expected_blocks * SENSOR.block_size_L
    total_samples_all_pixels = (
        expected_blocks * SENSOR.block_size_L * SENSOR.tof_h * SENSOR.tof_w
    )

    active_tof_time_per_pixel_s = total_samples_per_pixel / SENSOR.laser_rate_hz
    represented_duration_s = expected_blocks * block_duration_s
    duration_error_s = scene_duration_s - represented_duration_s

    expected_detected_samples_per_pixel = (
        total_samples_per_pixel * SENSOR.detection_probability_rho
    )
    expected_detected_samples_all_pixels = (
        total_samples_all_pixels * SENSOR.detection_probability_rho
    )

    print()
    print("=== Timestamp sample summary ===")
    print(f"Render frames: {len(depth_files)}")
    print(f"Render intervals: {num_render_intervals}")
    print(f"Render FPS: {render_fps:.3f} Hz")
    print(f"Scene duration: {scene_duration_s * 1e3:.4f} ms")
    print(f"ToF blocks/render interval: {blocks_per_render_interval:.4f}")
    print(f"ToF block duration: {block_duration_s * 1e6:.4f} us")
    print(f"ToF block rate: {block_rate_hz:.4f} Hz")
    print(f"Expected timestamp blocks: {expected_blocks}")
    print(f"Samples/block/pixel: {samples_per_block_per_pixel}")
    print(f"Samples/block/all pixels: {samples_per_block_all_pixels:,}")
    print(f"Total samples/pixel: {total_samples_per_pixel:,}")
    print(f"Total samples/all pixels: {total_samples_all_pixels:,}")
    print(f"Active ToF sampling time/pixel: {active_tof_time_per_pixel_s * 1e3:.4f} ms")
    print(f"ToF duration represented: {represented_duration_s * 1e3:.4f} ms")
    print(f"Unrepresented trailing scene time: {duration_error_s * 1e6:.4f} us")
    print(f"Expected detected samples/pixel: {expected_detected_samples_per_pixel:,.2f}")
    print(f"Expected detected samples/all pixels: {expected_detected_samples_all_pixels:,.2f}")
    print()

    blocks = [] if save_full_timestamp_dataset else None
    tof_depths = []
    all_I = []
    all_histograms = []

    depth_edges = np.linspace(hist_depth_min_m, hist_depth_max_m, hist_num_bins + 1)
    tau_edges = SENSOR.depth_to_timestamp(depth_edges).astype(np.float32)
    hist_bin_centers_tau = (0.5 * (tau_edges[:-1] + tau_edges[1:])).astype(np.float32)
    hist_bin_centers_depth_m = SENSOR.timestamp_to_depth(hist_bin_centers_tau)

    first_depth = load_depth_file(depth_files[0])
    image_h, image_w = first_depth.shape
    
    ray_dirs_full = SENSOR.build_fullres_ray_dirs(image_h, image_w)

    gate_tau = SENSOR.depth_to_timestamp(adaptive_gate_m)

    def process_block(block):
        if save_full_timestamp_dataset:
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

        I = compute_valid_detection_fraction(timestamps)

        tof_depths.append(tof_depth_hist)
        all_I.append(I)
        all_histograms.append(histograms)

    with tqdm(
        total=expected_blocks,
        desc="Generating timestamp blocks",
        unit="block",
    ) as pbar:

        current_pair_i = None

        depth1 = None
        depth2 = None
        normals1 = None
        normals2 = None

        for block_idx in range(expected_blocks):
            # Actual scene time represented by this ToF block.
            block_time_s = block_idx * block_duration_s

            # Continuous location in the rendered frame sequence.
            render_position = block_time_s * render_fps

            pair_i = int(np.floor(render_position))
            pair_i = min(pair_i, len(depth_files) - 2)

            alpha = render_position - pair_i

            # Only reload EXRs when entering a new render pair.
            if pair_i != current_pair_i:
                depth_path1 = depth_files[pair_i]
                depth_path2 = depth_files[pair_i + 1]

                normal_path1 = normal_files[pair_i]
                normal_path2 = normal_files[pair_i + 1]

                depth1 = load_depth_file(depth_path1)
                depth2 = load_depth_file(depth_path2)

                normals1 = load_normal_file(normal_path1)
                normals2 = load_normal_file(normal_path2)

                current_pair_i = pair_i

            block = simulate_timestamp_block(
                depth1=depth1,
                depth2=depth2,
                normals1=normals1,
                normals2=normals2,
                alpha=alpha,
                frame_number=block_idx + 1,
                source_depth_file=(
                    f"{depth_files[pair_i]} -> "
                    f"{depth_files[pair_i + 1]}, "
                    f"alpha={alpha:.6f}"
                ),
                source_normal_file=(
                    f"{normal_files[pair_i]} -> "
                    f"{normal_files[pair_i + 1]}, "
                    f"alpha={alpha:.6f}"
                ),
                ray_dirs_full=ray_dirs_full,
                tof_h=SENSOR.tof_h,
                tof_w=SENSOR.tof_w,
                L=SENSOR.block_size_L,
                rho=SENSOR.detection_probability_rho,
                jitter_std=SENSOR.timing_jitter_std_s,
                switch_dist_thresh_m=switch_dist_thresh_m,
            )

            process_block(block)
            pbar.update(1)
            
    if len(tof_depths) == 0:
        raise RuntimeError("No timestamp blocks were generated.")
    
    if len(tof_depths) != expected_blocks:
        raise RuntimeError(
            f"Generated {len(tof_depths)} timestamp blocks, "
            f"but expected {expected_blocks}."
        )

    tof_depths = np.stack(tof_depths, axis=0)
    all_I = np.stack(all_I, axis=0)
    all_histograms = np.stack(all_histograms, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    if save_precomputed_data:
        precomputed_path = output_dir / "timestamp_precomputed.npz"

        np.savez(
            precomputed_path,
            tof_depths=tof_depths,
            all_I=all_I,
            all_histograms=all_histograms,
            hist_bin_centers_tau=hist_bin_centers_tau,
            hist_bin_centers_depth_m=hist_bin_centers_depth_m,
        )

        print(f"Saved precomputed timestamp/histogram data to: {precomputed_path}")

    if save_full_timestamp_dataset:
        metadata = TimestampMetadata(
            tof_h=SENSOR.tof_h,
            tof_w=SENSOR.tof_w,
            block_size_L=SENSOR.block_size_L,
            laser_rate_hz=SENSOR.laser_rate_hz,
            detection_probability_rho=SENSOR.detection_probability_rho,
            timing_jitter_std_s=SENSOR.timing_jitter_std_s,
            fps=block_rate_hz,
            dt_s=block_duration_s,
            c_light=SENSOR.c_light,
            min_valid_depth_m=SENSOR.min_valid_depth_m,
            max_valid_depth_m=SENSOR.max_valid_depth_m,
            use_weighted_depth_sampling=USE_WEIGHTED_DEPTH_SAMPLING,
        )

        dataset = TimestampDataset(blocks=blocks, metadata=metadata)
        dataset.save(output_dir)

    print("Done.")
    
    
if __name__ == "__main__":
    main()