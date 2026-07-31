from __future__ import annotations

import numpy as np


def causal_nanmean(
    values: np.ndarray,
    window_blocks: int,
) -> np.ndarray:
    if window_blocks <= 1:
        return values.astype(np.float32, copy=True)

    finite = np.isfinite(values)

    sums = np.cumsum(
        np.where(finite, values, 0.0),
        axis=0,
        dtype=np.float64,
    )

    counts = np.cumsum(
        finite.astype(np.int32),
        axis=0,
        dtype=np.int64,
    )

    sums = np.concatenate(
        [np.zeros_like(sums[:1]), sums],
        axis=0,
    )

    counts = np.concatenate(
        [np.zeros_like(counts[:1]), counts],
        axis=0,
    )

    end_idx = np.arange(1, values.shape[0] + 1)
    start_idx = np.maximum(0, end_idx - window_blocks)

    window_sums = sums[end_idx] - sums[start_idx]
    window_counts = counts[end_idx] - counts[start_idx]

    result = np.full(
        values.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        window_sums,
        window_counts,
        out=result,
        where=window_counts > 0,
    )

    return result


def compute_time_to_contact(
    tof_depths: np.ndarray,
    block_times_s: np.ndarray,
    window_ms: float = 10.0,
    min_closing_speed_mps: float = 0.10,
    max_ttc_s: float = 10.0,
) -> dict[str, np.ndarray | int | float]:
    if window_ms <= 0:
        raise ValueError("TTC window must be positive.")

    if min_closing_speed_mps < 0:
        raise ValueError(
            "Minimum closing speed must be nonnegative."
        )

    if max_ttc_s <= 0:
        raise ValueError("Maximum TTC must be positive.")

    if tof_depths.shape[0] != block_times_s.shape[0]:
        raise ValueError(
            "block_times_s length must match tof_depths."
        )

    if block_times_s.size < 2:
        raise ValueError(
            "At least two timestamp blocks are required for TTC."
        )

    time_differences = np.diff(block_times_s)

    if (
        not np.all(np.isfinite(time_differences))
        or np.any(time_differences <= 0)
    ):
        raise ValueError(
            "Block times must be finite and strictly increasing."
        )

    median_dt_s = float(np.median(time_differences))

    window_blocks = max(
        1,
        int(round((window_ms * 1e-3) / median_dt_s)),
    )

    smoothed_depths = causal_nanmean(
        tof_depths,
        window_blocks,
    )

    radial_velocity = np.full_like(
        smoothed_depths,
        np.nan,
        dtype=np.float32,
    )

    lag = window_blocks

    if lag < smoothed_depths.shape[0]:
        dt = (
            block_times_s[lag:]
            - block_times_s[:-lag]
        )

        depth_delta = (
            smoothed_depths[lag:]
            - smoothed_depths[:-lag]
        )

        velocity = np.full_like(
            depth_delta,
            np.nan,
            dtype=np.float32,
        )

        np.divide(
            depth_delta,
            dt[:, None, None],
            out=velocity,
            where=dt[:, None, None] > 0,
        )

        radial_velocity[lag:] = velocity

    closing_speed = -radial_velocity

    valid_ttc = (
        np.isfinite(smoothed_depths)
        & np.isfinite(closing_speed)
        & (smoothed_depths > 0)
        & (closing_speed >= min_closing_speed_mps)
    )

    ttc = np.full_like(
        smoothed_depths,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        smoothed_depths,
        closing_speed,
        out=ttc,
        where=valid_ttc,
    )

    ttc[(ttc <= 0) | (ttc > max_ttc_s)] = np.nan

    return {
        "smoothed_depths_m": smoothed_depths,
        "radial_velocity_mps": radial_velocity,
        "closing_speed_mps": closing_speed,
        "time_to_contact_s": ttc,
        "window_blocks": window_blocks,
        "median_dt_s": median_dt_s,
    }