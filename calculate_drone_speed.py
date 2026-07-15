"""
Calculate camera/drone translation speed for a Blender animation at different FPS values.

The script uses the camera path keyframes from create_drone_wall_approach_blend.py.
Changing Blender FPS changes the real-time duration between fixed frame-number
keyframes, so the same path moves faster at higher FPS.

Examples:
    python calculate_drone_speed.py

    python calculate_drone_speed.py --fps 30 60 120 240

    python calculate_drone_speed.py --fps-start 30 --fps-stop 240 --fps-step 30

    python calculate_drone_speed.py --csv drone_speed_by_fps.csv
"""

from __future__ import annotations

import argparse
import csv
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DRONE_PATH_CONFIG_DIR = Path("configs/drone_paths")


def resolve_path_config(path_arg: Path) -> Path:
    """
    Resolve a drone path name or explicit YAML path.
    """
    # Explicit existing path.
    if path_arg.exists():
        return path_arg

    # Add .yaml when only a config name was provided.
    config_name = path_arg

    if config_name.suffix == "":
        config_name = config_name.with_suffix(".yaml")

    config_path = DRONE_PATH_CONFIG_DIR / config_name

    if not config_path.exists():
        raise FileNotFoundError(
            f"Drone path config not found: {config_path}"
        )

    return config_path


def load_path_config(path: Path) -> dict:
    """
    Load camera path keyframes and segment names from YAML.
    """
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if "camera_path" not in config:
        raise ValueError(
            f"Path config is missing 'camera_path': {path}"
        )

    camera_path = [
        (
            int(item["frame"]),
            tuple(float(value) for value in item["position"]),
        )
        for item in config["camera_path"]
    ]

    if len(camera_path) < 2:
        raise ValueError(
            "Camera path must contain at least two keyframes."
        )

    segment_names = config.get("segment_names")

    if segment_names is None:
        segment_names = [
            f"Segment {index + 1}"
            for index in range(len(camera_path) - 1)
        ]

    if len(segment_names) != len(camera_path) - 1:
        raise ValueError(
            "segment_names must contain one name for each "
            "camera path segment."
        )

    return {
        "name": config.get("name", path.stem),
        "description": config.get("description", ""),
        "camera_path": camera_path,
        "segment_names": segment_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate drone/camera translation speed for the fixed Blender "
            "keyframe path at different render FPS values."
        )
    )

    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Specific FPS values to test. Example: --fps 30 60 120 240. "
            "If omitted, defaults to 30, 60, 120, and 240 FPS."
        ),
    )
    
    parser.add_argument(
        "--path-config",
        type=Path,
        required=True,
        help=(
            "Drone path config name or YAML path. "
            "Names are loaded from configs/drone_paths."
        ),
    )

    parser.add_argument(
        "--fps-start",
        type=float,
        default=None,
        help="Start of an FPS sweep.",
    )

    parser.add_argument(
        "--fps-stop",
        type=float,
        default=None,
        help="End of an FPS sweep, inclusive when reached exactly.",
    )

    parser.add_argument(
        "--fps-step",
        type=float,
        default=None,
        help="Step size for an FPS sweep.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for saving detailed results as CSV.",
    )

    return parser.parse_args()


def resolve_fps_values(args: argparse.Namespace) -> list[float]:
    if args.fps is not None:
        fps_values = args.fps

    elif (
        args.fps_start is not None
        or args.fps_stop is not None
        or args.fps_step is not None
    ):
        if (
            args.fps_start is None
            or args.fps_stop is None
            or args.fps_step is None
        ):
            raise ValueError(
                "--fps-start, --fps-stop, and --fps-step must be provided together."
            )

        if args.fps_step <= 0:
            raise ValueError("--fps-step must be greater than 0.")

        fps_values = np.arange(
            args.fps_start,
            args.fps_stop + 0.5 * args.fps_step,
            args.fps_step,
        ).tolist()

    else:
        fps_values = [
            24.0,
            30.0,
            48.0,
            50.0,
            60.0,
            90.0,
            100.0,
            120.0,
            144.0,
            165.0,
            180.0,
            200.0,
            240.0,
        ]

    if any(fps <= 0 for fps in fps_values):
        raise ValueError("All FPS values must be greater than 0.")

    return sorted(set(float(fps) for fps in fps_values))


def calculate_segment_results(
    fps: float,
    camera_path: list[tuple],
) -> list[dict]:
    results = []

    for segment_idx in range(len(camera_path) - 1):
        frame_start, position_start = camera_path[segment_idx]
        frame_end, position_end = camera_path[segment_idx + 1]

        p0 = np.asarray(position_start, dtype=np.float64)
        p1 = np.asarray(position_end, dtype=np.float64)

        distance_m = float(np.linalg.norm(p1 - p0))
        frame_intervals = frame_end - frame_start
        duration_s = frame_intervals / fps
        speed_m_per_s = distance_m / duration_s

        results.append(
            {
                "fps": fps,
                "segment": segment_idx + 1,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_intervals": frame_intervals,
                "distance_m": distance_m,
                "duration_s": duration_s,
                "speed_m_per_s": speed_m_per_s,
            }
        )

    return results


def summarize_fps(
    fps: float,
    results: list[dict],
    camera_path: list[tuple],
) -> dict:
    total_distance_m = sum(row["distance_m"] for row in results)

    first_frame = camera_path[0][0]
    last_frame = camera_path[-1][0]
    total_frame_intervals = last_frame - first_frame
    total_duration_s = total_frame_intervals / fps

    average_path_speed_m_per_s = total_distance_m / total_duration_s
    max_segment_speed_m_per_s = max(row["speed_m_per_s"] for row in results)

    return {
        "fps": fps,
        "total_distance_m": total_distance_m,
        "total_duration_s": total_duration_s,
        "average_path_speed_m_per_s": average_path_speed_m_per_s,
        "max_segment_speed_m_per_s": max_segment_speed_m_per_s,
    }


def print_results(
    fps_values: list[float],
    all_results: list[dict],
    camera_path: list[tuple],
) -> None:
    print("=== Drone speed by FPS ===")
    print()

    for fps in fps_values:
        results = [row for row in all_results if row["fps"] == fps]
        summary = summarize_fps(
            fps,
            results,
            camera_path,
        )

        print(f"FPS: {fps:g}")
        print(f"Total path distance: {summary['total_distance_m']:.3f} m")
        print(f"Animation duration: {summary['total_duration_s']:.4f} s")
        print(
            "Average path speed: "
            f"{summary['average_path_speed_m_per_s']:.3f} m/s"
        )
        print(
            "Maximum segment speed: "
            f"{summary['max_segment_speed_m_per_s']:.3f} m/s"
        )

        print("Segments:")
        for row in results:
            print(
                f"  {row['segment']}: "
                f"frames {row['frame_start']} -> {row['frame_end']}, "
                f"distance={row['distance_m']:.3f} m, "
                f"duration={row['duration_s']:.4f} s, "
                f"speed={row['speed_m_per_s']:.3f} m/s"
            )

        print()
        
        
def save_speed_scatter(
    path: Path,
    all_results: list[dict],
    path_name: str,
    segment_names: list[str],
) -> None:
    """
    Plot render FPS versus camera translation speed.

    Each curve represents one camera path segment.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for segment_idx, segment_name in enumerate(
        segment_names,
        start=1,
    ):
        segment_rows = [
            row
            for row in all_results
            if row["segment"] == segment_idx
        ]

        fps = [row["fps"] for row in segment_rows]
        speed = [
            row["speed_m_per_s"]
            for row in segment_rows
        ]

        plt.plot(
            fps,
            speed,
            marker="o",
            linewidth=1.0,
            markersize=5,
            label=segment_name,
        )

    plt.xlabel("Blender render rate (FPS)")
    plt.ylabel("Camera translation speed (m/s)")

    plt.title(
        f"Camera translation speed: {path_name}"
    )

    plt.axvline(
        x=240,
        linestyle="--",
        linewidth=1.0,
        label="240 FPS",
    )

    plt.grid(True, alpha=0.3)

    plt.legend(
        fontsize=8,
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved speed plot: {path.resolve()}")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "fps",
        "segment",
        "frame_start",
        "frame_end",
        "frame_intervals",
        "distance_m",
        "duration_s",
        "speed_m_per_s",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path.resolve()}")


def main() -> None:
    args = parse_args()

    path_config_path = resolve_path_config(args.path_config)

    path_config = load_path_config(path_config_path)

    path_name = path_config["name"]
    camera_path = path_config["camera_path"]
    segment_names = path_config["segment_names"]

    fps_values = resolve_fps_values(args)

    all_results = []

    for fps in fps_values:
        all_results.extend(
            calculate_segment_results(
                fps=fps,
                camera_path=camera_path,
            )
        )

    print(f"Path: {path_name}")
    print()

    print_results(
        fps_values=fps_values,
        all_results=all_results,
        camera_path=camera_path,
    )

    save_speed_scatter(
        path=Path(f"{path_name}_speed_vs_fps.png"),
        all_results=all_results,
        path_name=path_name,
        segment_names=segment_names,
    )

    if args.csv is not None:
        save_csv(
            args.csv,
            all_results,
        )


if __name__ == "__main__":
    main()
