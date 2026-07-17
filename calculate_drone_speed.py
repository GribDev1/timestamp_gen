"""
Calculate camera/drone translation speed for a Blender animation at different FPS values.
 
The script uses the camera path keyframes from create_drone_wall_approach_blend.py.
Changing Blender FPS changes the real-time duration between fixed frame-number
keyframes, so the same path moves faster at higher FPS.
 
Examples:
    python calculate_drone_speed.py --path-config wall_approach
 
    python calculate_drone_speed.py --path-config wall_approach --fps 30 60 120 240
 
    python calculate_drone_speed.py --path-config wall_approach \
        --fps-start 30 --fps-stop 240 --fps-step 30
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


DRONE_PATH_CONFIG_DIR = Path("configs/drone_paths")

DEFAULT_FPS_VALUES = [
    24.0, 30.0, 48.0, 50.0, 60.0, 90.0, 100.0,
    120.0, 144.0, 165.0, 180.0, 200.0, 240.0,
]


def resolve_path_config(path_arg: Path) -> Path:
    """
    Resolve a drone path name or explicit YAML path.
    """
    if path_arg.exists():
        return path_arg

    # Add .yaml when only a config name was provided.
    config_name = path_arg if path_arg.suffix else path_arg.with_suffix(".yaml")
    config_path = DRONE_PATH_CONFIG_DIR / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Drone path config not found: {config_path}")

    return config_path


def _load_camera_path(config: dict) -> list[tuple[int, tuple[float, ...]]]:
    if "camera_path" not in config:
        raise ValueError(f"Path config is missing 'camera_path'.")

    camera_path = [
        (int(item["frame"]), tuple(float(v) for v in item["position"]))
        for item in config["camera_path"]
    ]

    if len(camera_path) < 2:
        raise ValueError("Camera path must contain at least two keyframes.")


    frames = [frame for frame, _ in camera_path]
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise ValueError("camera_path keyframes must be in strictly increasing frame order.")
    
    return camera_path


def _load_camera_rotation(config: dict) -> list[tuple[int, tuple[float, ...]]]:
    camera_rotation = [
        (int(item["frame"]), tuple(float(v) for v in item["rotation_deg"]))
        for item in config.get("camera_rotation", [])
    ]

    if camera_rotation and len(camera_rotation) < 2:
        raise ValueError("camera_rotation must contain at least two keyframes.")

    frames = [frame for frame, _ in camera_rotation]

    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise ValueError("camera_rotation keyframes must be in strictly increasing frame order.")

    return camera_rotation


def _resolve_segment_names(
    names: list[str] | None,
    count: int,
    label: str,
) -> list[str]:
    if names is None:
        return [f"{label} {index + 1}" for index in range(count)]
    
    if len(names) != count:
        raise ValueError(f"{label.lower().replace(' ', '_')}_names must contain one name per segment")
    
    return names
    
    
def load_path_config(path: Path) -> dict:
    """Load camera path keyframes and segment names from YAML."""
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    camera_path = _load_camera_path(config)
    camera_rotation = _load_camera_rotation(config)

    segment_names = _resolve_segment_names(
        config.get("segment_names"), len(camera_path) - 1, "Segment"
    )

    rotation_segment_names = (
        _resolve_segment_names(
            config.get("rotation_segment_names"),
            len(camera_rotation) - 1,
            "Rotation segment",
        )
        if camera_rotation
        else []
    )

    return {
        "name": config.get("name", path.stem),
        "description": config.get("description", ""),
        "camera_path": camera_path,
        "segment_names": segment_names,
        "camera_rotation": camera_rotation,
        "rotation_segment_names": rotation_segment_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate drone/camera translation speed for the fixed Blender "
            "keyframe path at different render FPS values."
        )
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
        "--fps",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Specific FPS values to test. Example: --fps 30 60 120 240. Defaults to a preset sweep."),
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
        "--trace-samples-per-segment",
        type=int,
        default=100,
        help=(
            "Number of samples per keyframe segment for position and "
            "velocity traces. Default: 100"
        ),
    )

    return parser.parse_args()


def resolve_fps_values(args: argparse.Namespace) -> list[float]:
    if args.fps is not None:
        fps_values = args.fps

    elif any((args.fps_start, args.fps_stop, args.fps_step)):
        if None in (args.fps_start, args.fps_stop, args.fps_step):
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
        fps_values = DEFAULT_FPS_VALUES

    if any(fps <= 0 for fps in fps_values):
        raise ValueError("All FPS values must be greater than 0.")

    return sorted(set(float(fps) for fps in fps_values))


def euler_xyz_deg_to_matrix(
    rotation_deg: tuple[float, float, float],
) -> np.ndarray:
    """
    Convert Blender XYZ Euler rotation angles in degrees to a
    3x3 rotation matrix.
    """
    rx, ry, rz = np.deg2rad(rotation_deg)

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x],
            [0.0, sin_x, cos_x],
        ],
        dtype=np.float64,
    )

    rotation_y = np.array(
        [
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ],
        dtype=np.float64,
    )

    rotation_z = np.array(
        [
            [cos_z, -sin_z, 0.0],
            [sin_z, cos_z, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Blender XYZ Euler composition for column vectors.
    return rotation_z @ rotation_y @ rotation_x


def orientation_change_deg(
    rotation_start_deg: tuple[float, float, float],
    rotation_end_deg: tuple[float, float, float],
) -> float:
    """
    Return the shortest 3D rotation angle between two camera
    orientations.

    The result is between 0 and 180 degrees.
    """
    matrix_start = euler_xyz_deg_to_matrix(rotation_start_deg)
    matrix_end = euler_xyz_deg_to_matrix(rotation_end_deg)

    relative_rotation = matrix_end @ matrix_start.T
    cosine_angle = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)

    return float(np.rad2deg(np.arccos(cosine_angle)))


def calculate_segment_results(fps: float, camera_path: list[tuple]) -> list[dict]:
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


def calculate_rotation_results(fps: float, camera_rotation: list[tuple],) -> list[dict]:
    """
    Calculate average camera angular velocity for each rotation
    keyframe interval.
    """
    results = []

    for segment_idx in range(len(camera_rotation) - 1):
        frame_start, rotation_start = camera_rotation[segment_idx]
        frame_end, rotation_end = camera_rotation[segment_idx + 1]

        frame_intervals = frame_end - frame_start

        if frame_intervals <= 0:
            raise ValueError(
                "Camera rotation keyframes must be in "
                "strictly increasing frame order."
            )

        duration_s = frame_intervals / fps

        angle_change_deg = orientation_change_deg(rotation_start,rotation_end,)
        angular_velocity_deg_per_s = (angle_change_deg / duration_s)
        angular_velocity_rad_per_s = np.deg2rad(angular_velocity_deg_per_s)

        results.append({
            "fps": fps,
            "rotation_segment": segment_idx + 1,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_intervals": frame_intervals,
            "rotation_start_x_deg": rotation_start[0],
            "rotation_start_y_deg": rotation_start[1],
            "rotation_start_z_deg": rotation_start[2],
            "rotation_end_x_deg": rotation_end[0],
            "rotation_end_y_deg": rotation_end[1],
            "rotation_end_z_deg": rotation_end[2],
            "angle_change_deg": angle_change_deg,
            "duration_s": duration_s,
            "angular_velocity_deg_per_s": angular_velocity_deg_per_s,
            "angular_velocity_rad_per_s":angular_velocity_rad_per_s,
        })

    return results


def sample_position_trajectory(
    fps: float,
    camera_path: list[tuple[int, tuple[float, ...]]],
    samples_per_segment: int = 100,
) -> dict[str, np.ndarray]:
    """
    Sample camera position and translation velocity over time.

    Assumes linear interpolation between camera position keyframes.

    Time zero corresponds to the first camera-path keyframe.
    """

    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")

    first_frame = camera_path[0][0]

    all_times = []
    all_positions = []
    all_velocities = []
    all_segment_indices = []

    for segment_idx in range(len(camera_path) - 1):
        frame_start, position_start = camera_path[segment_idx]
        frame_end, position_end = camera_path[segment_idx + 1]

        p0 = np.asarray(position_start, dtype=np.float64)
        p1 = np.asarray(position_end, dtype=np.float64)

        time_start_s = (frame_start - first_frame) / fps
        time_end_s = (frame_end - first_frame) / fps
        duration_s = time_end_s - time_start_s

        if duration_s <= 0:
            raise ValueError(
                "Camera path keyframes must be in increasing frame order."
            )

        # Avoid duplicating a shared endpoint between adjacent segments.
        endpoint = segment_idx == len(camera_path) - 2

        alpha = np.linspace(
            0.0,
            1.0,
            samples_per_segment,
            endpoint=endpoint,
            dtype=np.float64,
        )

        times_s = time_start_s + alpha * duration_s

        positions = (
            (1.0 - alpha[:, None]) * p0[None, :]
            + alpha[:, None] * p1[None, :]
        )

        # Linear interpolation gives constant velocity within each segment.
        velocity = (p1 - p0) / duration_s
        velocities = np.repeat(
            velocity[None, :],
            len(alpha),
            axis=0,
        )

        all_times.append(times_s)
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_segment_indices.append(
            np.full(len(alpha), segment_idx + 1, dtype=np.int32)
        )

    times_s = np.concatenate(all_times)
    positions = np.concatenate(all_positions, axis=0)
    velocities = np.concatenate(all_velocities, axis=0)
    segment_indices = np.concatenate(all_segment_indices)

    speeds_m_per_s = np.linalg.norm(velocities, axis=1)

    return {
        "time_s": times_s,
        "position_x_m": positions[:, 0],
        "position_y_m": positions[:, 1],
        "position_z_m": positions[:, 2],
        "velocity_x_m_per_s": velocities[:, 0],
        "velocity_y_m_per_s": velocities[:, 1],
        "velocity_z_m_per_s": velocities[:, 2],
        "speed_m_per_s": speeds_m_per_s,
        "segment": segment_indices,
    }


def summarize_fps(fps: float, results: list[dict], camera_path: list[tuple]) -> dict:
    total_distance_m = sum(row["distance_m"] for row in results)

    first_frame = camera_path[0][0]
    last_frame = camera_path[-1][0]
    total_duration_s = (last_frame - first_frame) / fps

    average_path_speed_m_per_s = total_distance_m / total_duration_s
    max_segment_speed_m_per_s = max(row["speed_m_per_s"] for row in results)

    return {
        "fps": fps,
        "total_distance_m": total_distance_m,
        "total_duration_s": total_duration_s,
        "average_path_speed_m_per_s": average_path_speed_m_per_s,
        "max_segment_speed_m_per_s": max_segment_speed_m_per_s,
    }


def print_results(fps_values: list[float], all_results: list[dict],camera_path: list[tuple]) -> None:
    print("=== Drone speed by FPS ===")
    print()

    for fps in fps_values:
        results = [row for row in all_results if row["fps"] == fps]
        summary = summarize_fps(fps,results,camera_path,)

        print(f"FPS: {fps:g}")
        print(f"Total path distance: {summary['total_distance_m']:.3f} m")
        print(f"Animation duration: {summary['total_duration_s']:.4f} s")
        print(f"Average path speed: {summary['average_path_speed_m_per_s']:.3f} m/s")
        print(f"Maximum segment speed: {summary['max_segment_speed_m_per_s']:.3f} m/s")

        print("Segments:")
        for row in results:
            print(
                f"  {row['segment']}: frames {row['frame_start']} -> {row['frame_end']}, "
                f"distance={row['distance_m']:.3f} m, duration={row['duration_s']:.4f} s, "
                f"speed={row['speed_m_per_s']:.3f} m/s"
            )

        print()
        

def print_rotation_results(
    fps_values: list[float],
    all_rotation_results: list[dict],
    rotation_segment_names: list[str],
) -> None:
    if not all_rotation_results:
        return

    print("=== Camera angular velocity by FPS ===")
    print()

    for fps in fps_values:
        results = [row for row in all_rotation_results if row["fps"] == fps]

        max_angular_velocity = max(row["angular_velocity_deg_per_s"] for row in results)
        moving_results = [row for row in results if row["angle_change_deg"] > 0]

        total_angle_deg = sum(row["angle_change_deg"] for row in results)
        total_rotation_time_s = sum(row["duration_s"] for row in moving_results)

        average_turning_velocity = (
            total_angle_deg / total_rotation_time_s if total_rotation_time_s > 0 else 0.0
        )

        print(f"FPS: {fps:g}")
        print(f"Total accumulated orientation change: {total_angle_deg:.3f} deg")
        print(f"Average angular velocity while turning: {average_turning_velocity:.3f} deg/s")
        print(f"Maximum angular velocity: {max_angular_velocity:.3f} deg/s")

        print("Rotation segments:")
        for row in results:
            segment_name = rotation_segment_names[row["rotation_segment"] - 1]

            print(
                f"  {row['rotation_segment']}: {segment_name}, "
                f"frames {row['frame_start']} -> {row['frame_end']}, "
                f"angle={row['angle_change_deg']:.3f} deg, duration={row['duration_s']:.4f} s, "
                f"angular velocity={row['angular_velocity_deg_per_s']:.3f} deg/s "
                f"({row['angular_velocity_rad_per_s']:.3f} rad/s)"
            )

        print()
        
        
def _plot_by_segment(
    path: Path,
    rows: list[dict],
    segment_key: str,
    value_key: str,
    segment_names: list[str],
    title: str,
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for segment_idx, segment_name in enumerate(segment_names, start=1):
        segment_rows = [row for row in rows if row[segment_key] == segment_idx]
        fps = [row["fps"] for row in segment_rows]
        values = [row[value_key] for row in segment_rows]
        plt.plot(fps, values, marker="o", linewidth=1.0, markersize=5, label=segment_name)

    plt.xlabel("Blender render rate (FPS)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axvline(x=240, linestyle="--", linewidth=1.0, label="240 FPS")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

        
def save_speed_scatter(
    path: Path,
    all_results: list[dict],
    path_name: str,
    segment_names: list[str],
) -> None:
    """
    Plot render FPS versus camera translation speed, one curve represents per path segment.
    """
    _plot_by_segment(
        path=path,
        rows=all_results,
        segment_key="segment",
        value_key="speed_m_per_s",
        segment_names=segment_names,
        title=f"Camera translation speed: {path_name}",
        ylabel="Camera translation speed (m/s)"
    )
    print(f"Saved speed plot: {path.resolve()}")
    
    
def save_angular_velocity_plot(
    path: Path,
    all_rotation_results: list[dict],
    path_name: str,
    rotation_segment_names: list[str],
) -> None:
    """
    Plot Blender FPS versus average camera angular velocity
    for each rotation-keyframe segment.
    """
    if not all_rotation_results:
        return

    _plot_by_segment(
        path=path,
        rows=all_rotation_results,
        segment_key="rotation_segment",
        value_key="angular_velocity_deg_per_s",
        segment_names=rotation_segment_names,
        title=f"Camera angular velocity: {path_name}",
        ylabel="Camera angular velocity (deg/s)",
    )

    print(f"Saved angular-velocity plot: {path.resolve()}")


def save_position_vs_time_plot(
    path: Path,
    trace: dict[str, np.ndarray],
    path_name: str,
) -> None:
    """
    Plot camera x, y, and z position versus elapsed scene time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    time_ms = trace["time_s"] * 1e3

    plt.figure(figsize=(10, 6))

    plt.plot(
        time_ms,
        trace["position_x_m"],
        label="x position",
        linewidth=1.5,
    )
    plt.plot(
        time_ms,
        trace["position_y_m"],
        label="y position",
        linewidth=1.5,
    )
    plt.plot(
        time_ms,
        trace["position_z_m"],
        label="z position",
        linewidth=1.5,
    )

    plt.xlabel("Simulation time (ms)")
    plt.ylabel("Camera position (m)")
    plt.title(f"Camera position versus time: {path_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved position trace: {path.resolve()}")
    
    
def save_velocity_vs_time_plot(
    path: Path,
    trace: dict[str, np.ndarray],
    path_name: str,
) -> None:
    """
    Plot camera translation speed versus elapsed scene time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    time_ms = trace["time_s"] * 1e3

    plt.figure(figsize=(10, 6))

    plt.plot(
        time_ms,
        trace["speed_m_per_s"],
        linewidth=1.5,
    )

    plt.xlabel("Simulation time (ms)")
    plt.ylabel("Camera translation speed (m/s)")
    plt.title(f"Camera speed versus time: {path_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved velocity trace: {path.resolve()}")
    
    
def save_velocity_components_vs_time_plot(
    path: Path,
    trace: dict[str, np.ndarray],
    path_name: str,
) -> None:
    """
    Plot x, y, and z translation velocity versus scene time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    time_ms = trace["time_s"] * 1e3

    plt.figure(figsize=(10, 6))

    plt.plot(
        time_ms,
        trace["velocity_x_m_per_s"],
        label=r"$v_x$",
        linewidth=1.5,
    )
    plt.plot(
        time_ms,
        trace["velocity_y_m_per_s"],
        label=r"$v_y$",
        linewidth=1.5,
    )
    plt.plot(
        time_ms,
        trace["velocity_z_m_per_s"],
        label=r"$v_z$",
        linewidth=1.5,
    )

    plt.xlabel("Simulation time (ms)")
    plt.ylabel("Velocity component (m/s)")
    plt.title(f"Camera velocity components: {path_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved velocity-component trace: {path.resolve()}")
    
    
def save_position_3d_plot(
    path: Path,
    trace: dict[str, np.ndarray],
    path_name: str,
) -> None:
    """
    Plot the camera translation path in Blender coordinates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")

    axis.plot(
        trace["position_x_m"],
        trace["position_y_m"],
        trace["position_z_m"],
        linewidth=1.5,
    )

    axis.scatter(
        trace["position_x_m"][0],
        trace["position_y_m"][0],
        trace["position_z_m"][0],
        marker="o",
        s=50,
        label="Start",
    )

    axis.scatter(
        trace["position_x_m"][-1],
        trace["position_y_m"][-1],
        trace["position_z_m"][-1],
        marker="x",
        s=60,
        label="End",
    )

    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_zlabel("z position (m)")
    axis.set_title(f"Camera position trace: {path_name}")
    axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved 3D position trace: {path.resolve()}")


def main() -> None:
    args = parse_args()

    path_config_path = resolve_path_config(args.path_config)
    path_config = load_path_config(path_config_path)

    path_name = path_config["name"]
    camera_path = path_config["camera_path"]
    segment_names = path_config["segment_names"]
    camera_rotation = path_config["camera_rotation"]
    rotation_segment_names = path_config["rotation_segment_names"]

    fps_values = resolve_fps_values(args)
    
    trace_fps = 240.0 if 240.0 in fps_values else max(fps_values)

    trace = sample_position_trajectory(
        fps=trace_fps,
        camera_path=camera_path,
        samples_per_segment=args.trace_samples_per_segment,
    )

    all_results = []
    all_rotation_results = []

    for fps in fps_values:
        all_results.extend(calculate_segment_results(fps=fps, camera_path=camera_path))

        if camera_rotation:
            all_rotation_results.extend(
                calculate_rotation_results(fps=fps, camera_rotation=camera_rotation,)
            )

    print(f"Path: {path_name}")
    print()

    print_results(fps_values=fps_values, all_results=all_results, camera_path=camera_path)
    print_rotation_results(
        fps_values=fps_values,
        all_rotation_results=all_rotation_results,
        rotation_segment_names=rotation_segment_names,
    )

    save_speed_scatter(
        path=Path(f"{path_name}_speed_vs_fps.png"),
        all_results=all_results,
        path_name=path_name,
        segment_names=segment_names,
    )
    
    save_angular_velocity_plot(
        path=Path(
            f"{path_name}_angular_velocity_vs_fps.png"
        ),
        all_rotation_results=all_rotation_results,
        path_name=path_name,
        rotation_segment_names=rotation_segment_names,
    )
    
    trace_output_dir = Path("outputs") / "drone_path_analysis" / path_name

    save_position_vs_time_plot(
        path=trace_output_dir / f"{path_name}_position_vs_time.png",
        trace=trace,
        path_name=path_name,
    )

    save_velocity_vs_time_plot(
        path=trace_output_dir / f"{path_name}_speed_vs_time.png",
        trace=trace,
        path_name=path_name,
    )

    save_velocity_components_vs_time_plot(
        path=trace_output_dir / f"{path_name}_velocity_components_vs_time.png",
        trace=trace,
        path_name=path_name,
    )

    save_position_3d_plot(
        path=trace_output_dir / f"{path_name}_position_trace_3d.png",
        trace=trace,
        path_name=path_name,
    )


if __name__ == "__main__":
    main()
