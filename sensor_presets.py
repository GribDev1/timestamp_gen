"""
Sensor preset loading for timestamp generation.

Sensor settings are stored as YAML files in configs/sensors/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "sensors"


@dataclass(frozen=True)
class ToFSensor:
    name: str

    tof_h: int
    tof_w: int

    laser_rate_hz: float
    block_size_L: int
    detection_probability_rho: float
    timing_jitter_std_s: float

    min_valid_depth_m: float
    max_valid_depth_m: float

    camera_fov_x_deg: float
    camera_fov_y_deg: float

    c_light: float = 299_792_458.0

    def depth_to_timestamp(self, depth_m):
        return 2.0 * depth_m / self.c_light

    def timestamp_to_depth(self, timestamp_s):
        return 0.5 * self.c_light * timestamp_s

    def build_ray_cos_map(self) -> np.ndarray:
        """
        Build a [tof_h, tof_w] map of cos(theta) for each ToF pixel.
        """
        fov_x = np.deg2rad(self.camera_fov_x_deg)
        fov_y = np.deg2rad(self.camera_fov_y_deg)

        xs = ((np.arange(self.tof_w) + 0.5) / self.tof_w) * 2.0 - 1.0
        ys = ((np.arange(self.tof_h) + 0.5) / self.tof_h) * 2.0 - 1.0

        nx, ny = np.meshgrid(xs, ys)

        ray_x = nx * np.tan(fov_x / 2.0)
        ray_y = ny * np.tan(fov_y / 2.0)
        ray_z = np.ones_like(ray_x)

        ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)

        return (ray_z / ray_norm).astype(np.float32)

    def build_fullres_ray_dirs(self, image_h: int, image_w: int) -> np.ndarray:
        """
        Build normalized ray directions for every full-resolution rendered pixel.
        Shape: [image_h, image_w, 3]
        """
        fov_x = np.deg2rad(self.camera_fov_x_deg)
        fov_y = np.deg2rad(self.camera_fov_y_deg)

        xs = ((np.arange(image_w) + 0.5) / image_w) * 2.0 - 1.0
        ys = ((np.arange(image_h) + 0.5) / image_h) * 2.0 - 1.0

        nx, ny = np.meshgrid(xs, ys)

        ray_x = nx * np.tan(fov_x / 2.0)
        ray_y = ny * np.tan(fov_y / 2.0)
        ray_z = np.ones_like(ray_x)

        rays = np.stack([ray_x, ray_y, ray_z], axis=-1)
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

        return rays.astype(np.float32)


def load_sensor_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Sensor YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Sensor YAML file is empty: {path}")

    return config


def sensor_from_yaml_config(config: dict) -> ToFSensor:
    camera_config = config["camera"]

    if "fov_x_deg" in camera_config and "fov_y_deg" in camera_config:
        camera_fov_x_deg = float(camera_config["fov_x_deg"])
        camera_fov_y_deg = float(camera_config["fov_y_deg"])

    else:
        focal_length_mm = float(camera_config["focal_length_mm"])
        sensor_width_mm = float(camera_config["sensor_width_mm"])

        if "sensor_height_mm" in camera_config:
            sensor_height_mm = float(camera_config["sensor_height_mm"])
        else:
            tof_h = int(config["tof"]["height"])
            tof_w = int(config["tof"]["width"])
            sensor_height_mm = sensor_width_mm * (tof_h / tof_w)

        camera_fov_x_deg = 2.0 * np.rad2deg(
            np.arctan(sensor_width_mm / (2.0 * focal_length_mm))
        )

        camera_fov_y_deg = 2.0 * np.rad2deg(
            np.arctan(sensor_height_mm / (2.0 * focal_length_mm))
        )

    return ToFSensor(
        name=config["name"],

        tof_h=int(config["tof"]["height"]),
        tof_w=int(config["tof"]["width"]),

        laser_rate_hz=float(config["timing"]["laser_rate_hz"]),
        block_size_L=int(config["timing"]["block_size_L"]),
        timing_jitter_std_s=float(config["timing"]["timing_jitter_std_s"]),

        detection_probability_rho=float(
            config["detection"]["detection_probability_rho"]
        ),

        min_valid_depth_m=float(config["depth"]["min_valid_depth_m"]),
        max_valid_depth_m=float(config["depth"]["max_valid_depth_m"]),

        camera_fov_x_deg=float(camera_fov_x_deg),
        camera_fov_y_deg=float(camera_fov_y_deg),

        c_light=float(config.get("constants", {}).get("c_light", 299_792_458.0)),
    )


def get_sensor_preset(name: str) -> ToFSensor:
    """
    Load a named sensor preset from configs/sensors/{name}.yaml.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    config = load_sensor_yaml(path)
    return sensor_from_yaml_config(config)


def list_sensor_presets() -> list[str]:
    """
    Return available sensor preset names.
    """
    if not CONFIG_DIR.exists():
        return []

    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))