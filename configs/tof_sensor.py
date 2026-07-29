from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ToFSensor:
    """
    Basic single-photon ToF sensor model.
    
    Stores sensor geometry, timestamp simulation parameters,
    and helper functions for converting between depth and timestamp.
    """
    
    name: str = "generic_spad_tof"
    source: str = "simulation default"
    notes: str = ""
    
    tof_h: int = 32
    tof_w: int = 64
    
    focal_length_mm: float = 24.0
    sensor_width_mm: float = 36.0
    
    wavelength_nm: float | None = None
    
    min_valid_depth_m: float = 0.01
    max_valid_depth_m: float = 20.0
    
    laser_rate_hz: float = 10e6
    block_size_L: int = 256
    detection_probability_rho: float = 0.05
    timing_jitter_std_s: float = 50e-12
    
    c_light: float = 299_792_458.0
    
    
    def __post_init__(self):
        self.validate()
        
        self.sensor_height_mm = self.sensor_width_mm * (self.tof_h / self.tof_w)
        
        self.camera_fov_x_deg = np.degrees(
            2.0 * np.arctan(self.sensor_width_mm / (2.0 * self.focal_length_mm))
        )
        
        self.camera_fov_y_deg = np.degrees(
            2.0 * np.arctan(self.sensor_height_mm / (2.0 * self.focal_length_mm))
        )
        
        
    def validate(self):
        """
        Check that sensor parameters are physically and numerically reasonable.
        """
        if self.tof_h <= 0 or self.tof_w <= 0:
            raise ValueError("ToF sensor dimensions must be positive.")

        if self.focal_length_mm <= 0:
            raise ValueError("focal_length_mm must be positive.")

        if self.sensor_width_mm <= 0:
            raise ValueError("sensor_width_mm must be positive.")

        if self.min_valid_depth_m <= 0:
            raise ValueError("min_valid_depth_m must be positive.")

        if self.max_valid_depth_m <= self.min_valid_depth_m:
            raise ValueError("max_valid_depth_m must be greater than min_valid_depth_m.")

        if self.laser_rate_hz <= 0:
            raise ValueError("laser_rate_hz must be positive.")

        if self.block_size_L <= 0:
            raise ValueError("block_size_L must be positive.")

        if not (0.0 <= self.detection_probability_rho <= 1.0):
            raise ValueError("detection_probability_rho must be between 0 and 1.")

        if self.timing_jitter_std_s < 0:
            raise ValueError("timing_jitter_std_s must be nonnegative.")
        
        
    @property
    def block_duration_s(self) -> float:
        """
        Time duration of one timestamp block.

        Example:
            L = 256 pulses
            laser_rate = 10 MHz

            block_duration = 256 / 10e6 = 25.6 us
        """
        return self.block_size_L / self.laser_rate_hz


    @property
    def block_rate_hz(self) -> float:
        """
        Number of timestamp blocks per second.
        """
        return self.laser_rate_hz / self.block_size_L
    
    
    def depth_to_timestamp(self, depth_m):
        """
        Convert depth/range in meters to round-trip ToF timestamp in seconds.
        """
        return 2.0 * depth_m / self.c_light
    
    
    def timestamp_to_depth(self, tau_s):
        """
        Convert round-trip ToF timestamp in seconds to depth/range in meters.
        """
        return tau_s * self.c_light / 2.0
    
    
    def valid_depth_mask(self, depth):
        """
        Return a boolean mask for finite depths within the sensor's valid range.
        """
        return (
            np.isfinite(depth)
            & (depth > self.min_valid_depth_m)
            & (depth < self.max_valid_depth_m)
        )
        
        
    def build_ray_cos_map(self):
        """
        Compute cos(theta) for each ToF pixel.

        Center pixels have cos(theta) close to 1.
        Edge and corner pixels have smaller values.
        """
        fov_x = np.deg2rad(self.camera_fov_x_deg)
        fov_y = np.deg2rad(self.camera_fov_y_deg)

        xs = np.arange(self.tof_w, dtype=np.float32)
        ys = np.arange(self.tof_h, dtype=np.float32)

        nx = ((xs + 0.5) / self.tof_w) * 2.0 - 1.0
        ny = ((ys + 0.5) / self.tof_h) * 2.0 - 1.0

        ray_x = nx[np.newaxis, :] * np.tan(fov_x / 2.0)
        ray_y = ny[:, np.newaxis] * np.tan(fov_y / 2.0)
        ray_z = 1.0

        ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)

        return (ray_z / ray_norm).astype(np.float32)
    
    def pixel_ray_cos(self, x: int, y: int) -> float:
        """
        Compute cos(theta) for one ToF pixel.
        """
        fov_x = np.deg2rad(self.camera_fov_x_deg)
        fov_y = np.deg2rad(self.camera_fov_y_deg)

        nx = ((x + 0.5) / self.tof_w) * 2.0 - 1.0
        ny = ((y + 0.5) / self.tof_h) * 2.0 - 1.0

        ray_x = nx * np.tan(fov_x / 2.0)
        ray_y = ny * np.tan(fov_y / 2.0)
        ray_z = 1.0

        ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)

        return float(ray_z / ray_norm)
    
    def normalize_vectors(self, v, eps=1e-12):
        """
        Normalize vectors along the last axis.
        """
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norm, eps)
    
    def build_fullres_ray_dirs(self, image_h: int, image_w: int) -> np.ndarray:
        """
        Build one camera ray direction per rendered high-resolution pixel.

        Output:
            ray_dirs_full: [image_h, image_w, 3]
        """
        sensor_height_mm = self.sensor_width_mm * (image_h / image_w)

        fov_x = 2.0 * np.arctan(self.sensor_width_mm / (2.0 * self.focal_length_mm))
        fov_y = 2.0 * np.arctan(sensor_height_mm / (2.0 * self.focal_length_mm))

        xs = np.arange(image_w, dtype=np.float32)
        ys = np.arange(image_h, dtype=np.float32)

        nx = ((xs + 0.5) / image_w) * 2.0 - 1.0
        ny = ((ys + 0.5) / image_h) * 2.0 - 1.0

        ray_x = nx[np.newaxis, :] * np.tan(fov_x / 2.0)
        ray_y = ny[:, np.newaxis] * np.tan(fov_y / 2.0)
        ray_z = np.ones((image_h, image_w), dtype=np.float32)

        ray_dirs = np.stack(
            [
                np.broadcast_to(ray_x, (image_h, image_w)),
                np.broadcast_to(ray_y, (image_h, image_w)),
                ray_z,
            ],
            axis=-1,
        )
        
        return self.normalize_vectors(ray_dirs).astype(np.float32)
    
    
    def to_metadata_dict(self) -> dict:
        """
        Export sensor settings as a plain dictionary.

        Useful for saving into metadata.json.
        """
        data = asdict(self)
        data["sensor_height_mm"] = self.sensor_height_mm
        data["camera_fov_x_deg"] = self.camera_fov_x_deg
        data["camera_fov_y_deg"] = self.camera_fov_y_deg
        data["block_duration_s"] = self.block_duration_s
        data["block_rate_hz"] = self.block_rate_hz
        return data