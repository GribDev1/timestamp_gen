import csv
from pathlib import Path

from tof_sensor import ToFSensor


SENSOR_CSV_PATH = Path("configs/tof_sensors.csv")


def _optional_float(value: str):
    """
    Convert a CSV string to float, allowing blank values.
    """
    value = value.strip()
    
    if value == "":
        return None
    
    return float(value)


def load_sensor_presets(csv_path: str | Path = SENSOR_CSV_PATH) -> dict[str, ToFSensor]:
    """
    Load ToF sensor presets from a CSV file.

    Returns:
        Dictionary mapping sensor name to ToFSensor object.
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find sensor preset CSV: {csv_path}")
    
    presets = {}
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            sensor = ToFSensor(
                name=row["name"],
                source=row["source"],
                notes=row["notes"],
                tof_h=int(row["tof_h"]),
                tof_w=int(row["tof_w"]),
                focal_length_mm=float(row["focal_length_mm"]),
                sensor_width_mm=float(row["sensor_width_mm"]),
                wavelength_nm=_optional_float(row["wavelength_nm"]),
                min_valid_depth_m=float(row["min_valid_depth_m"]),
                max_valid_depth_m=float(row["max_valid_depth_m"]),
                laser_rate_hz=float(row["laser_rate_hz"]),
                block_size_L=int(row["block_size_L"]),
                detection_probability_rho=float(row["detection_probability_rho"]),
                timing_jitter_std_s=float(row["timing_jitter_std_s"]),
            )
            
            if sensor.name in presets:
                raise ValueError(f"Duplicate sensor preset name in CSV: {sensor.name}")
            
            presets[sensor.name] = sensor
            
    return presets


def get_sensor_preset(
    name: str,
    csv_path: str | Path = SENSOR_CSV_PATH,
) -> ToFSensor:
    """
    Load one named ToF sensor preset from the CSV file.
    """
    presets = load_sensor_presets(csv_path)
    
    if name not in presets:
        available = ", ".join(sorted(presets.key()))
        raise ValueError(
            f"Unknown sensor preset '{name}'. Available presets: {available}"
        )
        
    return presets[name]