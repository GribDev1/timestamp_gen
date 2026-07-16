from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

@dataclass
class TimestampMetadata:
    tof_h: int
    tof_w: int
    block_size_L: int
    laser_rate_hz: float
    detection_probability_rho: float
    timing_jitter_std_s: float
    fps: float
    dt_s: float
    c_light: float
    min_valid_depth_m: float
    max_valid_depth_m: float
    use_weighted_depth_sampling: bool
    timestamp_units: str = "seconds"
    depth_units: str = "meters"
    missed_detection_value: str = "NaN"


@dataclass
class TimestampBlock:
    frame_number: int
    simulation_time_s: float # End time of this ToF acquisition block
    
    # Ideal geometry-derived data
    sampled_depths_m: np.ndarray # [L, H, W]
    timestamps_clean_s: np.ndarray # [L, H, W]
    
    # Noisy ToF model
    detection_mask: np.ndarray # [L, H, W], bool
    timestamps_noisy_s: np.ndarray # [L, H, W], NaN where no detection
    
    # Optional source tracking
    source_depth_file: str = ""
    source_normal_file: str = ""
    

@dataclass
class TimestampDataset:
    def __init__(self, blocks: list[TimestampBlock], metadata: TimestampMetadata):
        self.blocks = blocks
        self.metadata = metadata
    
    def __len__(self):
        return len(self.blocks)
    
    def get_block(self, frame_i: int, model: str = "noisy") -> np.ndarray:
        block = self.blocks[frame_i]
        
        if model == "clean":
            return block.timestamps_clean_s
        
        if model == "noisy":
            return block.timestamps_noisy_s
        
        raise ValueError("Model must be 'clean' or 'noisy'")
    
    def save(self, output_dir: str | Path):
        """
        Saves dataset as:
            output_dir/
                metadata.json
                frames/
                    frame_000001.npz
                    frame_000002.npz
                    ...
        """
        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(self.metadata), f, indent=4)
            
        # Save each TimestampBlock separately
        for i, block in enumerate(self.blocks):
            frame_path = frames_dir / f"frame_{block.frame_number:06d}.npz"

            np.savez(
                frame_path,
                frame_number=np.array(block.frame_number, dtype=np.int32),
                simulation_time_s=np.array(
                    block.simulation_time_s, 
                    dtype=np.float64
                ),
                sampled_depths_m=block.sampled_depths_m.astype(np.float32),
                timestamps_clean_s=block.timestamps_clean_s.astype(np.float32),
                detection_mask=block.detection_mask.astype(bool),
                timestamps_noisy_s=block.timestamps_noisy_s.astype(np.float32),
                source_depth_file=np.array(block.source_depth_file),
                source_normal_file=np.array(block.source_normal_file),
            )
            
        print(f"Saved timestamp dataset to: {output_dir.resolve()}")
        
    @staticmethod
    def load(input_dir: str | Path):
        """
        Loads a saved timestamp dataset.
        """
        input_dir = Path(input_dir)
        frames_dir = input_dir / "frames"
        
        metadata_path = input_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
            
        metadata = TimestampMetadata(**metadata_dict)
        
        frame_files = sorted(frames_dir.glob("frame_*.npz"))
        if not frame_files:
            raise RuntimeError(f"No frame files found in {frames_dir}")
        
        blocks = []
        
        for frame_path in frame_files:
            data = np.load(frame_path, allow_pickle=True)
            
            block = TimestampBlock(
                frame_number=int(data["frame_number"]),
                simulation_time_s=float(data["simulation_time_s"]),
                sampled_depths_m=data["sampled_depths_m"],
                timestamps_clean_s=data["timestamps_clean_s"],
                detection_mask=data["detection_mask"].astype(bool),
                timestamps_noisy_s=data["timestamps_noisy_s"],
                source_depth_file=str(data["source_depth_file"]),
                source_normal_file=str(data["source_normal_file"]),
            )
            
            blocks.append(block)
            
        return TimestampDataset(blocks=blocks, metadata=metadata)