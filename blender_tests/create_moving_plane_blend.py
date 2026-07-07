"""
Creates a simple Blender test scene for timestamp simulation:
- One camera at the origin looking along -Z
- One flat plane in front of the camera
- The plane moves toward the camera over time
- A large background plane is optional
- Saves a .blend file that can be rendered with VisionSIM

Run from command line, for example:
    blender --background --python create_moving_plane_blend.py
"""

from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import bpy
from scene_builder import RenderConfig, PlaneConfig, build_basic_scene   


render = RenderConfig(
    width=320,
    height=160,
    fps=240,
    frame_start=1,
    frame_end=30,
)

foreground = PlaneConfig(
    name="Moving Flat Plane",
    size=9,
    location=(0.0, 0.0, -6.0),
    rotation_deg=(0.0, 0.0, 0.0),
    color=(0.8, 0.8, 0.8, 1.0),
)

motion = {
    "keyframes": [
        (1, (0.0, 0.0, -6.0)),
        (30, (0.0, 0.0, -4.0)),
    ],
    "interpolation": "LINEAR",
}

build_basic_scene(
    output_path="test_flat_moving.blend",
    render_config=render,
    foreground_plane=foreground,
    foreground_motion=motion,
    background_plane=None,
)