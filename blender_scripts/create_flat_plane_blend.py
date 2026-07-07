"""

    To Run:
        blender --background --python blender_tests/create_flat_plane_blend.py
"""

from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from scene_builder import RenderConfig, PlaneConfig, build_basic_scene

render = RenderConfig(frame_start=1, frame_end=30)

foreground = PlaneConfig(
    name="Flat Stationary Plane",
    size=7.5,
    location=(0.0, 0.0, -5.0),
)

build_basic_scene(
    output_path="test_flat_stationary.blend",
    render_config=render,
    foreground_plane=foreground,
    foreground_motion=None,
)