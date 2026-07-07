"""

    To Run:
        blender --background --python blender_tests/create_occlusion_blend.py
"""

from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from scene_builder import RenderConfig, PlaneConfig, build_basic_scene

render = RenderConfig(frame_start=1, frame_end=30)

foreground = PlaneConfig(
    name="Foreground Plane",
    size=3.0,
    location=(1.5, 0.0, -4.0),
    color=(0.8, 0.8, 0.8, 1.0),
)

background = PlaneConfig(
    name="Background Plane",
    size=12.0,
    location=(0.0, 0.0, -8.0),
    color=(0.5, 0.5, 0.5, 1.0),
)

build_basic_scene(
    output_path="test_occlusion.blend",
    render_config=render,
    foreground_plane=foreground,
    foreground_motion=None,
    background_plane=background,
)