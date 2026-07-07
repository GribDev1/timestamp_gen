"""
Create Blender scene for proof 3:

Flat background + tilted foreground.

Goal:
    Center ToF pixel sees:
        frames 1  -> 10 : tilted foreground translation
        frames 10 -> 20 : visibility switch / occlusion
        frames 20 -> 30 : flat background translation

To Run:
    blender --background --python tests/blender_tests/create_tilted_foreground_blend.py
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from scene_builder import (
    RenderConfig,
    PlaneConfig,
    clear_scene,
    set_scene_settings,
    add_camera,
    add_light,
    add_plane,
    animate_location,
    save_blend,
)


def main():
    render = RenderConfig(
        width=320,
        height=160,
        fps=240,
        frame_start=1,
        frame_end=30,
    )
    
    clear_scene()
    set_scene_settings(render)
    
    add_camera(
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        lens=24,
    )
    
    add_light(
        location=(0.0, 0.0, 1.0),
        energy=300,
        size=5.0,
    )
    
    background = add_plane(
        PlaneConfig(
            name="Flat Background Plane",
            size=12.0,
            location=(0.0, 0.0, -4.00),
            rotation_deg=(0.0, 0.0, 0.0),
            color=(1.0, 1.0, 1.0, 1.0),
        )
    )
    
    foreground = add_plane(
        PlaneConfig(
            name="Tilted Foreground Plane",
            size=2.0,
            location=(0.0, 0.0, -3.80),
            rotation_deg=(0.0, -20.0, 0.0),
            color=(1.0, 1.0, 1.0, 1.0),
        )
    )
    
    animate_location(
        foreground,
        frame_locations=[
            (1, (0.0, 0.0, -3.80)),
            (10, (0.0, 0.0, -3.50)),
            (20, (3.0, 0.0, -3.50)),
            (30, (3.0, 0.0, -3.50)),
        ],
        interpolation="LINEAR",
    )
    
    animate_location(
        background,
        frame_locations=[
            (1, (0.0, 0.0, -4.00)),
            (20, (0.0, 0.0, -4.00)),
            (30, (0.0, 0.0, -3.88)),
        ],
        interpolation="LINEAR",
    )
    
    save_blend("test_tilted_foreground.blend")
    

if __name__ == "__main__":
    main()