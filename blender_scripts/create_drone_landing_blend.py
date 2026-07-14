r"""
Create a simple landing pad for a mini-drone.

The camera represents a drone landing on different surfaces:
    1. Flat
    2. Slanted
    3. Bumpy

Run:
    blender --background --python .\blender_scripts\create_drone_landing_blend.py
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import scene_builder as sb
import math


def main():
    render = sb.RenderConfig(
        width=320,
        height=160,
        fps=240,
        frame_start=1,
        frame_end=360,
    )

    sb.clear_scene()
    sb.set_scene_settings(render)

    camera = sb.add_camera(
        location=(0.0, 0.0, 0.0),
        rotation=(math.radians(-90.0), 0.0, 0.0),
        lens=24,
    )

    sb.set_world_background(color=(0.8, 0.8, 0.8), strength=0.8)

    # Lights
    sb.add_light(location=(0.0, 2.0, -4.0), energy=700, size=7.0)
    sb.add_light(location=(-3.0, 2.0, -8.0), energy=400, size=5.0)
    sb.add_light(location=(3.0, 2.0, -8.0), energy=400, size=5.0)
    

    # 1. Flat landing pad
    sb.add_cube(
        sb.CubeConfig(
            name="Flat Landing Pad",
            size=(2.0, 0.08, 1.4),
            location=(0.0, -1.15, -2.4),
            rotation_deg=(0.0, 0.0, 0.0),
            color=(0.75, 0.75, 0.75, 1.0),
        )
    )

    # 2. Slanted landing pad
    sb.add_cube(
        sb.CubeConfig(
            name="Slanted Landing Pad",
            size=(2.0, 0.08, 1.4),
            location=(0.0, -1.12, -5.0),
            rotation_deg=(0.0, 0.0, 12.0),
            color=(0.80, 0.80, 0.80, 1.0),
        )
    )

    # 3. Bumpy landing pad
    sb.add_wavy_pad(
        name="Wavy Landing Pad",
        size_x=2.0,
        size_z=1.4,
        location=(0.0, -1.11, -7.6),
        amplitude=0.06,
        freq_x=5.0,
        freq_z=7.0,
        subdivisions=25,
        color=(0.70, 0.70, 0.70, 1.0),
    )

    # Drone path
    sb.animate_camera_path(
        camera,
        frame_locations=[
            # Flat pad
            (1,   (0.0,  1.6,  -2.4)),   # high above flat pad
            (60,  (0.0,  0.4,  -2.4)),   # descend
            (95,  (0.0, -0.80, -2.4)),   # close hover over flat pad
            (120, (0.0, -0.80, -2.4)),   # hold

            # Rise before moving to next pad
            (145, (0.0,  1.0,  -2.4)),

            # Translate high over to slanted pad
            (175, (0.0,  1.0,  -5.0)),

            # Slanted pad
            (200, (0.0,  0.4,  -5.0)),   # descend
            (225, (0.0, -0.75, -5.0)),   # close hover over slanted pad
            (250, (0.0, -0.75, -5.0)),   # hold

            # Rise before moving to next pad
            (275, (0.0,  1.0,  -5.0)),

            # Translate high over to bumpy pad
            (305, (0.0,  1.0,  -7.6)),

            # Bumpy pad
            (325, (0.0,  0.4,  -7.6)),   # descend
            (345, (0.0, -0.75, -7.6)),   # close hover over bumpy pad
            (360, (0.0, -0.75, -7.6)),   # hold
        ],
        interpolation="LINEAR",
    )

    sb.save_blend("drone_landing.blend")


if __name__ == "__main__":
    main()