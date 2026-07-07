"""
Create Blender scene for proof 1:

Two flat surfaces with full/equal reflectivity.

Goal:
    Center ToF pixel sees:
        frames 1  -> 10 : foreground translation
        frames 10 -> 20 : occlusion / visibility switch
        frames 20 -> 30 : background translation

To Run:
    blender --background --python tests\blender_tests\create_two_flat_reflective_blend.py
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

    # Large flat background.
    # This should always cover the camera view.
    background = add_plane(
        PlaneConfig(
            name="Background Plane",
            size=12.0,
            location=(0.0, 0.0, -4.00),
            rotation_deg=(0.0, 0.0, 0.0),
            color=(1.0, 1.0, 1.0, 1.0),
        )
    )

    # Smaller flat foreground.
    # Starts centered in the view, then moves sideways out of the center pixel.
    foreground = add_plane(
        PlaneConfig(
            name="Foreground Plane",
            size=2.0,
            location=(0.0, 0.0, -3.80),
            rotation_deg=(0.0, 0.0, 0.0),
            color=(1.0, 1.0, 1.0, 1.0),
        )
    )

    # Foreground:
    # frame 1 -> 10:
    #     same surface remains visible at center, moving in depth.
    #
    # frame 10 -> 20:
    #     foreground shifts sideways out of the center pixel.
    #
    # frame 20 -> 30:
    #     foreground stays out of center.
    animate_location(
        foreground,
        frame_locations=[
            (1,  (0.0, 0.0, -3.80)),
            (10, (0.0, 0.0, -3.68)),
            (20, (3.0, 0.0, -3.68)),
            (30, (3.0, 0.0, -3.68)),
        ],
        interpolation="LINEAR",
    )

    # Background:
    # frame 1 -> 20:
    #     background stays fixed behind foreground.
    #
    # frame 20 -> 30:
    #     after it becomes visible at center, it translates in depth.
    animate_location(
        background,
        frame_locations=[
            (1,  (0.0, 0.0, -4.00)),
            (20, (0.0, 0.0, -4.00)),
            (30, (0.0, 0.0, -3.88)),
        ],
        interpolation="LINEAR",
    )

    save_blend("test_two_flat_reflective.blend")


if __name__ == "__main__":
    main()