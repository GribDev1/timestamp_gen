"""
Create a simple empty-room microdrone scene.

The camera represents a drone flying through an empty rectangular room.
The path:
    1. Moves along one wall.
    2. Turns and moves along a second wall.
    3. Turns 135 degrees in the opposite direction.
    4. Moves diagonally across the room.
    5. Moves along the other two walls.

Run:
    blender --background --python .\blender_scripts\create_drone_wall_approach_blend.py
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import scene_builder as sb


def main():
    render = sb.RenderConfig(
        width=320,
        height=160,
        fps=240,
        frame_start=1,
        frame_end=300,
    )

    sb.clear_scene()
    sb.set_scene_settings(render)

    camera = sb.add_camera(
        location=(-3.0, 0.0, 3.0),
        rotation=(0.0, 0.0, 0.0),
        lens=24,
    )

    sb.add_light(location=(0.0, 0.0, 2.0), energy=800, size=8.0)
    sb.add_light(location=(-3.0, 0.0, 1.0), energy=400, size=5.0)
    sb.add_light(location=(3.0, 0.0, 1.0), energy=400, size=5.0)
    
    sb.set_world_background(color=(0.8, 0.8, 0.8), strength=0.8)

    # Back wall
    sb.add_cube(
        sb.CubeConfig(
            name="Back Wall",
            size=(10.0, 3.0, 0.1),
            location=(0.0, 0.0, 5.0),
            color=(0.70, 0.70, 0.70, 1.0),
        )
    )
    
    # Front wall
    sb.add_cube(
        sb.CubeConfig(
            name="Front Wall",
            size=(10.0, 3.0, 0.1),
            location=(0.0, 0.0, -5.0),
            color=(0.70, 0.70, 0.70, 1.0),
        )
    )

    # Left wall
    sb.add_cube(
        sb.CubeConfig(
            name="Left Wall",
            size=(0.1, 3.0, 10.0),
            location=(-5.0, 0.0, 0.0),
            color=(0.60, 0.60, 0.60, 1.0),
        )
    )

    # Right wall
    sb.add_cube(
        sb.CubeConfig(
            name="Right Wall",
            size=(0.1, 3.0, 10.0),
            location=(5.0, 0.0, 0.0),
            color=(0.60, 0.60, 0.60, 1.0),
        )
    )

    # Floor
    sb.add_cube(
        sb.CubeConfig(
            name="Floor",
            size=(10.0, 0.1, 10.0),
            location=(0.0, -1.5, 0.0),
            color=(0.45, 0.45, 0.45, 1.0),
        )
    )

    # Ceiling
    sb.add_cube(
        sb.CubeConfig(
            name="Ceiling",
            size=(10.0, 0.1, 10.0),
            location=(0.0, 1.5, 0.0),
            color=(0.50, 0.50, 0.50, 1.0),
        )
    )

    # -------------------------
    # Drone path
    # -------------------------
    # Room coordinates:
    #   x = -5 left wall
    #   x = +5 right wall
    #   z = -5 front wall
    #   z = +5 back wall
    #
    # The drone stays roughly 2 m away from each wall:
    #   x = -3 or +3
    #   z = -3 or +3
    sb.animate_camera_path(
        camera,
        frame_locations=[
            # Along wall 1: near left wall, moving toward front wall.
            (1,   (-3.0, 0.0,  3.0)),
            (60,  (-3.0, 0.0, -3.0)),

            # Along wall 2: near front wall, moving toward right wall.
            (120, ( 3.0, 0.0, -3.0)),

            # 135-degree opposite-direction turn, then diagonal across room.
            (180, (-3.0, 0.0,  3.0)),

            # Along wall 3: near back wall, moving toward right wall.
            (240, ( 3.0, 0.0,  3.0)),

            # Along wall 4: near right wall, moving toward front wall.
            (300, ( 3.0, 0.0, -3.0)),
        ],
        interpolation="LINEAR",
    )

    # -------------------------
    # Camera orientation
    # -------------------------
    # Blender cameras look along local -Z.
    #
    # Approximate yaw convention using rotation around Y:
    #   0 deg:    look toward -Z
    #   -90 deg:  look toward +X
    #   -225 deg: look diagonally toward -X/+Z
    #   -90 deg:  look toward +X
    #   0 deg:    look toward -Z
    #
    # This intentionally creates an opposite-direction 135-degree turn
    # when going from wall 2 into the diagonal crossing.
    sb.animate_camera_rotation(
        camera,
        frame_rotations_deg=[
            # Move along wall 1, looking toward front wall.
            (1,   (0.0,    0.0, 0.0)),
            (60,  (0.0,    0.0, 0.0)),

            # Turn right and move along wall 2.
            (75,  (0.0,  -90.0, 0.0)),
            (120, (0.0,  -90.0, 0.0)),

            # Opposite-direction 135 degree turn into diagonal.
            # From -90 to -225 is a -135 deg turn.
            (140, (0.0, -225.0, 0.0)),
            (180, (0.0, -225.0, 0.0)),

            # Turn back in the opposite direction to follow back wall.
            # From -225 to -90 is +135 deg.
            (200, (0.0,  -90.0, 0.0)),
            (240, (0.0,  -90.0, 0.0)),

            # Turn to follow right wall.
            (260, (0.0,    0.0, 0.0)),
            (300, (0.0,    0.0, 0.0)),
        ],
        interpolation="LINEAR",
    )

    sb.save_blend("drone_wall_approach.blend")


if __name__ == "__main__":
    main()