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
    
    
    frame_locations = [
        (1,   (-3.0, 0.0,  3.0)),
        (60,  (-3.0, 0.0, -3.0)),
        (120, ( 3.0, 0.0, -3.0)),
        (180, (-3.0, 0.0,  3.0)),
        (240, ( 3.0, 0.0,  3.0)),
        (300, ( 3.0, 0.0, -3.0)),
    ]

    translation_segment_names = [
        "Left wall toward front",
        "Front wall toward right",
        "Diagonal right-front to left-back",
        "Back wall toward right",
        "Right wall toward front",
    ]
    
    frame_rotations = [
        (1, (0.0, 0.0, 0.0)),
        (60, (0.0, 0.0, 0.0)),
        (75, (0.0, -90.0, 0.0)),
        (120, (0.0, -90.0, 0.0)),
        (140, (0.0, -225.0, 0.0)),
        (180, (0.0, -225.0, 0.0)),
        (200, (0.0, -90.0, 0.0)),
        (240, (0.0, -90.0, 0.0)),
        (260, (0.0, 0.0, 0.0)),
        (300, (0.0, 0.0, 0.0)),
    ]
    
    rotation_segment_names = [
        "Hold initial heading",
        "Turn toward right wall",
        "Hold right-wall heading",
        "Turn toward rear diagonal",
        "Hold rear-diagonal heading",
        "Turn toward right wall",
        "Hold right-wall heading",
        "Turn toward front",
        "Hold final heading",
    ]
    
    # Drone path   
    sb.animate_camera_path(
        camera,
        frame_locations,
    )

    # Camera orientation
    sb.animate_camera_rotation(
        camera,
        frame_rotations,
    )
    
    sb.save_drone_path_config(
        name="drone_wall_approach",
        frame_locations=frame_locations,
        segment_names=translation_segment_names,
        frame_rotations=frame_rotations,
        rotation_segment_names=rotation_segment_names,
        description=(
            "Microdrone camera path through a rectangular room, "
            "including translation and orientation keyframes."
        ),
    )

    sb.save_blend("drone_wall_approach.blend")


if __name__ == "__main__":
    main()