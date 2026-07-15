r"""
Create a simple obstacle course for a mini-drone.

The camera represents a drone flying around obstacles in a limited area.
The obstacles:
    1. Plane
    2. Cube
    3. Cylinder
    4. Sphere

Run:
    blender --background --python .\blender_scripts\create_drone_flyby_blend.py
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
        frame_end=240,
    )

    sb.clear_scene()
    sb.set_scene_settings(render)

    camera = sb.add_camera(
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        lens=24,
    )

    sb.set_world_background(color=(0.8, 0.8, 0.8), strength=0.8)

    # Lights
    sb.add_light(location=(0.0, 2.0, -4.0), energy=700, size=7.0)
    sb.add_light(location=(-3.0, 2.0, -8.0), energy=400, size=5.0)
    sb.add_light(location=(3.0, 2.0, -8.0), energy=400, size=5.0)

    # Ground
    sb.add_plane(
        sb.PlaneConfig(
            name="Ground",
            size=22.0,
            location=(0.0, -1.2, -5.0),
            rotation_deg=(90.0, 0.0, 0.0),
            color=(0.45, 0.45, 0.45, 1.0),
        )
    )

    # Background wall kept within 10 m range
    sb.add_plane(
        sb.PlaneConfig(
            name="Background Wall",
            size=12.0,
            location=(0.0, 0.0, -9.5),
            rotation_deg=(0.0, 0.0, 0.0),
            color=(0.65, 0.65, 0.65, 1.0),
        )
    )

    # 1. Plane obstacle
    sb.add_plane(
        sb.PlaneConfig(
            name="Plane",
            size=1.3,
            location=(-0.45, 0.0, -2.0),
            rotation_deg=(0.0, 20.0, 0.0),
            color=(0.75, 0.75, 0.75, 1.0),
        )
    )

    # 2. Cube obstacle
    sb.add_cube(
        sb.CubeConfig(
            name="Cube",
            size=(0.8, 0.8, 0.8),
            location=(0.45, -0.1, -3.6),
            rotation_deg=(0.0, 20.0, 0.0),
            color=(0.85, 0.85, 0.85, 1.0),
        )
    )

    # 3. Cylinder obstacle
    sb.add_cylinder(
        sb.CylinderConfig(
            name="Cylinder",
            radius=0.18,
            depth=2.0,
            location=(-0.40, 0.0, -5.2),
            rotation_deg=(90.0, 0.0, 0.0),
            color=(0.90, 0.90, 0.90, 1.0),
        )
    )

    # 4. Sphere obstacle
    sb.add_sphere(
        sb.SphereConfig(
            name="Sphere",
            radius=0.45,
            location=(0.40, 0.0, -6.8),
            color=(0.88, 0.88, 0.88, 1.0),
        )
    )

    # Drone path
    
    frame_locations = [
        (1,   (0.0,   0.0,  0.0)),
        (45,  (0.65,  0.0, -1.8)),
        (90,  (-0.65, 0.0, -3.2)),
        (135, (0.60,  0.0, -4.6)),
        (180, (-0.60, 0.0, -6.0)),
        (240, (0.0,   0.0, -7.4)),
    ]

    segment_names = [
        "Approach plane",
        "Plane to cube",
        "Cube to cylinder",
        "Cylinder to sphere",
        "Sphere exit",
    ]

    sb.animate_camera_path(
        camera,
        frame_locations=frame_locations,
        interpolation="LINEAR",
    )

    sb.animate_camera_rotation(
        camera,
        frame_rotations_deg=[
            (1,   (0.0,   0.0, 0.0)),
            (45,  (0.0,  16.0, 0.0)),
            (90,  (0.0, -16.0, 0.0)),
            (135, (0.0,  14.0, 0.0)),
            (180, (0.0, -14.0, 0.0)),
            (240, (0.0,   0.0, 0.0)),
        ],
        interpolation="LINEAR",
    )
    
    sb.save_drone_path_config(
        name="drone_flyby",
        frame_locations=frame_locations,
        segment_names=segment_names,
        description=(
            "Microdrone flyby path through a plane, cube, "
            "cylinder, and sphere obstacle course."
        ),
    )

    sb.save_blend("drone_flyby.blend")


if __name__ == "__main__":
    main()