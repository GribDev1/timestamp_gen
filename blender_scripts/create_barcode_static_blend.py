r"""
Create a static line of planes to test spacial resolution of the ToF sensor.

The camera remains stationary while viewing foreground bars placed in front
of a flat background wall. Four horizontal barcode bands use progressively
smaller bar and gap widths.

This test is intended to examine:
    1. Spatial resolution
    2. Foreground/background depth separation
    3. Mixed-surface timestamp returns
    4. Histogram peak separation

Run:
    blender --background --python .\blender_scripts\create_barcode_static_blend.py
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import scene_builder as sb


def add_barcode_band(
        name,
        y_center,
        band_height,
        bar_width,
        gap_width,
        x_min,
        x_max,
        bar_depth,
        bar_thickness,
        color,
):
    """
    Create one horizontal barcode band.

    Foreground cubes form the bars. Empty spaces between cubes reveal the
    background wall behind them.
    """
    x_position = x_min
    bar_index = 0

    while x_position < x_max:
        remaining_width = x_max - x_position
        current_width = min(bar_width, remaining_width)

        if current_width <= 0:
            break

        bar_center_x = x_position + current_width / 2.0

        sb.add_cube(
            sb.CubeConfig(
                name=f"{name}_Bar_{bar_index:03d}",
                size=(
                    current_width,
                    band_height,
                    bar_thickness,
                ),
                location=(
                    bar_center_x,
                    y_center,
                    bar_depth,
                ),
                rotation_deg=(0.0, 0.0, 0.0),
                color=color,
            )
        )

        x_position += bar_width + gap_width
        bar_index += 1



def main():
    render = sb.RenderConfig(
        width=160,
        height=160,
        fps=240,
        frame_start=1,
        frame_end=10,
    )


    sb.clear_scene()
    sb.set_scene_settings(render)

    camera = sb.add_camera(
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        lens=35.0,
    )

    sb.set_world_background(color=(0.8, 0.8, 0.8), strength=0.8)

    # Lights
    sb.add_light(
        location=(0.0, 2.5, -1.0),
        energy=700,
        size=6.0,
    )

    sb.add_light(
        location=(-3.0, 1.5, -3.0),
        energy=350,
        size=4.0,
    )

    sb.add_light(
        location=(3.0, 1.5, -3.0),
        energy=350,
        size=4.0,
    )

    # Background wall at 2.5 m
    sb.add_plane(
        sb.PlaneConfig(
            name="Background Wall",
            size=6.0,
            location=(0.0, 0.0, -2.5),
            rotation_deg=(0.0, 0.0, 0.0),
            color=(0.75, 0.75, 0.75, 1.0),
        )
    )

    x_min = -1.2
    x_max = 1.2

    bar_depth = -2.0
    bar_thickness = 0.03
    band_height = 0.42

    # Band 1: Widest bars
    add_barcode_band(
        name="Barcode_Wide",
        y_center=0.78,
        band_height=band_height,
        bar_width=0.30,
        gap_width=0.30,
        x_min=x_min,
        x_max=x_max,
        bar_depth=bar_depth,
        bar_thickness=bar_thickness,
        color=(0.20, 0.20, 0.20, 1.0),
    )

    # Band 2: Medium bars
    add_barcode_band(
        name="Barcode_Medium",
        y_center=0.26,
        band_height=band_height,
        bar_width=0.20,
        gap_width=0.20,
        x_min=x_min,
        x_max=x_max,
        bar_depth=bar_depth,
        bar_thickness=bar_thickness,
        color=(0.25, 0.25, 0.25, 1.0),
    )

    # Band 3: Narrow bars
    add_barcode_band(
        name="Barcode_Narrow",
        y_center=-0.26,
        band_height=band_height,
        bar_width=0.10,
        gap_width=0.10,
        x_min=x_min,
        x_max=x_max,
        bar_depth=bar_depth,
        bar_thickness=bar_thickness,
        color=(0.30, 0.30, 0.30, 1.0),
    )

    # Band 4: Narrowest bars
    add_barcode_band(
        name="Barcode_Very_Narrow",
        y_center=-0.78,
        band_height=band_height,
        bar_width=0.05,
        gap_width=0.05,
        x_min=x_min,
        x_max=x_max,
        bar_depth=bar_depth,
        bar_thickness=bar_thickness,
        color=(0.35, 0.35, 0.35, 1.0),
    )

    sb.save_blend("barcode_static.blend")

if __name__ == "__main__":
    main()