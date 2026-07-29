r"""
Create a temporary .blend file with a restricted animation frame range.

Blender usage:

    blender --background source.blend--python create_segment_blend.py -- output.blend 1 75
"""

from pathlib import Path
import sys

import bpy


def get_script_args() -> list[str]:
    """
    Return arguments placed after Blender's `--` separator.
    """
    if "--" not in sys.argv:
        raise RuntimeError(
            "Missing Blender argument separator '--'. "
            "Expected: -- output.blend first_frame last_frame"
        )

    separator_index = sys.argv.index("--")
    return sys.argv[separator_index + 1 :]


def main() -> None:
    args = get_script_args()

    if len(args) != 3:
        raise RuntimeError(
            "Usage: -- <output.blend> <first_frame> <last_frame>"
        )

    output_path = Path(args[0]).expanduser().resolve()
    first_frame = int(args[1])
    last_frame = int(args[2])

    if first_frame < 1:
        raise ValueError("first_frame must be at least 1.")

    if last_frame < first_frame:
        raise ValueError("last_frame must be greater than or equal to first_frame.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.frame_start = first_frame
    scene.frame_end = last_frame

    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

    print(f"Saved segmented blend file: {output_path}")
    print(f"Frame range: {first_frame} through {last_frame}")


if __name__ == "__main__":
    main()