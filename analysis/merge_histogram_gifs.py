from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-pixel histogram GIFs into one sensor-grid GIF."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing pixel_yY_xX folders with "
            "histogram_yY_xX.gif files."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the combined GIF.",
    )

    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Sensor grid height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Sensor grid width.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help=(
            "Scale factor applied to each individual GIF frame. "
            "Default: 0.5"
        ),
    )

    parser.add_argument(
        "--duration-ms",
        type=int,
        default=None,
        help=(
            "Optional output frame duration in milliseconds. "
            "Default: use duration from the first GIF."
        ),
    )

    return parser.parse_args()


def pixel_gif_path(
    input_dir: Path,
    pixel_y: int,
    pixel_x: int,
) -> Path:
    return (
        input_dir
        / f"pixel_y{pixel_y}_x{pixel_x}"
        / f"histogram_y{pixel_y}_x{pixel_x}.gif"
    )


def load_gif_frames(path: Path) -> tuple[list[Image.Image], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Pixel GIF not found: {path}")

    frames: list[Image.Image] = []

    with Image.open(path) as image:
        duration_ms = int(image.info.get("duration", 100))

        frame_index = 0

        while True:
            try:
                image.seek(frame_index)
            except EOFError:
                break

            frames.append(image.convert("RGB").copy())
            frame_index += 1

    if not frames:
        raise RuntimeError(f"No frames found in GIF: {path}")

    return frames, duration_ms


def main() -> None:
    args = parse_args()

    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")

    if args.scale <= 0:
        raise ValueError("--scale must be positive.")

    all_pixel_frames: dict[tuple[int, int], list[Image.Image]] = {}

    frame_count: int | None = None
    source_duration_ms: int | None = None
    source_size: tuple[int, int] | None = None

    for pixel_y in range(args.height):
        for pixel_x in range(args.width):
            path = pixel_gif_path(
                args.input_dir,
                pixel_y,
                pixel_x,
            )

            frames, duration_ms = load_gif_frames(path)

            if frame_count is None:
                frame_count = len(frames)
                source_duration_ms = duration_ms
                source_size = frames[0].size
            else:
                if len(frames) != frame_count:
                    raise ValueError(
                        f"{path} has {len(frames)} frames, "
                        f"expected {frame_count}."
                    )

                if frames[0].size != source_size:
                    raise ValueError(
                        f"{path} has frame size {frames[0].size}, "
                        f"expected {source_size}."
                    )

            all_pixel_frames[(pixel_y, pixel_x)] = frames

    assert frame_count is not None
    assert source_duration_ms is not None
    assert source_size is not None

    tile_width = max(1, int(round(source_size[0] * args.scale)))
    tile_height = max(1, int(round(source_size[1] * args.scale)))

    canvas_width = args.width * tile_width
    canvas_height = args.height * tile_height

    combined_frames: list[Image.Image] = []

    for frame_index in range(frame_count):
        canvas = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            "white",
        )

        for pixel_y in range(args.height):
            for pixel_x in range(args.width):
                frame = all_pixel_frames[
                    (pixel_y, pixel_x)
                ][frame_index]

                if frame.size != (tile_width, tile_height):
                    frame = frame.resize(
                        (tile_width, tile_height),
                        Image.Resampling.LANCZOS,
                    )

                left = pixel_x * tile_width
                top = pixel_y * tile_height

                canvas.paste(frame, (left, top))

        combined_frames.append(canvas)

        if frame_index % 25 == 0:
            print(
                f"Combined frame {frame_index + 1:,} "
                f"of {frame_count:,}"
            )

    output_duration_ms = (
        args.duration_ms
        if args.duration_ms is not None
        else source_duration_ms
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_frames[0].save(
        args.output,
        save_all=True,
        append_images=combined_frames[1:],
        duration=output_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    print(f"Combined GIF saved to: {args.output}")
    print(f"Grid: {args.height}x{args.width}")
    print(f"Frames: {frame_count}")
    print(f"Tile size: {tile_width}x{tile_height}")
    print(f"Output size: {canvas_width}x{canvas_height}")


if __name__ == "__main__":
    main()