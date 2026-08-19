from pathlib import Path
import argparse
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-pixel timestamp-vs-time PNGs into a ToF grid image."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing timestamps_vs_time_y*_x*.png files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged grid PNG path.",
    )

    parser.add_argument(
        "--tof-width",
        type=int,
        default=8,
        help="ToF grid width. Default: 8",
    )

    parser.add_argument(
        "--tof-height",
        type=int,
        default=8,
        help="ToF grid height. Default: 8",
    )

    parser.add_argument(
        "--label-tiles",
        action="store_true",
        help="Draw y/x labels on each tile.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_grid = []

    for y in range(args.tof_height):
        row = []
        for x in range(args.tof_width):
            path = args.input_dir / f"timestamps_vs_time_y{y}_x{x}.png"
            if not path.exists():
                raise FileNotFoundError(f"Missing plot: {path}")
            row.append(Image.open(path).convert("RGB"))
        image_grid.append(row)

    tile_w, tile_h = image_grid[0][0].size

    canvas_w = args.tof_width * tile_w
    canvas_h = args.tof_height * tile_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")

    for y in range(args.tof_height):
        for x in range(args.tof_width):
            img = image_grid[y][x]

            if args.label_tiles:
                draw = ImageDraw.Draw(img)
                draw.rectangle((5, 5, 110, 28), fill="white")
                draw.text((10, 8), f"y={y}, x={x}", fill="black")

            paste_x = x * tile_w
            paste_y = y * tile_h
            canvas.paste(img, (paste_x, paste_y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)

    print(f"Saved merged grid image to: {args.output}")


if __name__ == "__main__":
    main()