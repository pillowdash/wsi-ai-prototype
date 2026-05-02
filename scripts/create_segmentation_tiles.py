from pathlib import Path
import argparse
import csv
import random
import xml.etree.ElementTree as ET

import numpy as np
import openslide
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paired image/mask tiles for tumor segmentation from CAMELYON WSI + XML polygons."
    )
    parser.add_argument(
        "--slide",
        required=True,
        help="Path to WSI slide, e.g. data/raw/camelyon16/images/test_001.tif",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to CAMELYON XML annotation, e.g. data/raw/camelyon16/annotations/test_001.xml",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/segmentation",
        help="Output root folder for segmentation dataset.",
    )
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-positive", type=int, default=1000)
    parser.add_argument("--max-negative", type=int, default=1000)
    parser.add_argument(
        "--min-positive-mask-ratio",
        type=float,
        default=0.01,
        help="Minimum tumor-mask fraction for a tile to be saved as positive.",
    )
    parser.add_argument(
        "--max-negative-mask-ratio",
        type=float,
        default=0.001,
        help="Maximum tumor-mask fraction for a tile to be saved as negative.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Padding fraction around polygon bounding boxes for positive sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_polygons(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []

    for annotation in root.iter():
        if annotation.tag.lower().endswith("annotation"):
            coords = []

            for coord in annotation.iter():
                if coord.tag.lower().endswith("coordinate"):
                    x = coord.attrib.get("X")
                    y = coord.attrib.get("Y")

                    if x is not None and y is not None:
                        coords.append((float(x), float(y)))

            if coords:
                polygons.append(coords)

    return polygons


def polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def bboxes_intersect(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    return not (
        ax1 < bx0 or
        bx1 < ax0 or
        ay1 < by0 or
        by1 < ay0
    )


def is_mostly_background(tile: Image.Image, threshold: int = 220, white_ratio: float = 0.80) -> bool:
    arr = np.array(tile)
    gray = arr.mean(axis=2)
    return (gray > threshold).mean() > white_ratio


def create_tile_mask(
    polygons,
    polygon_bboxes,
    x_level,
    y_level,
    downsample,
    tile_size,
):
    """
    Create a binary tumor mask for one tile.

    Polygon coordinates are in level-0 slide coordinates.
    Tile x/y are in selected pyramid level coordinates.
    """
    mask = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(mask)

    x0_level0 = x_level * downsample
    y0_level0 = y_level * downsample
    x1_level0 = x0_level0 + tile_size * downsample
    y1_level0 = y0_level0 + tile_size * downsample

    tile_bbox_level0 = (x0_level0, y0_level0, x1_level0, y1_level0)

    for poly, poly_bbox in zip(polygons, polygon_bboxes):
        if not bboxes_intersect(tile_bbox_level0, poly_bbox):
            continue

        scaled_points = []

        for px, py in poly:
            tx = int(round((px - x0_level0) / downsample))
            ty = int(round((py - y0_level0) / downsample))
            scaled_points.append((tx, ty))

        if len(scaled_points) >= 3:
            draw.polygon(scaled_points, fill=255)

    return mask


def positive_candidate_coords(
    polygon_bboxes,
    level_width,
    level_height,
    downsample,
    tile_size,
    stride,
    padding,
):
    coords = set()

    for bbox in polygon_bboxes:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0

        x0 -= w * padding
        y0 -= h * padding
        x1 += w * padding
        y1 += h * padding

        x0_level = max(0, int(x0 / downsample))
        y0_level = max(0, int(y0 / downsample))
        x1_level = min(level_width - tile_size, int(x1 / downsample))
        y1_level = min(level_height - tile_size, int(y1 / downsample))

        start_x = max(0, (x0_level // stride) * stride)
        start_y = max(0, (y0_level // stride) * stride)
        end_x = max(0, (x1_level // stride) * stride)
        end_y = max(0, (y1_level // stride) * stride)

        for y in range(start_y, end_y + 1, stride):
            for x in range(start_x, end_x + 1, stride):
                if x + tile_size <= level_width and y + tile_size <= level_height:
                    coords.add((x, y))

    return list(coords)


def all_grid_coords(level_width, level_height, tile_size, stride):
    coords = []

    for y in range(0, level_height - tile_size + 1, stride):
        for x in range(0, level_width - tile_size + 1, stride):
            coords.append((x, y))

    return coords


def save_tile_pair(
    slide,
    slide_id,
    x,
    y,
    level,
    downsample,
    tile_size,
    image_dir,
    mask_dir,
    polygons,
    polygon_bboxes,
):
    x_level0 = int(x * downsample)
    y_level0 = int(y * downsample)

    tile = slide.read_region(
        (x_level0, y_level0),
        level,
        (tile_size, tile_size),
    ).convert("RGB")

    mask = create_tile_mask(
        polygons=polygons,
        polygon_bboxes=polygon_bboxes,
        x_level=x,
        y_level=y,
        downsample=downsample,
        tile_size=tile_size,
    )

    mask_arr = np.array(mask)
    mask_ratio = float((mask_arr > 0).mean())

    tile_name = f"{slide_id}_x{x}_y{y}.png"
    mask_name = f"{slide_id}_x{x}_y{y}_mask.png"

    image_path = image_dir / tile_name
    mask_path = mask_dir / mask_name

    tile.save(image_path)
    mask.save(mask_path)

    return {
        "slide_id": slide_id,
        "tile_name": tile_name,
        "mask_name": mask_name,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "x_level": x,
        "y_level": y,
        "x_level0": x_level0,
        "y_level0": y_level0,
        "level": level,
        "tile_size": tile_size,
        "mask_ratio": round(mask_ratio, 6),
        "label": "tumor" if mask_ratio > 0 else "normal",
    }, tile, mask_ratio


def main():
    args = parse_args()
    random.seed(args.seed)

    slide_path = Path(args.slide)
    annotation_path = Path(args.annotation)
    output_root = Path(args.output_root)

    if not slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")

    slide_id = slide_path.stem

    image_dir = output_root / "images" / slide_id
    mask_dir = output_root / "masks" / slide_id
    manifest_dir = output_root / "manifests"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"{slide_id}_segmentation_tiles.csv"

    polygons = parse_polygons(annotation_path)
    polygon_bboxes = [polygon_bbox(poly) for poly in polygons]

    print(f"Slide: {slide_path.name}")
    print(f"Annotation: {annotation_path.name}")
    print(f"Polygons found: {len(polygons)}")

    slide = openslide.OpenSlide(str(slide_path))

    level_width, level_height = slide.level_dimensions[args.level]
    downsample = slide.level_downsamples[args.level]

    print(f"Level: {args.level}")
    print(f"Level dimensions: {level_width} x {level_height}")
    print(f"Downsample: {downsample}")

    rows = []
    saved_positive = 0
    saved_negative = 0

    # 1. Positive sampling around polygon regions
    pos_coords = positive_candidate_coords(
        polygon_bboxes=polygon_bboxes,
        level_width=level_width,
        level_height=level_height,
        downsample=downsample,
        tile_size=args.tile_size,
        stride=args.stride,
        padding=args.padding,
    )

    random.shuffle(pos_coords)

    print(f"Positive candidate tiles: {len(pos_coords)}")

    used_coords = set()

    for x, y in pos_coords:
        if saved_positive >= args.max_positive:
            break

        row, tile, mask_ratio = save_tile_pair(
            slide=slide,
            slide_id=slide_id,
            x=x,
            y=y,
            level=args.level,
            downsample=downsample,
            tile_size=args.tile_size,
            image_dir=image_dir,
            mask_dir=mask_dir,
            polygons=polygons,
            polygon_bboxes=polygon_bboxes,
        )

        if mask_ratio < args.min_positive_mask_ratio:
            # Remove saved non-useful files
            Path(row["image_path"]).unlink(missing_ok=True)
            Path(row["mask_path"]).unlink(missing_ok=True)
            continue

        if is_mostly_background(tile):
            Path(row["image_path"]).unlink(missing_ok=True)
            Path(row["mask_path"]).unlink(missing_ok=True)
            continue

        row["label"] = "tumor"
        rows.append(row)
        used_coords.add((x, y))
        saved_positive += 1

        if saved_positive % 100 == 0:
            print(f"Saved positive tiles: {saved_positive}")

    # 2. Negative sampling from tissue regions outside polygons
    neg_coords = all_grid_coords(
        level_width=level_width,
        level_height=level_height,
        tile_size=args.tile_size,
        stride=args.stride,
    )

    random.shuffle(neg_coords)

    print(f"Negative candidate tiles: {len(neg_coords)}")

    for x, y in neg_coords:
        if saved_negative >= args.max_negative:
            break

        if (x, y) in used_coords:
            continue

        row, tile, mask_ratio = save_tile_pair(
            slide=slide,
            slide_id=slide_id,
            x=x,
            y=y,
            level=args.level,
            downsample=downsample,
            tile_size=args.tile_size,
            image_dir=image_dir,
            mask_dir=mask_dir,
            polygons=polygons,
            polygon_bboxes=polygon_bboxes,
        )

        if mask_ratio > args.max_negative_mask_ratio:
            Path(row["image_path"]).unlink(missing_ok=True)
            Path(row["mask_path"]).unlink(missing_ok=True)
            continue

        if is_mostly_background(tile):
            Path(row["image_path"]).unlink(missing_ok=True)
            Path(row["mask_path"]).unlink(missing_ok=True)
            continue

        row["label"] = "normal"
        rows.append(row)
        saved_negative += 1

        if saved_negative % 100 == 0:
            print(f"Saved negative tiles: {saved_negative}")

    slide.close()

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slide_id",
                "tile_name",
                "mask_name",
                "image_path",
                "mask_path",
                "x_level",
                "y_level",
                "x_level0",
                "y_level0",
                "level",
                "tile_size",
                "mask_ratio",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSegmentation tile generation complete.")
    print(f"Positive tiles saved: {saved_positive}")
    print(f"Negative tiles saved: {saved_negative}")
    print(f"Total tiles saved:    {len(rows)}")
    print(f"Manifest saved to:    {manifest_path}")
    print(f"Images saved to:      {image_dir}")
    print(f"Masks saved to:       {mask_dir}")


if __name__ == "__main__":
    main()
