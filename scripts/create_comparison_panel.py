from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import openslide
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create side-by-side comparison panel: thumbnail, polygon overlay, heatmap overlay."
    )
    parser.add_argument(
        "--slide",
        required=True,
        help="Path to WSI slide file, e.g. data/raw/camelyon16/images/test_001.tif",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to XML annotation file, e.g. data/raw/camelyon16/annotations/test_001.xml",
    )
    parser.add_argument(
        "--heatmap",
        required=True,
        help="Path to heatmap overlay image, e.g. data/processed/heatmaps/test_001_overlay.png",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional output path for the panel image",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=1200,
        help="Maximum thumbnail width/height",
    )
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


def scale_polygon(polygon, scale_x, scale_y):
    return [(x * scale_x, y * scale_y) for x, y in polygon]


def create_polygon_overlay(thumbnail: Image.Image, polygons, slide_w, slide_h):
    thumb_w, thumb_h = thumbnail.size
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h

    overlay = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    outline_color = (255, 0, 0, 255)
    fill_color = (255, 0, 0, 70)

    for polygon in polygons:
        scaled = scale_polygon(polygon, scale_x, scale_y)
        if len(scaled) >= 2:
            draw.polygon(scaled, fill=fill_color, outline=outline_color)

    combined = Image.alpha_composite(thumbnail.convert("RGBA"), overlay)
    return combined


def add_title(image: Image.Image, title: str, title_height: int = 50):
    width, height = image.size
    canvas = Image.new("RGB", (width, height + title_height), "white")
    canvas.paste(image.convert("RGB"), (0, title_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = (width - text_w) // 2
    text_y = (title_height - text_h) // 2

    draw.text((text_x, text_y), title, fill="black", font=font)
    return canvas


def make_panel(images, gap=20, background="white"):
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]

    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)

    panel = Image.new("RGB", (total_width, max_height), background)

    x = 0
    for img in images:
        panel.paste(img, (x, 0))
        x += img.size[0] + gap

    return panel


def main():
    args = parse_args()

    slide_path = Path(args.slide)
    annotation_path = Path(args.annotation)
    heatmap_path = Path(args.heatmap)

    if not slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")
    if not heatmap_path.exists():
        raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")

    slide_id = slide_path.stem

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("data/processed/comparisons")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{slide_id}_comparison_panel.png"

    slide = openslide.OpenSlide(str(slide_path))
    slide_w, slide_h = slide.dimensions

    thumbnail = slide.get_thumbnail((args.thumb_size, args.thumb_size)).convert("RGBA")
    polygons = parse_polygons(annotation_path)
    polygon_overlay = create_polygon_overlay(thumbnail, polygons, slide_w, slide_h)

    heatmap_img = Image.open(heatmap_path).convert("RGB")
    heatmap_img = heatmap_img.resize(thumbnail.size, Image.Resampling.LANCZOS)

    thumb_panel = add_title(thumbnail, "Thumbnail")
    polygon_panel = add_title(polygon_overlay, "Ground Truth Polygon Overlay")
    heatmap_panel = add_title(heatmap_img, "Heatmap Overlay")

    final_panel = make_panel([thumb_panel, polygon_panel, heatmap_panel], gap=30)
    final_panel.save(output_path)

    slide.close()

    print(f"Saved comparison panel to: {output_path}")
    print(f"Slide: {slide_path.name}")
    print(f"Annotation: {annotation_path.name}")
    print(f"Heatmap: {heatmap_path.name}")
    print(f"Polygons found: {len(polygons)}")


if __name__ == "__main__":
    main()
