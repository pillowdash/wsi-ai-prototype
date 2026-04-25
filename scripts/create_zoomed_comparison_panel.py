from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import openslide
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a zoomed 4-panel comparison around an annotated tumor region."
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
        help="Optional output path",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=1400,
        help="Maximum thumbnail width/height",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.20,
        help="Padding fraction around selected polygon bounding box",
    )
    parser.add_argument(
        "--polygon-mode",
        choices=["largest", "all"],
        default="largest",
        help="Use largest polygon bbox or bbox covering all polygons",
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


def polygon_area(poly):
    """Shoelace formula."""
    if len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def get_bbox_for_polygon(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def get_bbox_for_polygons(polygons):
    xs = []
    ys = []
    for poly in polygons:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def apply_padding_to_bbox(bbox, slide_w, slide_h, padding_frac):
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    pad_x = w * padding_frac
    pad_y = h * padding_frac

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(slide_w, x1 + pad_x)
    y1 = min(slide_h, y1 + pad_y)

    return x0, y0, x1, y1


def scale_polygon(polygon, scale_x, scale_y):
    return [(x * scale_x, y * scale_y) for x, y in polygon]


def create_polygon_overlay(base_image, polygons, slide_w, slide_h):
    base_rgba = base_image.convert("RGBA")
    width, height = base_rgba.size
    scale_x = width / slide_w
    scale_y = height / slide_h

    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    outline_color = (255, 0, 0, 255)
    fill_color = (255, 0, 0, 70)

    for polygon in polygons:
        scaled = scale_polygon(polygon, scale_x, scale_y)
        if len(scaled) >= 2:
            draw.polygon(scaled, fill=fill_color, outline=outline_color)

    return Image.alpha_composite(base_rgba, overlay)


def create_combined_overlay(base_image, heatmap_img, polygons, slide_w, slide_h):
    base_rgba = base_image.convert("RGBA")
    heatmap_rgba = heatmap_img.convert("RGBA").resize(base_rgba.size, Image.Resampling.LANCZOS)

    blended = Image.blend(base_rgba, heatmap_rgba, alpha=0.45)

    width, height = blended.size
    scale_x = width / slide_w
    scale_y = height / slide_h

    polygon_layer = Image.new("RGBA", blended.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(polygon_layer, "RGBA")

    outline_color = (255, 0, 0, 255)
    fill_color = (255, 0, 0, 50)

    for polygon in polygons:
        scaled = scale_polygon(polygon, scale_x, scale_y)
        if len(scaled) >= 2:
            draw.polygon(scaled, fill=fill_color, outline=outline_color)

    return Image.alpha_composite(blended, polygon_layer)


def slide_bbox_to_thumb_bbox(bbox, slide_w, slide_h, thumb_w, thumb_h):
    x0, y0, x1, y1 = bbox
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h
    return (
        int(x0 * scale_x),
        int(y0 * scale_y),
        int(x1 * scale_x),
        int(y1 * scale_y),
    )


def crop_image(img, bbox):
    return img.crop(bbox)


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


def resize_to_common_height(images, target_height=700):
    resized = []
    for img in images:
        w, h = img.size
        new_w = int(w * (target_height / h))
        resized.append(img.resize((new_w, target_height), Image.Resampling.LANCZOS))
    return resized


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
        output_path = output_dir / f"{slide_id}_zoomed_comparison_panel.png"

    slide = openslide.OpenSlide(str(slide_path))
    slide_w, slide_h = slide.dimensions

    thumbnail = slide.get_thumbnail((args.thumb_size, args.thumb_size)).convert("RGBA")
    polygons = parse_polygons(annotation_path)

    if not polygons:
        raise ValueError("No polygons found in annotation file.")

    if args.polygon_mode == "largest":
        selected_poly = max(polygons, key=polygon_area)
        selected_polygons = [selected_poly]
        bbox = get_bbox_for_polygon(selected_poly)
    else:
        selected_polygons = polygons
        bbox = get_bbox_for_polygons(polygons)

    padded_bbox = apply_padding_to_bbox(bbox, slide_w, slide_h, args.padding)

    polygon_overlay = create_polygon_overlay(thumbnail, polygons, slide_w, slide_h)

    heatmap_img = Image.open(heatmap_path).convert("RGB")
    heatmap_img = heatmap_img.resize(thumbnail.size, Image.Resampling.LANCZOS)

    combined_overlay = create_combined_overlay(
        thumbnail,
        heatmap_img,
        polygons,
        slide_w,
        slide_h,
    )

    thumb_w, thumb_h = thumbnail.size
    crop_bbox_thumb = slide_bbox_to_thumb_bbox(padded_bbox, slide_w, slide_h, thumb_w, thumb_h)

    thumb_crop = crop_image(thumbnail, crop_bbox_thumb)
    polygon_crop = crop_image(polygon_overlay, crop_bbox_thumb)
    heatmap_crop = crop_image(heatmap_img, crop_bbox_thumb)
    combined_crop = crop_image(combined_overlay, crop_bbox_thumb)

    thumb_crop = add_title(thumb_crop, "Zoomed Thumbnail")
    polygon_crop = add_title(polygon_crop, "Zoomed Polygon Overlay")
    heatmap_crop = add_title(heatmap_crop, "Zoomed Heatmap Overlay")
    combined_crop = add_title(combined_crop, "Zoomed Combined Overlay")

    panels = resize_to_common_height(
        [thumb_crop, polygon_crop, heatmap_crop, combined_crop],
        target_height=700
    )

    final_panel = make_panel(panels, gap=30)
    final_panel.save(output_path)

    slide.close()

    print(f"Saved zoomed comparison panel to: {output_path}")
    print(f"Slide: {slide_path.name}")
    print(f"Annotation: {annotation_path.name}")
    print(f"Heatmap: {heatmap_path.name}")
    print(f"Polygons found: {len(polygons)}")
    print(f"Polygon mode: {args.polygon_mode}")
    print(f"Crop bbox (slide coords): {padded_bbox}")


if __name__ == "__main__":
    main()
