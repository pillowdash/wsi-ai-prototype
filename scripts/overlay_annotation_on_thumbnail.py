from pathlib import Path
import xml.etree.ElementTree as ET

import openslide
from PIL import Image, ImageDraw


SLIDE_PATH = Path("data/raw/camelyon16/images/test_001.tif")
ANNOTATION_PATH = Path("data/raw/camelyon16/annotations/test_001.xml")
OUTPUT_DIR = Path("data/processed/annotations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "test_001_thumbnail_polygon_overlay.png"

THUMBNAIL_MAX_SIZE = (1600, 1600)

# RGBA colors
POLYGON_OUTLINE = (255, 0, 0, 255)      # red outline
POLYGON_FILL = (255, 0, 0, 70)          # semi-transparent red fill


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


def main():
    if not SLIDE_PATH.exists():
        raise FileNotFoundError(f"Slide not found: {SLIDE_PATH}")

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Annotation not found: {ANNOTATION_PATH}")

    slide = openslide.OpenSlide(str(SLIDE_PATH))
    slide_w, slide_h = slide.dimensions

    thumbnail = slide.get_thumbnail(THUMBNAIL_MAX_SIZE).convert("RGBA")
    thumb_w, thumb_h = thumbnail.size

    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h

    polygons = parse_polygons(ANNOTATION_PATH)
    print(f"Found {len(polygons)} polygons")

    overlay = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for polygon in polygons:
        scaled = scale_polygon(polygon, scale_x, scale_y)

        # Need at least 2 points to draw
        if len(scaled) >= 2:
            draw.polygon(scaled, fill=POLYGON_FILL, outline=POLYGON_OUTLINE)

    combined = Image.alpha_composite(thumbnail, overlay)
    combined.save(OUTPUT_IMAGE)

    print(f"Saved polygon overlay image to: {OUTPUT_IMAGE}")
    print(f"Slide dimensions: {slide_w} x {slide_h}")
    print(f"Thumbnail size: {thumb_w} x {thumb_h}")

    slide.close()


if __name__ == "__main__":
    main()
