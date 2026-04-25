from pathlib import Path
import xml.etree.ElementTree as ET

ANNOTATION_PATH = Path("data/raw/camelyon16/annotations/test_001.xml")

def main():
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Annotation not found: {ANNOTATION_PATH}")

    tree = ET.parse(ANNOTATION_PATH)
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

    print(f"Annotation file: {ANNOTATION_PATH.name}")
    print(f"Polygons found: {len(polygons)}")

    for i, poly in enumerate(polygons[:5]):
        print(f"Polygon {i}: {len(poly)} points")
        print(poly[:3], "...")

if __name__ == "__main__":
    main()
