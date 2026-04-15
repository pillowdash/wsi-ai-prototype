from pathlib import Path
import openslide

SLIDES_DIR = Path("data/raw/camelyon16/images")
THUMB_DIR = Path("data/interim/thumbnails")
THUMB_DIR.mkdir(parents=True, exist_ok=True)

def inspect_slide(slide_path: Path) -> None:
    slide = openslide.OpenSlide(str(slide_path))

    print(f"\nSlide: {slide_path.name}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Level count: {slide.level_count}")
    print(f"Level dimensions: {slide.level_dimensions}")
    print(f"Level downsamples: {slide.level_downsamples}")

    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    thumb_path = THUMB_DIR / f"{slide_path.stem}_thumbnail.png"
    thumb.save(thumb_path)
    print(f"Saved thumbnail to: {thumb_path}")

    slide.close()

def main() -> None:
    slide_files = sorted(SLIDES_DIR.glob("*.tif")) + sorted(SLIDES_DIR.glob("*.svs"))
    if not slide_files:
        print("No slides found in data/raw/camelyon16/images")
        return

    for slide_path in slide_files:
        inspect_slide(slide_path)

if __name__ == "__main__":
    main()
