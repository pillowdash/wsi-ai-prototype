from pathlib import Path
import openslide
import numpy as np
from PIL import Image

SLIDE_PATH = Path("data/raw/camelyon16/images/tumor_005.tif")
OUTPUT_DIR = Path("data/processed/tiles/tumor_005")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 256
LEVEL = 2
MAX_TILES = 300
STRIDE = 256

def is_mostly_background(img: Image.Image, threshold: int = 220, white_ratio: float = 0.80) -> bool:
    arr = np.array(img)
    gray = arr.mean(axis=2)
    return (gray > threshold).mean() > white_ratio

def main() -> None:
    slide = openslide.OpenSlide(str(SLIDE_PATH))

    level_width, level_height = slide.level_dimensions[LEVEL]
    downsample = slide.level_downsamples[LEVEL]

    print(f"Processing: {SLIDE_PATH.name}")
    print(f"Level {LEVEL} dimensions: {level_width} x {level_height}")
    print(f"Downsample: {downsample}")

    saved = 0
    checked = 0

    for y in range(0, level_height - TILE_SIZE + 1, STRIDE):
        for x in range(0, level_width - TILE_SIZE + 1, STRIDE):
            checked += 1

            x_level0 = int(x * downsample)
            y_level0 = int(y * downsample)

            tile = slide.read_region(
                (x_level0, y_level0),
                LEVEL,
                (TILE_SIZE, TILE_SIZE)
            ).convert("RGB")

            if is_mostly_background(tile):
                continue

            tile_name = f"{SLIDE_PATH.stem}_x{x}_y{y}.png"
            tile_path = OUTPUT_DIR / tile_name
            tile.save(tile_path)

            saved += 1

            if saved % 25 == 0:
                print(f"Saved {saved} tiles so far...")

            if saved >= MAX_TILES:
                print(f"Reached MAX_TILES={MAX_TILES}")
                print(f"Checked {checked} regions, saved {saved} tiles.")
                slide.close()
                return

    print(f"Finished. Checked {checked} regions, saved {saved} tiles.")
    slide.close()

if __name__ == "__main__":
    main()
