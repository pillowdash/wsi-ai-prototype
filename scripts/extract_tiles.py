from pathlib import Path
import csv

import numpy as np
import openslide
from PIL import Image

SLIDE_PATH = Path("data/raw/camelyon16/images/tumor_005.tif")
OUTPUT_DIR = Path("data/processed/tiles/tumor_005")
INDEX_DIR = Path("data/interim/tile_index")
THUMB_DIR = Path("data/interim/thumbnails")
MASK_DIR = Path("data/interim/tissue_masks")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV = INDEX_DIR / f"{SLIDE_PATH.stem}_tile_index.csv"
THUMB_PATH = THUMB_DIR / f"{SLIDE_PATH.stem}_thumbnail.png"
MASK_PATH = MASK_DIR / f"{SLIDE_PATH.stem}_tissue_mask.png"

TILE_SIZE = 256
LEVEL = 2
STRIDE = 256
MAX_TILES = 1000

# Border exclusion in LEVEL coordinate space
BORDER_MARGIN_X = 512
BORDER_MARGIN_Y = 512

# Thumbnail size for tissue masking
THUMB_MAX_SIZE = (1024, 1024)

# Tissue mask parameters
SATURATION_THRESHOLD = 20
VALUE_THRESHOLD = 245
MIN_TISSUE_RATIO = 0.10


def rgb_to_hsv_np(arr: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8 image [H,W,3] to HSV uint8-like scale:
    H in [0, 255], S in [0, 255], V in [0, 255]
    """
    arr = arr.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    v = maxc

    deltac = maxc - minc
    s = np.zeros_like(maxc)
    nonzero = maxc != 0
    s[nonzero] = deltac[nonzero] / maxc[nonzero]

    h = np.zeros_like(maxc)

    mask = deltac != 0
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)

    rc[mask] = (maxc[mask] - r[mask]) / deltac[mask]
    gc[mask] = (maxc[mask] - g[mask]) / deltac[mask]
    bc[mask] = (maxc[mask] - b[mask]) / deltac[mask]

    rmask = (r == maxc) & mask
    gmask = (g == maxc) & mask
    bmask = (b == maxc) & mask

    h[rmask] = bc[rmask] - gc[rmask]
    h[gmask] = 2.0 + rc[gmask] - bc[gmask]
    h[bmask] = 4.0 + gc[bmask] - rc[bmask]

    h = (h / 6.0) % 1.0

    hsv = np.stack([h * 255.0, s * 255.0, v * 255.0], axis=-1)
    return hsv.astype(np.uint8)


def build_tissue_mask(thumbnail: Image.Image) -> np.ndarray:
    """
    Build a simple tissue mask from thumbnail using HSV rules:
    - tissue tends to have higher saturation than blank background
    - blank background tends to have very high value (brightness)
    """
    thumb_arr = np.array(thumbnail.convert("RGB"))
    hsv = rgb_to_hsv_np(thumb_arr)

    saturation = hsv[..., 1]
    value = hsv[..., 2]

    tissue_mask = (saturation > SATURATION_THRESHOLD) & (value < VALUE_THRESHOLD)
    return tissue_mask.astype(np.uint8)


def save_mask_visual(mask: np.ndarray, path: Path) -> None:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img.save(path)


def get_thumbnail_and_mask(slide: openslide.OpenSlide):
    thumbnail = slide.get_thumbnail(THUMB_MAX_SIZE).convert("RGB")
    thumbnail.save(THUMB_PATH)

    tissue_mask = build_tissue_mask(thumbnail)
    save_mask_visual(tissue_mask, MASK_PATH)

    return thumbnail, tissue_mask


def level_to_thumb_coords(
    x_level: int,
    y_level: int,
    level_width: int,
    level_height: int,
    thumb_width: int,
    thumb_height: int,
):
    tx = int(x_level / level_width * thumb_width)
    ty = int(y_level / level_height * thumb_height)

    tx = max(0, min(tx, thumb_width - 1))
    ty = max(0, min(ty, thumb_height - 1))
    return tx, ty


def tile_is_tissue_positive(
    x_level: int,
    y_level: int,
    tile_size: int,
    level_width: int,
    level_height: int,
    tissue_mask: np.ndarray,
) -> float:
    """
    Estimate tissue coverage by projecting tile corners/region into thumbnail mask.
    """
    thumb_h, thumb_w = tissue_mask.shape

    x2 = min(x_level + tile_size, level_width - 1)
    y2 = min(y_level + tile_size, level_height - 1)

    tx1, ty1 = level_to_thumb_coords(x_level, y_level, level_width, level_height, thumb_w, thumb_h)
    tx2, ty2 = level_to_thumb_coords(x2, y2, level_width, level_height, thumb_w, thumb_h)

    if tx2 < tx1:
        tx1, tx2 = tx2, tx1
    if ty2 < ty1:
        ty1, ty2 = ty2, ty1

    patch = tissue_mask[ty1:ty2 + 1, tx1:tx2 + 1]
    if patch.size == 0:
        return 0.0

    return float(patch.mean())


def is_mostly_background(tile: Image.Image, threshold: int = 220, white_ratio: float = 0.80) -> bool:
    arr = np.array(tile)
    gray = arr.mean(axis=2)
    return (gray > threshold).mean() > white_ratio


def main() -> None:
    if not SLIDE_PATH.exists():
        raise FileNotFoundError(f"Slide not found: {SLIDE_PATH}")

    slide = openslide.OpenSlide(str(SLIDE_PATH))
    level_width, level_height = slide.level_dimensions[LEVEL]
    downsample = slide.level_downsamples[LEVEL]

    print(f"Processing slide: {SLIDE_PATH.name}")
    print(f"Level {LEVEL} dimensions: {level_width} x {level_height}")
    print(f"Downsample: {downsample}")

    thumbnail, tissue_mask = get_thumbnail_and_mask(slide)
    print(f"Saved thumbnail to: {THUMB_PATH}")
    print(f"Saved tissue mask to: {MASK_PATH}")

    saved = 0
    checked = 0
    skipped_border = 0
    skipped_tissue = 0
    skipped_background = 0

    rows = []

    for y in range(0, level_height - TILE_SIZE + 1, STRIDE):
        for x in range(0, level_width - TILE_SIZE + 1, STRIDE):
            checked += 1

            # Border exclusion
            if (
                x < BORDER_MARGIN_X
                or y < BORDER_MARGIN_Y
                or x + TILE_SIZE > level_width - BORDER_MARGIN_X
                or y + TILE_SIZE > level_height - BORDER_MARGIN_Y
            ):
                skipped_border += 1
                continue

            tissue_ratio = tile_is_tissue_positive(
                x_level=x,
                y_level=y,
                tile_size=TILE_SIZE,
                level_width=level_width,
                level_height=level_height,
                tissue_mask=tissue_mask,
            )

            if tissue_ratio < MIN_TISSUE_RATIO:
                skipped_tissue += 1
                continue

            x_level0 = int(x * downsample)
            y_level0 = int(y * downsample)

            tile = slide.read_region(
                (x_level0, y_level0),
                LEVEL,
                (TILE_SIZE, TILE_SIZE),
            ).convert("RGB")

            if is_mostly_background(tile):
                skipped_background += 1
                continue

            tile_name = f"{SLIDE_PATH.stem}_x{x}_y{y}.png"
            tile_path = OUTPUT_DIR / tile_name
            tile.save(tile_path)

            rows.append({
                "tile_name": tile_name,
                "slide_name": SLIDE_PATH.name,
                "level": LEVEL,
                "x_level": x,
                "y_level": y,
                "x_level0": x_level0,
                "y_level0": y_level0,
                "tile_size": TILE_SIZE,
                "stride": STRIDE,
                "tissue_ratio": round(tissue_ratio, 4),
                "thumbnail_path": str(THUMB_PATH),
                "tissue_mask_path": str(MASK_PATH),
                "tile_path": str(tile_path),
            })

            saved += 1

            if saved % 50 == 0:
                print(f"Saved {saved} tiles so far...")

            if saved >= MAX_TILES:
                print(f"Reached MAX_TILES={MAX_TILES}")
                break

        if saved >= MAX_TILES:
            break

    slide.close()

    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_name",
                "slide_name",
                "level",
                "x_level",
                "y_level",
                "x_level0",
                "y_level0",
                "tile_size",
                "stride",
                "tissue_ratio",
                "thumbnail_path",
                "tissue_mask_path",
                "tile_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nExtraction complete")
    print(f"Checked regions:        {checked}")
    print(f"Skipped border:         {skipped_border}")
    print(f"Skipped low tissue:     {skipped_tissue}")
    print(f"Skipped background:     {skipped_background}")
    print(f"Saved tiles:            {saved}")
    print(f"Tile index saved to:    {INDEX_CSV}")
    print(f"Tiles saved to:         {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
