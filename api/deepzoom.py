from functools import lru_cache
from io import BytesIO
from pathlib import Path
import os

import openslide
from openslide.deepzoom import DeepZoomGenerator
from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/slides", tags=["Deep Zoom Viewer"])

SLIDE_ROOT = Path(
    os.environ.get("SLIDE_ROOT", "/app/data/raw/camelyon16/images")
)

SUPPORTED_SUFFIXES = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".svslide",
    ".dcm",
}

TILE_SIZE = 254
TILE_OVERLAP = 1
TILE_FORMAT = "jpeg"
JPEG_QUALITY = 85


def safe_slide_path(slide_name: str) -> Path:
    """
    Prevent path traversal. Only allow files directly under SLIDE_ROOT.
    Example allowed:
      tumor_005.tif
      normal_019.tif
    """
    clean_name = Path(slide_name).name
    slide_path = (SLIDE_ROOT / clean_name).resolve()
    root = SLIDE_ROOT.resolve()

    if root not in slide_path.parents:
        raise HTTPException(status_code=400, detail="Invalid slide path")

    if not slide_path.exists():
        raise HTTPException(status_code=404, detail=f"Slide not found: {clean_name}")

    if slide_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported slide type: {slide_path.suffix}",
        )

    return slide_path


@lru_cache(maxsize=8)
def get_generator(slide_name: str) -> DeepZoomGenerator:
    slide_path = safe_slide_path(slide_name)
    slide = openslide.OpenSlide(str(slide_path))

    return DeepZoomGenerator(
        slide,
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP,
        limit_bounds=False,
    )


@router.get("")
def list_slides():
    if not SLIDE_ROOT.exists():
        return {"slides": []}

    slides = []
    for path in sorted(SLIDE_ROOT.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            slides.append(path.name)

    return {"slides": slides}


@router.get("/{slide_name}/info")
def slide_info(slide_name: str):
    generator = get_generator(slide_name)

    width, height = generator.level_dimensions[-1]

    return {
        "slide_name": slide_name,
        "width": width,
        "height": height,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "tile_format": TILE_FORMAT,
        "level_count": generator.level_count,
        "level_dimensions": generator.level_dimensions,
        "level_tiles": generator.level_tiles,
        "tile_count": generator.tile_count,
    }


@router.get("/{slide_name}/dzi")
def slide_dzi(slide_name: str):
    generator = get_generator(slide_name)
    dzi_xml = generator.get_dzi(TILE_FORMAT)

    return Response(
        content=dzi_xml,
        media_type="application/xml",
    )


@router.get("/{slide_name}/tiles/{level}/{x}_{y}.{fmt}")
def get_tile(
    slide_name: str,
    level: int,
    x: int,
    y: int,
    fmt: str,
):
    if fmt.lower() not in {"jpeg", "jpg", "png"}:
        raise HTTPException(status_code=400, detail="Unsupported tile format")

    generator = get_generator(slide_name)

    if level < 0 or level >= generator.level_count:
        raise HTTPException(status_code=404, detail="Invalid Deep Zoom level")

    tiles_x, tiles_y = generator.level_tiles[level]
    if x < 0 or y < 0 or x >= tiles_x or y >= tiles_y:
        raise HTTPException(status_code=404, detail="Tile out of range")

    try:
        tile = generator.get_tile(level, (x, y)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    buffer = BytesIO()

    if fmt.lower() == "png":
        tile.save(buffer, format="PNG")
        media_type = "image/png"
    else:
        tile.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        media_type = "image/jpeg"

    return Response(
        content=buffer.getvalue(),
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=86400",
        },
    )
