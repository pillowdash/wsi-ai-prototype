import subprocess
import time
import math

import csv
from PIL import Image, ImageDraw
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
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/app/data"))

PREDICTION_DIR = DATA_ROOT / "processed/predictions"
TILE_INDEX_DIR = DATA_ROOT / "interim/tile_index"

DEFAULT_PREDICTION_LEVEL = int(os.environ.get("PREDICTION_LEVEL", "2"))
DEFAULT_PREDICTION_TILE_SIZE = int(os.environ.get("PREDICTION_TILE_SIZE", "256"))
HEATMAP_SCORE_COLUMN = os.environ.get("HEATMAP_SCORE_COLUMN", "prob_class_1")
HEATMAP_MIN_SCORE = float(os.environ.get("HEATMAP_MIN_SCORE", "0.05"))

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

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))


def run_script(cmd: list[str], env_extra: dict | None = None) -> str:
    env = os.environ.copy()

    if env_extra:
        env.update(env_extra)

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    return result.stdout


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


def slide_stem(slide_name: str) -> str:
    return Path(slide_name).stem


def prediction_csv_path(slide_name: str) -> Path:
    return PREDICTION_DIR / f"{slide_stem(slide_name)}_predictions.csv"


def tile_index_csv_path(slide_name: str) -> Path:
    return TILE_INDEX_DIR / f"{slide_stem(slide_name)}_tile_index.csv"


def file_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


def parse_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def load_tile_index(index_path: Path) -> dict:
    """
    Reads data/interim/tile_index/{slide}_tile_index.csv if available.

    This is useful because extract_tiles.py records:
      - level
      - x_level
      - y_level
      - x_level0
      - y_level0
      - tile_size
    """
    if not index_path.exists():
        return {}

    with index_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {
            row["tile_name"]: row
            for row in reader
            if row.get("tile_name")
        }


def score_to_rgba(score: float):
    """
    Simple yellow-to-red heatmap.

    Low score: more yellow / transparent
    High score: red / stronger alpha
    """
    score = max(0.0, min(1.0, score))

    red = 255
    green = int(255 * (1.0 - score))
    blue = 0
    alpha = int(40 + score * 150)

    return red, green, blue, alpha


def empty_png(width: int, height: int) -> bytes:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def get_prediction_boxes(slide_name: str):
    """
    Public wrapper.

    The mtime values are included in the cached function signature so that
    if you regenerate prediction CSVs, the cache automatically refreshes.
    """
    pred_path = prediction_csv_path(slide_name)
    index_path = tile_index_csv_path(slide_name)

    return _load_prediction_boxes_cached(
        slide_name,
        file_mtime_ns(pred_path),
        file_mtime_ns(index_path),
    )


@lru_cache(maxsize=16)
def _load_prediction_boxes_cached(
    slide_name: str,
    pred_mtime_ns: int,
    index_mtime_ns: int,
):
    pred_path = prediction_csv_path(slide_name)
    index_path = tile_index_csv_path(slide_name)

    if not pred_path.exists():
        return []

    slide = get_slide(slide_name)
    tile_index = load_tile_index(index_path)

    boxes = []

    with pred_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            score = parse_float(row.get(HEATMAP_SCORE_COLUMN), 0.0)

            if score < HEATMAP_MIN_SCORE:
                continue

            tile_name = row.get("tile_name", "")

            # Best path: use tile_index CSV from extract_tiles.py.
            # This gives exact level-0 coordinates.
            if tile_name in tile_index:
                idx = tile_index[tile_name]

                level = parse_int(idx.get("level"), DEFAULT_PREDICTION_LEVEL)
                tile_size = parse_int(idx.get("tile_size"), DEFAULT_PREDICTION_TILE_SIZE)

                x0 = parse_float(idx.get("x_level0"), 0.0)
                y0 = parse_float(idx.get("y_level0"), 0.0)

                downsample = float(slide.level_downsamples[level])
                x1 = x0 + tile_size * downsample
                y1 = y0 + tile_size * downsample

            else:
                # Fallback path: current run_inference.py outputs x/y parsed
                # from filenames like tumor_005_x123_y456.png.
                # In your project, those x/y values are LEVEL=2 coordinates.
                level = parse_int(row.get("level"), DEFAULT_PREDICTION_LEVEL)
                tile_size = parse_int(row.get("tile_size"), DEFAULT_PREDICTION_TILE_SIZE)

                downsample = float(slide.level_downsamples[level])

                x_level = parse_float(row.get("x_level", row.get("x")), 0.0)
                y_level = parse_float(row.get("y_level", row.get("y")), 0.0)

                x0 = x_level * downsample
                y0 = y_level * downsample
                x1 = (x_level + tile_size) * downsample
                y1 = (y_level + tile_size) * downsample

            boxes.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "score": score,
                    "tile_name": tile_name,
                }
            )

    return boxes




@router.post("/{slide_name}/infer")
def infer_slide(slide_name: str):
    """
    Runs tile extraction and TIAToolbox inference for the selected slide.

    Outputs:
      data/processed/predictions/{slide_stem}_predictions.csv

    The AI heatmap overlay reads that prediction CSV.
    """
    slide_path = safe_slide_path(slide_name)
    stem = slide_stem(slide_name)

    tiles_dir = DATA_ROOT / "processed/tiles" / stem
    output_csv = PREDICTION_DIR / f"{stem}_predictions.csv"

    start_time = time.time()

    env_extra = {
        "SLIDE_ID": stem,
        "TILES_DIR": str(tiles_dir),
        "OUTPUT_DIR": str(PREDICTION_DIR),
        "OUTPUT_CSV": str(output_csv),
        "SLIDE_PATH": str(slide_path),
    }

    try:
        extract_output = run_script(
            [
                "python",
                "scripts/extract_tiles.py",
                "--slide",
                str(slide_path),
            ],
            env_extra=env_extra,
        )

        inference_output = run_script(
            [
                "python",
                "scripts/run_inference_tiatoolbox.py",
                "--slide-id",
                stem,
                "--model",
                os.environ.get("TIATOOLBOX_MODEL", "resnet18-pcam"),
                "--batch-size",
                os.environ.get("TIATOOLBOX_BATCH_SIZE", "32"),
                "--device",
                os.environ.get("TIATOOLBOX_DEVICE", "cuda"),
            ],
            env_extra=env_extra,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _load_prediction_boxes_cached.cache_clear()

    boxes = get_prediction_boxes(slide_name)
    scores = [box["score"] for box in boxes]

    elapsed = round(time.time() - start_time, 2)

    return {
        "slide_name": slide_name,
        "status": "completed",
        "model": os.environ.get("TIATOOLBOX_MODEL", "resnet18-pcam"),
        "device": os.environ.get("TIATOOLBOX_DEVICE", "cuda"),
        "elapsed_seconds": elapsed,
        "tiles_dir": str(tiles_dir),
        "prediction_csv": str(output_csv),
        "box_count": len(boxes),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "extract_log_tail": extract_output[-1200:],
        "inference_log_tail": inference_output[-1200:],
    }

@router.get("/{slide_name}/heatmap/info")
def heatmap_info(slide_name: str):
    pred_path = prediction_csv_path(slide_name)
    index_path = tile_index_csv_path(slide_name)
    boxes = get_prediction_boxes(slide_name)

    scores = [box["score"] for box in boxes]

    return {
        "slide_name": slide_name,
        "available": pred_path.exists() and len(boxes) > 0,
        "prediction_csv": str(pred_path),
        "tile_index_csv": str(index_path),
        "box_count": len(boxes),
        "score_column": HEATMAP_SCORE_COLUMN,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }


@router.get("/{slide_name}/heatmap-tiles/{level}/{x}_{y}.png")
def get_heatmap_tile(
    slide_name: str,
    level: int,
    x: int,
    y: int,
):
    generator = get_generator(slide_name)
    slide = get_slide(slide_name)

    if level < 0 or level >= generator.level_count:
        raise HTTPException(status_code=404, detail="Invalid Deep Zoom level")

    tiles_x, tiles_y = generator.level_tiles[level]
    if x < 0 or y < 0 or x >= tiles_x or y >= tiles_y:
        raise HTTPException(status_code=404, detail="Tile out of range")

    tile_width, tile_height = generator.get_tile_dimensions(level, (x, y))

    boxes = get_prediction_boxes(slide_name)

    if not boxes:
        return Response(
            content=empty_png(tile_width, tile_height),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    # DeepZoomGenerator gives the exact OpenSlide read_region arguments
    # for this Deep Zoom tile.
    location, slide_level, region_size = generator.get_tile_coordinates(level, (x, y))

    tile_x0_level0 = float(location[0])
    tile_y0_level0 = float(location[1])

    downsample = float(slide.level_downsamples[slide_level])

    tile_x1_level0 = tile_x0_level0 + float(region_size[0]) * downsample
    tile_y1_level0 = tile_y0_level0 + float(region_size[1]) * downsample

    tile_level0_width = tile_x1_level0 - tile_x0_level0
    tile_level0_height = tile_y1_level0 - tile_y0_level0

    overlay = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for box in boxes:
        # Skip prediction boxes that do not overlap this Deep Zoom tile.
        if (
            box["x1"] <= tile_x0_level0
            or box["x0"] >= tile_x1_level0
            or box["y1"] <= tile_y0_level0
            or box["y0"] >= tile_y1_level0
        ):
            continue

        overlap_x0 = max(box["x0"], tile_x0_level0)
        overlap_y0 = max(box["y0"], tile_y0_level0)
        overlap_x1 = min(box["x1"], tile_x1_level0)
        overlap_y1 = min(box["y1"], tile_y1_level0)

        px0 = int((overlap_x0 - tile_x0_level0) / tile_level0_width * tile_width)
        py0 = int((overlap_y0 - tile_y0_level0) / tile_level0_height * tile_height)
        px1 = int((overlap_x1 - tile_x0_level0) / tile_level0_width * tile_width)
        py1 = int((overlap_y1 - tile_y0_level0) / tile_level0_height * tile_height)

        if px1 <= px0 or py1 <= py0:
            continue

        draw.rectangle(
            [px0, py0, px1, py1],
            fill=score_to_rgba(box["score"]),
        )

    buffer = BytesIO()
    overlay.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@lru_cache(maxsize=8)
def get_slide(slide_name: str) -> openslide.OpenSlide:
    slide_path = safe_slide_path(slide_name)
    return openslide.OpenSlide(str(slide_path))

@lru_cache(maxsize=8)
def get_generator(slide_name: str) -> DeepZoomGenerator:
    slide = get_slide(slide_name)

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


    # This shows the actual translation.
    location, slide_level, region_size = generator.get_tile_coordinates(level, (x, y))

    print(
        f"Deep Zoom request: level={level}, tile=({x}, {y}) | "
        f"OpenSlide read_region: location={location}, "
        f"slide_level={slide_level}, region_size={region_size}"
    )


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



def distance_to_box(x: float, y: float, box: dict) -> float:
    dx = max(box["x0"] - x, 0, x - box["x1"])
    dy = max(box["y0"] - y, 0, y - box["y1"])

    return math.sqrt(dx * dx + dy * dy)


@router.get("/{slide_name}/prediction-at")
def prediction_at(
    slide_name: str,
    x: float,
    y: float,
    radius: float = 4096.0,
):
    """
    Returns the prediction box at or near a clicked WSI coordinate.

    x/y are level-0 full-slide image coordinates.
    """
    boxes = get_prediction_boxes(slide_name)

    if not boxes:
        return {
            "available": False,
            "message": "No prediction CSV found for this slide.",
            "x": x,
            "y": y,
        }

    containing = [
        box
        for box in boxes
        if box["x0"] <= x <= box["x1"] and box["y0"] <= y <= box["y1"]
    ]

    if containing:
        selected = max(containing, key=lambda b: b["score"])
        distance = 0.0
        hit_type = "inside_prediction_tile"
    else:
        nearest = min(boxes, key=lambda b: distance_to_box(x, y, b))
        distance = distance_to_box(x, y, nearest)

        if distance > radius:
            return {
                "available": True,
                "hit": False,
                "message": "No nearby prediction tile found.",
                "x": x,
                "y": y,
                "radius": radius,
                "nearest_distance": distance,
            }

        selected = nearest
        hit_type = "nearest_prediction_tile"

    return {
        "available": True,
        "hit": True,
        "hit_type": hit_type,
        "x": x,
        "y": y,
        "distance": distance,
        "tile_name": selected.get("tile_name"),
        "score": selected["score"],
        "box": {
            "x0": selected["x0"],
            "y0": selected["y0"],
            "x1": selected["x1"],
            "y1": selected["y1"],
        },
    }    
