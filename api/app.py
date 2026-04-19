from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import os
import time

app = FastAPI(title="WSI AI Inference API")

# Paths (container-friendly)
DATA_DIR = Path("/app/data")
RAW_DIR = DATA_DIR / "raw/camelyon16/images"
TILES_DIR = DATA_DIR / "processed/tiles"
PRED_DIR = DATA_DIR / "processed/predictions"
HEATMAP_DIR = DATA_DIR / "processed/heatmaps"

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "/app/models_external/best_model.pth"
)

class InferenceRequest(BaseModel):
    slide_name: str  # e.g. tumor_005.tif

class InferenceResponse(BaseModel):
    slide_name: str
    predictions_csv: str
    heatmap_image: str
    tiles_processed: int
    status: str


def run_script(cmd: list):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


@app.get("/")
def root():
    return {"message": "WSI AI Inference API is running"}


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    slide_path = RAW_DIR / req.slide_name

    if not slide_path.exists():
        raise HTTPException(status_code=404, detail=f"Slide not found: {slide_path}")

    start_time = time.time()

    try:
        # Step 1: Extract tiles
        run_script([
            "python", "scripts/extract_tiles.py",
            "--slide", str(slide_path)
        ])

        # Step 2: Run inference
        run_script([
            "python", "scripts/run_inference.py"
        ])

        # Step 3: Generate heatmap
        run_script([
            "python", "scripts/generate_heatmap.py"
        ])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - start_time, 2)

    # Expected outputs
    slide_id = req.slide_name.replace(".tif", "")
    pred_csv = PRED_DIR / f"{slide_id}_predictions.csv"
    heatmap = HEATMAP_DIR / f"{slide_id}_overlay.png"

    return InferenceResponse(
        slide_name=req.slide_name,
        predictions_csv=str(pred_csv),
        heatmap_image=str(heatmap),
        tiles_processed=0,  # optional: can parse later
        status=f"completed in {elapsed}s"
    )
