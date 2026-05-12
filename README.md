# WSI AI Prototype

## Overview

This project is a prototype for applying AI to Whole Slide Images (WSIs) in digital pathology.

It demonstrates an end-to-end workflow that:

- Loads high-resolution pathology slides such as `.tif`, `.tiff`, and `.svs`
- Reads large WSIs using OpenSlide
- Extracts tissue-aware tiles from gigapixel slides
- Runs tile-level inference using a pathology-specific TIAToolbox PCam model
- Aggregates prediction scores into heatmaps
- Serves WSI image tiles through FastAPI
- Displays slides in a zero-footprint browser viewer using OpenSeadragon
- Overlays AI heatmap tiles as a second transparent layer
- Allows users to click regions and inspect tile-level prediction scores

The goal is to simulate core components of a digital pathology AI system, including WSI ingestion, scalable tile processing, model inference, heatmap visualization, and browser-based slide review.

This is a research and engineering prototype. It is not clinically validated and must not be used for diagnosis.

---

## Key Features

- WSI ingestion with OpenSlide
- Support for large `.tif`, `.tiff`, and `.svs` pathology slides
- Tissue-aware tile extraction
- Border and background filtering
- Tile metadata export for coordinate mapping
- TIAToolbox PCam patch classification
- GPU-accelerated inference when CUDA is available
- Prediction CSV generation
- Static heatmap image generation for reports/screenshots
- Interactive zero-footprint WSI viewer
- FastAPI tile server using OpenSlide `DeepZoomGenerator`
- OpenSeadragon pan/zoom viewer
- Custom OpenSeadragon tile source using Deep Zoom-style `level/x/y` coordinates
- Optional DZI metadata endpoint
- AI heatmap overlay as a second tiled layer
- Heatmap toggle and opacity slider
- Browser-based `Run AI Inference` button
- Click-to-inspect prediction panel
- Desktop OpenSeadragon navigator/minimap
- Responsive mobile layout for phone/tablet testing

---

## Project Structure

```text
wsi-ai-prototype/
├── api/
│   ├── app.py                    # FastAPI app entry point
│   └── deepzoom.py               # WSI tile, heatmap, inference, and inspection routes
│
├── viewer/
│   └── index.html                # Zero-footprint OpenSeadragon WSI viewer
│
├── scripts/
│   ├── extract_tiles.py          # Tissue-aware WSI tile extraction
│   ├── run_inference_tiatoolbox.py
│   ├── generate_heatmap.py
│   └── download_dataset.sh
│
├── src/
│   └── ...                       # Supporting model / utility code
│
├── data/
│   ├── raw/                      # Local WSI files, excluded from Git
│   ├── interim/                  # Thumbnails, tissue masks, tile index CSVs
│   └── processed/                # Extracted tiles, predictions, heatmaps
│
├── img/                          # README screenshots / comparison panels
├── Dockerfile
├── requirements.txt
└── README.md
```

Large WSI files, extracted tiles, prediction outputs, and model artifacts are intentionally excluded from Git.

---

## Architecture

```text
Browser / OpenSeadragon
  ├── Base WSI tile layer
  │     └── GET /slides/{slide}/tiles/{level}/{x}_{y}.jpeg
  │
  └── AI heatmap overlay layer
        └── GET /slides/{slide}/heatmap-tiles/{level}/{x}_{y}.png

FastAPI backend
  ├── OpenSlide
  ├── DeepZoomGenerator
  ├── TIAToolbox PCam patch inference
  ├── prediction CSV parsing
  ├── heatmap tile rendering
  └── click-to-inspect prediction lookup
```

The browser does not download the full WSI. It requests only the visible tiles needed for the current viewport, similar to how map applications load only visible map tiles.

---

## Zero-Footprint WSI Viewer

The project includes a browser-based zero-footprint WSI viewer.

The viewer uses:

- FastAPI as the backend tile server
- OpenSlide / `DeepZoomGenerator` for reading WSI files
- OpenSeadragon for browser-based pan and zoom
- A custom OpenSeadragon tile source
- Docker for reproducible local deployment

Open the viewer at:

```text
http://localhost:8000/viewer
```

### Viewer Features

- Slide dropdown for selecting local WSI files
- Browser pan/zoom using OpenSeadragon
- Desktop navigator/minimap
- AI heatmap toggle
- Heatmap opacity slider
- `Run AI Inference` button
- Click-to-inspect prediction panel
- Mobile responsive layout
- Mobile bottom drawer for prediction inspection
- Hidden minimap on mobile to preserve viewing space

---

## Deep Zoom, DZI, and Custom Tile Source

The viewer follows the Deep Zoom tile coordinate convention:

```text
level = Deep Zoom pyramid level
x     = tile column at that Deep Zoom level
y     = tile row at that Deep Zoom level
```

The frontend does not need to fetch a `.dzi` XML file directly. Instead, it manually provides OpenSeadragon with metadata and tile URLs:

```javascript
tileSources: {
  width: info.width,
  height: info.height,
  tileSize: info.tile_size,
  tileOverlap: info.tile_overlap,
  minLevel: 0,
  maxLevel: info.level_count - 1,

  getTileUrl: function(level, x, y) {
    return `/slides/${encodedSlide}/tiles/${level}/${x}_${y}.jpeg`;
  }
}
```

The backend still exposes a DZI-compatible endpoint:

```text
GET /slides/{slide_name}/dzi
```

This is useful for compatibility and debugging, but the current viewer uses a custom OpenSeadragon tile source.

The important backend mapping is:

```python
location, slide_level, region_size = generator.get_tile_coordinates(level, (x, y))
tile = generator.get_tile(level, (x, y))
```

This maps OpenSeadragon’s Deep Zoom `level/x/y` tile request to an actual OpenSlide region in the WSI.

---

## AI Heatmap Overlay

The AI heatmap is rendered as a second tiled layer on top of the WSI base image.

```text
Base image tile:
GET /slides/{slide}/tiles/{level}/{x}_{y}.jpeg

AI heatmap tile:
GET /slides/{slide}/heatmap-tiles/{level}/{x}_{y}.png
```

The heatmap tiles are transparent PNGs. They are generated from the prediction CSV for each slide.

Example prediction CSV:

```text
data/processed/predictions/tumor_005_predictions.csv
```

Typical columns:

```text
tile_name,x,y,prob_class_0,prob_class_1,predicted_class
```

The viewer currently uses:

```text
prob_class_1
```

as the heatmap score.

For this demo, `prob_class_1` is treated as the class-1 / positive / suspicious probability. Low `prob_class_1` does not necessarily mean the model has low confidence overall; it may mean the model is confident in class 0.

---

## Click-to-Inspect Prediction

Clicking a region in the viewer calls:

```text
GET /slides/{slide_name}/prediction-at?x={wsi_x}&y={wsi_y}
```

The backend receives level-0 WSI coordinates and returns the nearest or containing prediction tile.

The inspector panel displays:

- Clicked WSI coordinate
- Class-1 probability
- Hit type
- Tile name
- Prediction box coordinates

This makes the heatmap more explainable than a static overlay image.

---

## Run AI Inference from the Viewer

The `Run AI Inference` button calls:

```text
POST /slides/{slide_name}/infer
```

The backend then:

1. Validates the selected slide path
2. Extracts tissue tiles from the selected WSI
3. Runs TIAToolbox patch inference
4. Writes a slide-specific prediction CSV
5. Clears cached heatmap data
6. Reloads the heatmap overlay in the viewer

Example output:

```text
data/processed/predictions/tumor_005_predictions.csv
```

The viewer then reloads the AI heatmap layer using a cache-busting version parameter:

```text
/slides/{slide}/heatmap-tiles/{level}/{x}_{y}.png?v={timestamp}
```

---

## Main API Endpoints

```text
GET  /viewer
GET  /docs

GET  /slides
GET  /slides/{slide_name}/info
GET  /slides/{slide_name}/dzi
GET  /slides/{slide_name}/tiles/{level}/{x}_{y}.jpeg

GET  /slides/{slide_name}/heatmap/info
GET  /slides/{slide_name}/heatmap-tiles/{level}/{x}_{y}.png
GET  /slides/{slide_name}/prediction-at?x={x}&y={y}

POST /slides/{slide_name}/infer

POST /predict
```

The `/predict` endpoint supports the earlier API-style inference workflow.

The `/slides/...` endpoints support the interactive zero-footprint viewer.

---

## Docker

The project can be containerized for reproducible execution. Large datasets and model artifacts are excluded from the image and should be mounted at runtime.

### Build

```bash
docker build -t wsi-ai .
```

### Verify NVIDIA Docker Runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Verify GPU Inside the Project Container

```bash
docker run --rm \
  --gpus all \
  -v "$(pwd)/data:/app/data" \
  wsi-ai \
  python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Example output:

```text
2.5.1+cu121
12.1
True
NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

---

## Run the ZFP Viewer with CAMELYON Slides

```bash
docker rm -f wsi-ai-viewer 2>/dev/null || true

docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e SLIDE_ROOT=/app/data/raw/camelyon16/images \
  -e TIATOOLBOX_MODEL=resnet18-pcam \
  -e TIATOOLBOX_BATCH_SIZE=32 \
  -e TIATOOLBOX_DEVICE=cuda \
  --name wsi-ai-viewer \
  wsi-ai
```

Open:

```text
http://localhost:8000/viewer
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## Run the ZFP Viewer with TCGA SVS Slides

Place downloaded TCGA `.svs` files under:

```text
data/raw/tcga/images/
```

Then run:

```bash
docker rm -f wsi-ai-viewer 2>/dev/null || true

docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e SLIDE_ROOT=/app/data/raw/tcga/images \
  -e TIATOOLBOX_MODEL=resnet18-pcam \
  -e TIATOOLBOX_BATCH_SIZE=32 \
  -e TIATOOLBOX_DEVICE=cuda \
  --name wsi-ai-viewer \
  wsi-ai
```

Open:

```text
http://localhost:8000/viewer
```

TCGA `.svs` files are useful for testing:

- Real SVS compatibility
- OpenSlide support
- Large WSI tile streaming
- Mobile browser access
- Viewer performance

For TCGA slides, AI heatmap output should be treated as exploratory model visualization only, because the PCam model is more aligned with CAMELYON-style lymph node metastasis slides.

---

## CPU-Only Viewer Run

```bash
docker rm -f wsi-ai-viewer 2>/dev/null || true

docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e SLIDE_ROOT=/app/data/raw/camelyon16/images \
  -e TIATOOLBOX_MODEL=resnet18-pcam \
  -e TIATOOLBOX_BATCH_SIZE=16 \
  -e TIATOOLBOX_DEVICE=cpu \
  --name wsi-ai-viewer \
  wsi-ai
```

CPU mode is useful for testing the viewer and API, but inference will be slower.

---

## Avoid Root-Owned Output Files

Docker may create files under `data/` as root. To avoid permission issues, run the container with your local user ID:

```bash
docker rm -f wsi-ai-viewer 2>/dev/null || true

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e SLIDE_ROOT=/app/data/raw/camelyon16/images \
  -e TIATOOLBOX_MODEL=resnet18-pcam \
  -e TIATOOLBOX_BATCH_SIZE=32 \
  -e TIATOOLBOX_DEVICE=cuda \
  --name wsi-ai-viewer \
  wsi-ai
```

If files were already created as root:

```bash
sudo chown -R "$USER:$USER" data/processed data/interim
```

---

## Run Tile Extraction and TIAToolbox Inference Manually

Extract tissue tiles:

```bash
python scripts/extract_tiles.py \
  --slide data/raw/camelyon16/images/tumor_005.tif
```

Run TIAToolbox inference:

```bash
python scripts/run_inference_tiatoolbox.py \
  --slide-id tumor_005 \
  --model resnet18-pcam \
  --batch-size 32 \
  --device cuda
```

Generate a static heatmap image if needed:

```bash
python scripts/generate_heatmap.py
```

Expected outputs:

```text
data/processed/tiles/tumor_005/
data/interim/tile_index/tumor_005_tile_index.csv
data/processed/predictions/tumor_005_predictions.csv
data/processed/heatmaps/tumor_005_overlay.png
```

---

## Example Output

```text
Using device: cuda
Found 300 tiles
Saved predictions to: data/processed/predictions/tumor_005_predictions.csv
Saved heatmap overlay to: data/processed/heatmaps/tumor_005_overlay.png
```

---

## Results and Observations

After introducing tissue-aware tile extraction and border filtering, the heatmap shifted from slide-edge artifacts to tissue-localized activations.

This improvement was driven by:

- excluding border regions during tile extraction
- building a tissue mask from the slide thumbnail
- sampling only tissue-positive tile locations
- exporting tile metadata for structured downstream processing

The project originally reused a model from the Bone-Fracture-Detection project, which was not pathology-specific. The current version uses a pathology-relevant TIAToolbox PCam model for tile-level inference on CAMELYON-style slides.

---

## Model Upgrade: Before vs After Heatmaps

![Heatmap before TIAToolbox upgrade](img/tumor_005_overlay_fracture_model.png)

*Before: ResNet from the Bone-Fracture-Detection project, with stronger edge and non-tissue artifacts.*

![Heatmap after TIAToolbox + PCam model](img/tumor_005_overlay_tiatoolbox_pcam.png)

*After: TIAToolbox PCam pathology model, with activations more concentrated in tissue regions and fewer obvious non-tissue artifacts.*

---

## Ground Truth vs Heatmap Comparison

The figure below compares the original WSI thumbnail, CAMELYON ground-truth polygon annotations, the model-generated heatmap, and a combined overlay. This provides a qualitative view of how inferred high-activation regions relate to annotated tumor regions.

![WSI comparison panel](img/test_001_panel_v2.png)

The figure below compares a zoomed original WSI thumbnail, CAMELYON ground-truth polygon annotations, the model-generated heatmap, and a combined overlay using the largest polygon region.

![WSI comparison panel zoomed](img/test_001_zoomed_panel.png)

---

## Data

Large datasets are excluded from this repository.

To reproduce results:

1. Download sample WSI slides
2. Run tile extraction
3. Run inference
4. Generate static heatmaps or use the live ZFP viewer

Expected local structure:

```text
data/
├── raw/
│   ├── camelyon16/images/
│   └── tcga/images/
├── interim/
│   ├── thumbnails/
│   ├── tissue_masks/
│   └── tile_index/
└── processed/
    ├── tiles/
    ├── predictions/
    └── heatmaps/
```

---

## Download Sample Data

```bash
./scripts/download_dataset.sh
```

For CAMELYON16 public S3 access, example files used in this project include:

```text
normal_019.tif
tumor_005.tif
test_001.tif
```

---

## Notes on Annotations

Annotation XML files are not required for the basic inference pipeline.

They are used for:

- ground-truth visualization
- qualitative heatmap comparison
- polygon overlay generation
- future supervised training
- future segmentation evaluation

The `metadata/` folder stores remote annotation listings from CAMELYON16. Specific XML annotation files are downloaded only when needed.

---

## Working with SVS Files

The viewer supports Aperio `.svs` files through OpenSlide.

Some tools, such as ImageMagick `identify`, may fail on certain SVS files because of compression formats such as JPEG2000 / YUV. That does not necessarily mean the file is invalid.

Use OpenSlide tools instead:

```bash
openslide-show-properties data/raw/tcga/images/example.svs
```

Useful properties include:

```text
openslide.vendor
openslide.level-count
openslide.level[0].width
openslide.level[0].height
openslide.mpp-x
openslide.mpp-y
openslide.objective-power
```

If `openslide-show-properties` works, the file should usually work in the ZFP viewer.

---

## Mobile and Remote Access

The viewer can be opened from another device on the same network, such as a phone or tablet:

```text
http://<linux-host-ip>:8000/viewer
```

This validates the zero-footprint design:

```text
WSI file stays on the Linux server
FastAPI streams visible tiles
Phone/tablet browser only requests tiles
No desktop WSI software is required on the client
```

Mobile mode uses:

- wrapped header controls
- hidden OpenSeadragon navigator
- bottom drawer for prediction inspection
- truncated status text
- mobile-friendly heatmap controls

Mobile viewing is useful for quick review, teaching, remote preview, and AI result inspection. It is not intended as a primary diagnostic viewer.

---

## Clinical Status

This project is a research and engineering prototype.

It is not clinically validated and should not be used for diagnosis.

The current TIAToolbox PCam model is more domain-aligned for CAMELYON-style lymph node metastasis slides than the earlier fracture-model baseline. However, model output should still be treated as experimental.

For external slides such as TCGA `.svs` files, the AI heatmap should be described as exploratory model visualization only.

---

## Troubleshooting

### Container name already in use

```bash
docker rm -f wsi-ai-viewer
```

Then rerun the container.

### Permission denied when deleting generated tiles

This usually happens when Docker created files as root.

Fix ownership:

```bash
sudo chown -R "$USER:$USER" data/processed data/interim
```

Then delete normally:

```bash
rm -rf data/processed/tiles/<slide_id>
```

### TIAToolbox / OpenCV error: `libGL.so.1` missing

If inference fails with:

```text
ImportError: libGL.so.1: cannot open shared object file
```

install the missing system libraries in the Dockerfile:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools \
    libopenslide0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*
```

Then rebuild:

```bash
docker build --no-cache -t wsi-ai .
```

### CUDA unavailable

Check GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

Then check PyTorch:

```bash
docker run --rm \
  --gpus all \
  -v "$(pwd)/data:/app/data" \
  wsi-ai \
  python -c "import torch; print(torch.cuda.is_available())"
```

### Heatmap does not show

Check that a prediction CSV exists:

```text
data/processed/predictions/{slide_id}_predictions.csv
```

Then check:

```bash
curl http://localhost:8000/slides/{slide_name}/heatmap/info
```

If no prediction CSV exists, run inference first.

---

## Use Cases

- Digital pathology AI prototyping
- Whole-slide image processing workflows
- Browser-based WSI viewer experiments
- AI heatmap visualization
- Model inference and visualization demos
- Teaching and educational demonstrations
- Remote zero-footprint WSI viewing
- Portfolio demonstration of AI + medical imaging engineering

---

## Roadmap

Planned future work includes:

- background inference jobs instead of synchronous `POST /infer`
- job status endpoint and progress polling
- pre-generated heatmap tile cache
- disk cache for frequently requested WSI tiles
- stronger model evaluation with accuracy, precision, recall, F1, Dice, and IoU
- polygon-supervised tumor-region segmentation
- U-Net / ResNet-UNet segmentation branch
- improved CAMELYON XML annotation handling
- DICOM WSI exploration
- optional Iris RESTful or other high-performance tile server experiments
- React frontend version with a richer case list and side panels

A future `feature/tumor-segmentation` branch may extend the current heatmap localization pipeline into polygon-supervised tumor-region segmentation using CAMELYON XML annotations, U-Net / ResNet-UNet models, and Dice/IoU evaluation.
