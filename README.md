## Overview

This project is a prototype for applying AI to Whole Slide Images (WSIs) in digital pathology.

It demonstrates an end-to-end pipeline that:

- Loads high-resolution pathology slides (SVS/TIFF) using OpenSlide
- Extracts tissue tiles from large WSIs
- Performs tile-level inference using a ResNet-based deep learning model (PyTorch)
- Aggregates predictions into region-level heatmaps
- Visualizes results as overlays on slide thumbnails

The goal is to simulate core components of a digital pathology AI system, including image ingestion, scalable tile processing, model inference, and visualization.

## Key Features

- WSI ingestion with OpenSlide
- Tile-based processing for large gigapixel images
- GPU-accelerated inference using PyTorch
- Heatmap generation for region-level interpretation
- Modular and extensible project structure

## Docker

The project can be containerized for reproducible execution. Large datasets and model checkpoints are excluded from the image and should be mounted at runtime.

Example:

```bash
docker build -t wsi-ai .

# configure Docker for NVIDIA runtime:
sudo nvidia-ctk runtime configure --runtime=docker
# Restart Docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "/home/pillowdash/git_projects/Bone-Fracture-Detection/outputs/models:/app/models_external" \
  -e CHECKPOINT_PATH=/app/models_external/best_model.pth \
  wsi-ai
```

### Verify GPU inside Docker

```bash
docker run --rm \
  --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "/path/to/your/models:/app/models_external" \
  wsi-ai \
  python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

```md
Example output:

==========
== CUDA ==
==========

CUDA Version 12.1.1

2.5.1+cu121
12.1
True
NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

```md
This confirms that:

- CUDA is available inside the container
- PyTorch is using GPU acceleration
- The container is correctly configured for NVIDIA runtime
```

```md
run to execute the inference script:
```

```bash
docker run --rm \
  --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "/path/to/your/models:/app/models_external" \
  -e CHECKPOINT_PATH=/app/models_external/best_model.pth \
  wsi-ai \
  python scripts/run_inference.py --input_dir /app/data/raw --output_dir /app/data/processed
```

```md
enter an interactive shell for debugging:
```

```bash
docker run --rm --gpus all -it -v ... wsi-ai bash
```

## Example Output

- Using device: cuda
- Found 300 tiles
- Saved predictions to: `data/processed/predictions/tumor_005_predictions.csv`
- Saved heatmap overlay to: `data/processed/heatmaps/tumor_005_overlay.png`

### Run inference

```bash
python scripts/run_inference.py
python scripts/generate_heatmap.py
```

## Results and Observations

After introducing tissue-aware tile extraction and border filtering, the heatmap shifted from slide-edge artifacts to tissue-localized activations. This shows that the pipeline is correctly focusing inference on biologically relevant regions, even though the current model is not yet trained on histopathology data.

This improvement was driven by:

- excluding border regions during tile extraction
- building a tissue mask from the slide thumbnail
- sampling only tissue-positive tile locations
- exporting tile metadata for more structured downstream processing

## Current Status

This prototype validates the technical pipeline for WSI processing and visualization.

UPDATE : This prototype now uses a pathology-specific pretrained patch classification model from TIAToolbox (PCam) for tile-level inference on CAMELYON slides. Compared with the earlier fracture-trained baseline, this improves the domain relevance of the generated heatmaps for digital pathology workflows.

The current pipeline still uses a simplified tile extraction and aggregation strategy, so future work will focus on stronger pathology-specific training, evaluation, and improved region-level visualization.
~~It currently reuses a trained model from our [Bone-Fracture-Detection](https://github.com/pillowdash/Bone-Fracture-Detection) project. Since that model was trained for fracture detection on X-ray images rather than histopathology data, the current heatmaps are not yet clinically meaningful for pathology use and may emphasize non-tissue artifacts.~~

Future work will focus on replacing the current model with a pathology-specific model trained on WSI tile datasets (e.g., CAMELYON), along with improved tissue-region sampling and evaluation.

### Model Upgrade: Before vs After Heatmaps

![Heatmap before TIAToolbox upgrade](img/tumor_005_overlay_usingBoneFractureDectionTraning.png)
*Before: ResNet from Bone-Fracture-Detection project (note edge/non-tissue artifacts).*

![Heatmap after TIAToolbox + PCam model](images/tumor_005_overlay_usingTIAToolbox.png)
*After: TIAToolbox PCam model — predictions are now focused on relevant histopathology features.*

## Use Cases

- Digital pathology AI prototyping
- Whole-slide image processing workflows
- Model inference and visualization experiments
- Educational demonstrations of WSI-based AI systems

## Data

Large datasets (WSIs, extracted tiles, and model checkpoints) are excluded from this repository.

To reproduce results:

1. Download sample slides
2. Run tile extraction
3. Run inference
4. Generate heatmap overlays

## Download Sample Data

```bash
./scripts/download_dataset.sh
```

## API Usage

The project includes a lightweight **FastAPI** backend (`api/app.py`) that exposes a REST endpoint for WSI inference. This demonstrates how the pipeline could be served in a clinical IMS for on-demand AI predictions and heatmap generation.

### 1. Build and Run the API with Docker (GPU-enabled)

```bash
# Build the image (once)
docker build -t wsi-ai .

# Run the API server
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "/absolute/path/to/your/models:/app/models_external:ro" \
  -e CHECKPOINT_PATH=/app/models_external/best_model.pth \
  --name wsi-ai-api \
  wsi-ai
```

### 2.Access the interative Swagger UI:
http://localhost:8000/docs

### 3.Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"slide_name": "tumor_005.tif"}'
```

### 4.Example response:

```bash
{
  "slide_name": "tumor_005.tif",
  "predictions_csv": "data/processed/predictions/tumor_005_predictions.csv",
  "heatmap_image": "data/processed/heatmaps/tumor_005_overlay.png",
  "tiles_processed": 877,
  "status": "completed in 12.4s"
}
```
