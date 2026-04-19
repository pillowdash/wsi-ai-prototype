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
  -v "/home/your-user/path/to/models:/app/models_external" \
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

It currently reuses a trained model from our [Bone-Fracture-Detection](https://github.com/pillowdash/Bone-Fracture-Detection) project. Since that model was trained for fracture detection on X-ray images rather than histopathology data, the current heatmaps are not yet clinically meaningful for pathology use and may emphasize non-tissue artifacts.

Future work will focus on replacing the current model with a pathology-specific model trained on WSI tile datasets (e.g., CAMELYON), along with improved tissue-region sampling and evaluation.

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
