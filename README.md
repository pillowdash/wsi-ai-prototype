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

## Current Status

This prototype validates the technical pipeline for WSI processing and visualization.

It currently uses a model trained from a separate fracture detection project, which is not optimized for histopathology data. As a result, the generated heatmaps are not yet clinically meaningful and may highlight non-tissue artifacts.

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
