#!/usr/bin/env bash
set -euo pipefail

echo "=== WSI Dataset Download Script ==="

# Resolve project root (run from anywhere inside repo)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data/raw/camelyon16"
IMAGES_DIR="$DATA_DIR/images"
ANNOTATIONS_DIR="$DATA_DIR/annotations"
METADATA_DIR="$DATA_DIR/metadata"

# Create directories
mkdir -p "$IMAGES_DIR" "$ANNOTATIONS_DIR" "$METADATA_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Downloading to: $DATA_DIR"

# List available tumor slides (preview)
echo "Listing sample tumor slides..."
aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON16/images/ | grep tumor_ | head -n 10

# Download selected slides
echo "Downloading normal slide..."
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/normal_019.tif "$IMAGES_DIR/"

echo "Downloading tumor slide..."
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/tumor_001.tif "$IMAGES_DIR/"

# Download annotation (if available)
echo "Downloading annotation..."
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/annotations/tumor_001.xml "$ANNOTATIONS_DIR/" || echo "Annotation not found in this bucket (safe to ignore)."

# Save metadata listings
echo "Saving metadata listings..."
aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON16/images/ > "$METADATA_DIR/images_list.txt"

aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON16/annotations/ > "$METADATA_DIR/annotations_list.txt"

echo "=== Download complete ==="
