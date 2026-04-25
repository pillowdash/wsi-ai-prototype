from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

PREDICTIONS_CSV = Path("data/processed/predictions/test_001_predictions.csv")
THUMBNAIL_PATH = Path("data/interim/thumbnails/test_001_thumbnail.png")
OUTPUT_PATH = Path("data/processed/heatmaps/tumor_001_overlay.png")


# PREDICTIONS_CSV = Path("data/processed/predictions/tumor_005_predictions.csv")
# THUMBNAIL_PATH = Path("data/interim/thumbnails/tumor_005_thumbnail.png")
# OUTPUT_PATH = Path("data/processed/heatmaps/tumor_005_overlay.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 256
LEVEL = 2


def main() -> None:
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {PREDICTIONS_CSV}")

    if not THUMBNAIL_PATH.exists():
        raise FileNotFoundError(f"Thumbnail not found: {THUMBNAIL_PATH}")

    df = pd.read_csv(PREDICTIONS_CSV)

    if df.empty:
        raise ValueError("Predictions CSV is empty.")

    thumb = Image.open(THUMBNAIL_PATH).convert("RGB")
    thumb_arr = np.array(thumb)

    # Build score grid in LEVEL=2 coordinate space
    max_x = int(df["x"].max())
    max_y = int(df["y"].max())

    grid_w = max_x // TILE_SIZE + 1
    grid_h = max_y // TILE_SIZE + 1

    heatmap_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    count_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    for _, row in df.iterrows():
        gx = int(row["x"] // TILE_SIZE)
        gy = int(row["y"] // TILE_SIZE)
        score = float(row["prob_class_1"])

        heatmap_grid[gy, gx] += score
        count_grid[gy, gx] += 1.0

    mask = count_grid > 0
    heatmap_grid[mask] /= count_grid[mask]

    # Normalize to 0..1
    if heatmap_grid.max() > 0:
        heatmap_grid = heatmap_grid / heatmap_grid.max()

    # Convert to image and resize to thumbnail size
    heatmap_img = Image.fromarray((heatmap_grid * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(thumb.size, resample=Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img)

    # Plot overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(thumb_arr)
    plt.imshow(heatmap_arr, cmap="jet", alpha=0.40)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved heatmap overlay to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
