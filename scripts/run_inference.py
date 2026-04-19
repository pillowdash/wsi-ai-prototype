from pathlib import Path
import re
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

TILES_DIR = Path("data/processed/tiles/tumor_005")
OUTPUT_DIR = Path("data/processed/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "tumor_005_predictions.csv"

# Update this path to your real trained checkpoint - not using Docker
# CHECKPOINT_PATH = Path("/home/pillowdash/git_projects/Bone-Fracture-Detection/outputs/models/best_model.pth")

# For Docker : Mount the model folder into the container and update the script to use the container path
# CHECKPOINT_PATH = Path("/app/models_external/best_model.pth")
import os
from pathlib import Path

CHECKPOINT_PATH = Path(
    os.environ.get("CHECKPOINT_PATH", "/app/models_external/best_model.pth")
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16

tile_pattern = re.compile(r".*_x(\d+)_y(\d+)\.png$")


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: Path) -> nn.Module:
    model = build_model(num_classes=2)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Common patterns for checkpoint structure
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def parse_coords(filename: str):
    match = tile_pattern.match(filename)
    if not match:
        raise ValueError(f"Could not parse coordinates from filename: {filename}")
    x = int(match.group(1))
    y = int(match.group(2))
    return x, y


def load_batch(image_paths, transform):
    images = []
    meta = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        x, y = parse_coords(path.name)

        images.append(tensor)
        meta.append({
            "tile_name": path.name,
            "x": x,
            "y": y,
        })

    batch_tensor = torch.stack(images).to(DEVICE)
    return batch_tensor, meta


def main():
    tile_paths = sorted(TILES_DIR.glob("*.png"))
    if not tile_paths:
        print(f"No tiles found in {TILES_DIR}")
        return

    if not CHECKPOINT_PATH.exists():
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("Update CHECKPOINT_PATH in this script.")
        return

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {CHECKPOINT_PATH}")
    print(f"Found {len(tile_paths)} tiles")

    model = load_model(CHECKPOINT_PATH)
    transform = get_transform()

    results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(tile_paths), BATCH_SIZE), desc="Running inference"):
            batch_paths = tile_paths[i:i + BATCH_SIZE]
            batch_tensor, batch_meta = load_batch(batch_paths, transform)

            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)

            for meta, prob_vec in zip(batch_meta, probs.cpu()):
                results.append({
                    "tile_name": meta["tile_name"],
                    "x": meta["x"],
                    "y": meta["y"],
                    "prob_class_0": float(prob_vec[0]),
                    "prob_class_1": float(prob_vec[1]),
                    "predicted_class": int(torch.argmax(prob_vec).item()),
                })

    df = pd.DataFrame(results).sort_values(by=["y", "x"])
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved predictions to: {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
