from pathlib import Path
import argparse
import re
import pandas as pd

from tiatoolbox.models.engine.patch_predictor import PatchPredictor

tile_pattern = re.compile(r".*_x(\d+)_y(\d+)\.png$")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TIAToolbox pathology inference on extracted tiles.")
    parser.add_argument(
        "--slide-id",
        required=True,
        help="Slide ID without extension, e.g. tumor_005 or normal_019",
    )
    parser.add_argument(
        "--model",
        default="resnet18-pcam",
        help="TIAToolbox pretrained model name, e.g. resnet18-pcam or resnet50-pcam",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for patch inference",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device to use: "cuda" or "cpu"',
    )
    return parser.parse_args()


def parse_coords(filename: str):
    match = tile_pattern.match(filename)
    if not match:
        raise ValueError(f"Could not parse coordinates from filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def main():
    args = parse_args()

    tiles_dir = Path(f"data/processed/tiles/{args.slide_id}")
    output_dir = Path("data/processed/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.slide_id}_predictions.csv"

    tile_paths = sorted(tiles_dir.glob("*.png"))
    if not tile_paths:
        print(f"No tiles found in {tiles_dir}")
        return

    tile_paths_str = [str(p) for p in tile_paths]

    print(f"Using TIAToolbox model: {args.model}")
    print(f"Using device: {args.device}")
    print(f"Found {len(tile_paths)} tiles")

    predictor = PatchPredictor(
        model=args.model,
        batch_size=args.batch_size,
        num_workers=0,
        device=args.device,
        verbose=True,
    )

    output = predictor.run(
        tile_paths_str,
        patch_mode=True,
        return_probabilities=True,
    )

    # TIAToolbox patch-mode output includes predictions and probabilities
    predictions = output["predictions"]
    probabilities = output["probabilities"]

    rows = []
    for tile_path, pred, prob_vec in zip(tile_paths, predictions, probabilities):
        x, y = parse_coords(tile_path.name)

        # Binary PCam models produce 2-class probabilities
        prob_class_0 = float(prob_vec[0])
        prob_class_1 = float(prob_vec[1])
        predicted_class = int(pred)

        rows.append(
            {
                "tile_name": tile_path.name,
                "x": x,
                "y": y,
                "prob_class_0": prob_class_0,
                "prob_class_1": prob_class_1,
                "predicted_class": predicted_class,
            }
        )

    df = pd.DataFrame(rows).sort_values(by=["y", "x"])
    df.to_csv(output_csv, index=False)

    print(f"Saved predictions to: {output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()
