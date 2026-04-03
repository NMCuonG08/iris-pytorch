from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Add project root to path so we can import src.* modules when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_yaml_config
from src.utils.visualization import load_segformer_checkpoint, predict_mask, save_comparison_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SegFormer prediction versus ground truth")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--checkpoint", type=str, default="runs/iris-segformer/checkpoints/best.pt", help="Checkpoint path")
    parser.add_argument("--image", type=str, default="", help="Path to an image file")
    parser.add_argument("--mask", type=str, default="", help="Path to the matching mask file")
    parser.add_argument("--stem", type=str, default="", help="File stem inside dataset/image_dir and dataset/mask_dir")
    parser.add_argument("--output", type=str, default="runs/iris-segformer/visualizations/comparison.png", help="Output figure path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for prediction mask")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    return parser.parse_args()


def _resolve_paths(config: dict, args: argparse.Namespace) -> tuple[Path, Path]:
    if args.image and args.mask:
        return Path(args.image), Path(args.mask)

    if not args.stem:
        raise ValueError("Provide either --image and --mask or --stem")

    data_cfg = config["data"]
    root_dir = Path(data_cfg.get("root_dir", "dataset"))
    image_dir = root_dir / data_cfg.get("image_dir", "images")
    mask_dir = root_dir / data_cfg.get("mask_dir", "masks")
    image_ext = data_cfg.get("image_ext", ".jpg")
    mask_ext = data_cfg.get("mask_ext", ".png")

    return image_dir / f"{args.stem}{image_ext}", mask_dir / f"{args.stem}{mask_ext}"


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    image_path, mask_path = _resolve_paths(config, args)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    data_cfg = config["data"]
    model_cfg = config["model"]
    input_size = tuple(data_cfg.get("input_size", [512, 512]))

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if mask_gray is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ground_truth_mask = (mask_gray > 127).astype("uint8")

    model = load_segformer_checkpoint(
        checkpoint_path=args.checkpoint,
        backbone_name=model_cfg["backbone_name"],
        num_classes=model_cfg.get("num_classes", 1),
        decoder_channels=model_cfg.get("decoder_channels", 256),
        dropout=model_cfg.get("dropout", 0.1),
        device=args.device,
    )
    prediction_mask = predict_mask(
        model=model,
        image_rgb=image_rgb,
        input_size=input_size,
        device=args.device,
        threshold=args.threshold,
    )

    output_path = save_comparison_figure(
        image_rgb=image_rgb,
        ground_truth_mask=ground_truth_mask,
        prediction_mask=prediction_mask,
        output_path=args.output,
    )
    print(f"Saved comparison figure to: {output_path}")


if __name__ == "__main__":
    main()