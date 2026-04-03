from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import create_dataloaders
from src.evaluation import ModelEvaluator
from src.models import SegFormerCustom
from src.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained iris segmentation model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--checkpoint", type=str, default="runs/iris-segformer/checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="runs/iris-segformer/evaluation", help="Directory to save evaluation outputs")
    parser.add_argument("--iou-threshold", type=float, default=0.8, help="IoU threshold for failed cases")
    parser.add_argument("--save-predictions", action="store_true", help="Save prediction visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    _, val_loader = create_dataloaders(config)

    model_cfg = config["model"]
    model = SegFormerCustom(
        backbone_name=model_cfg["backbone_name"],
        num_classes=model_cfg["num_classes"],
        decoder_channels=model_cfg.get("decoder_channels", 256),
        dropout=model_cfg.get("dropout", 0.1),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    requested_device = config["training"].get("device", "cuda")
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    evaluator = ModelEvaluator(
        model=model,
        device=device,
        num_classes=2,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    results = evaluator.evaluate(
        dataloader=val_loader,
        save_failed_cases=True,
        iou_threshold=args.iou_threshold,
    )

    speed = evaluator.benchmark_speed(input_size=(1, 3, 512, 512), num_runs=50)
    results["speed"] = speed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete")
    print(results["evaluation_summary"])
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
