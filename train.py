import argparse
from pathlib import Path

import torch

from src.data import create_dataloaders
from src.losses import CombinedDiceFocalLoss
from src.models import SegFormerCustom
from src.training import Trainer
from src.utils import load_yaml_config, set_seed
from src.utils.logger import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer-based iris segmentation model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[DEBUG] Loading config from: {args.config}")
    config = load_yaml_config(args.config)
    print(f"[DEBUG] Config loaded")

    output_dir = Path(config["project"]["output_dir"])
    logger = build_logger(str(output_dir / "logs"), name="trainer")

    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info("Using seed: %d", seed)

    print(f"[DEBUG] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"[DEBUG] Dataloaders created")

    model_cfg = config["model"]
    model = SegFormerCustom(
        backbone_name=model_cfg["backbone_name"],
        num_classes=model_cfg["num_classes"],
        decoder_channels=model_cfg.get("decoder_channels", 256),
        dropout=model_cfg.get("dropout", 0.1),
    )

    loss_cfg = config["loss"]
    criterion = CombinedDiceFocalLoss(
        dice_weight=loss_cfg.get("dice_weight", 0.6),
        focal_weight=loss_cfg.get("focal_weight", 0.4),
        focal_alpha=loss_cfg.get("focal_alpha", 0.8),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("lr", 1e-4),
        weight_decay=config["training"].get("weight_decay", 1e-5),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
