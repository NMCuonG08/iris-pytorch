from pathlib import Path
import random

from torch.utils.data import DataLoader

from src.data.transforms import build_train_transforms, build_val_transforms
from src.data.ubiris_dataset import UBIRISV2Dataset


def split_stems(stems: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_ratio}")

    shuffled = list(stems)
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))

    train_stems = sorted(shuffled[:split_idx])
    val_stems = sorted(shuffled[split_idx:])
    return train_stems, val_stems


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]

    root_dir = Path(data_cfg["root_dir"])
    image_dir = root_dir / data_cfg["image_dir"]
    image_ext = data_cfg.get("image_ext", ".jpg")

    print(f"[DEBUG] Loading from: {image_dir}")
    print(f"[DEBUG] Image extension: {image_ext}")
    stems = sorted([p.stem for p in image_dir.glob(f"*{image_ext}")])
    print(f"[DEBUG] Found {len(stems)} images")
    if not stems:
        raise ValueError(f"No training images found in {image_dir}")

    train_stems, val_stems = split_stems(
        stems=stems,
        train_ratio=data_cfg.get("train_split", 0.8),
        seed=train_cfg.get("seed", 42),
    )
    print(f"[DEBUG] Train: {len(train_stems)}, Val: {len(val_stems)}")

    input_size = tuple(data_cfg.get("input_size", [512, 512]))
    train_dataset = UBIRISV2Dataset(
        root_dir=str(root_dir),
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        image_ext=data_cfg.get("image_ext", ".jpg"),
        mask_ext=data_cfg.get("mask_ext", ".png"),
        transform=build_train_transforms(input_size),
        file_stems=train_stems,
    )

    val_dataset = UBIRISV2Dataset(
        root_dir=str(root_dir),
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        image_ext=data_cfg.get("image_ext", ".jpg"),
        mask_ext=data_cfg.get("mask_ext", ".png"),
        transform=build_val_transforms(input_size),
        file_stems=val_stems,
    )

    print(
        f"[DEBUG] Creating train DataLoader with batch_size={train_cfg.get('batch_size', 8)}, "
        f"num_workers={data_cfg.get('num_workers', 4)}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    print("[DEBUG] Train DataLoader ready")

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    return train_loader, val_loader
