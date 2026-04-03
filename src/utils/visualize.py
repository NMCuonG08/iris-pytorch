from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_display_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        # Assume normalized with ImageNet stats when tensor comes from Albumentations pipeline.
        image_np = image_np * _IMAGENET_STD + _IMAGENET_MEAN
    else:
        image_np = image.astype(np.float32)
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
    return np.clip(image_np, 0.0, 1.0)


def _to_mask_np(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
        if m.ndim == 3:
            m = m[0]
    else:
        m = mask
    return (m > 0.5).astype(np.uint8)


def visualize_dataset(
    dataloader,
    num_samples: int = 5,
    output_dir: str = "augmentation_visualizations",
    file_name: str = "dataset_preview.png",
    seed: int = 42,
) -> Path:
    """Sample pairs from DataLoader dataset, apply augmentation, and save side-by-side visualization."""
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise ValueError("dataloader must have a dataset attribute")

    if not hasattr(dataset, "_load_pair") or not hasattr(dataset, "file_stems"):
        raise ValueError("dataset must provide _load_pair(stem) and file_stems")

    total = len(dataset)
    if total == 0:
        raise ValueError("dataset is empty")

    rng = random.Random(seed)
    chosen = rng.sample(range(total), k=min(num_samples, total))

    fig, axes = plt.subplots(len(chosen), 4, figsize=(16, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    for row, idx in enumerate(chosen):
        stem = dataset.file_stems[idx]
        raw_image, raw_mask = dataset._load_pair(stem)

        if dataset.transform is not None:
            transformed = dataset.transform(image=raw_image, mask=raw_mask)
            aug_image = transformed["image"]
            aug_mask = transformed["mask"]
        else:
            aug_image = raw_image
            aug_mask = raw_mask

        raw_img_disp = _to_display_image(raw_image)
        aug_img_disp = _to_display_image(aug_image)
        raw_mask_disp = _to_mask_np(raw_mask)
        aug_mask_disp = _to_mask_np(aug_mask)

        axes[row, 0].imshow(raw_img_disp)
        axes[row, 0].set_title(f"Raw Image ({stem})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(raw_mask_disp, cmap="gray")
        axes[row, 1].set_title("Raw Mask")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(aug_img_disp)
        axes[row, 2].set_title("Augmented Image")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(aug_mask_disp, cmap="gray")
        axes[row, 3].set_title("Augmented Mask")
        axes[row, 3].axis("off")

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_overlay(
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    output_name: str,
    output_dir: str = "overlay_test_results",
    alpha: float = 0.5,
) -> Path:
    """Overlay predicted mask on original image and save to overlay_test_results/."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = image_rgb.astype(np.float32).copy()
    if image.max() > 1.0:
        image = image / 255.0

    mask_bool = pred_mask.astype(bool)
    color = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # green

    image[mask_bool] = (1.0 - alpha) * image[mask_bool] + alpha * color

    out_img = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    out_path = out_dir / output_name
    cv2.imwrite(str(out_path), out_bgr)
    return out_path
