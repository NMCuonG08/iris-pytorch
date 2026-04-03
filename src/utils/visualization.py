from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models import SegFormerCustom


def load_segformer_checkpoint(
    checkpoint_path: str,
    backbone_name: str,
    num_classes: int,
    decoder_channels: int = 256,
    dropout: float = 0.1,
    device: str = "cpu",
) -> SegFormerCustom:
    model = SegFormerCustom(
        backbone_name=backbone_name,
        num_classes=num_classes,
        decoder_channels=decoder_channels,
        dropout=dropout,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_rgb: np.ndarray, input_size: Tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_mask(
    model: SegFormerCustom,
    image_rgb: np.ndarray,
    input_size: Tuple[int, int],
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    inputs = preprocess_image(image_rgb, input_size).to(device)
    logits = model(inputs)
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float().squeeze().cpu().numpy().astype(np.uint8)
    return mask


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    overlay = base.copy()
    color_arr = np.array(color, dtype=np.float32)

    mask_2d = mask.squeeze().astype(bool)
    overlay[mask_2d] = (1.0 - alpha) * overlay[mask_2d] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_comparison_figure(
    image_rgb: np.ndarray,
    ground_truth_mask: np.ndarray,
    prediction_mask: np.ndarray,
    alpha: float = 0.45,
    figsize: Tuple[int, int] = (18, 6),
):
    gt_overlay = _overlay_mask(image_rgb, ground_truth_mask, color=(255, 0, 0), alpha=alpha)
    pred_overlay = image_rgb.copy().astype(np.float32)
    gt_mask_2d = ground_truth_mask.squeeze().astype(bool)
    pred_mask_2d = prediction_mask.squeeze().astype(bool)

    pred_overlay[gt_mask_2d] = (1.0 - alpha) * pred_overlay[gt_mask_2d] + alpha * np.array([255, 0, 0], dtype=np.float32)
    pred_overlay[pred_mask_2d] = (1.0 - alpha) * pred_overlay[pred_mask_2d] + alpha * np.array([0, 255, 0], dtype=np.float32)
    pred_overlay = np.clip(pred_overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (image_rgb, "Original Image"),
        (gt_overlay, "Ground Truth (red)"),
        (pred_overlay, "Prediction (green) + GT (red)"),
    ]

    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    return fig


def save_comparison_figure(
    image_rgb: np.ndarray,
    ground_truth_mask: np.ndarray,
    prediction_mask: np.ndarray,
    output_path: str,
    alpha: float = 0.45,
) -> Path:
    fig = create_comparison_figure(image_rgb, ground_truth_mask, prediction_mask, alpha=alpha)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path