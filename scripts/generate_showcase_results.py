from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import SegFormerCustom
from src.utils import load_yaml_config
from src.utils.visualization import load_segformer_checkpoint, preprocess_image
from src.utils.visualize import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate before/after training showcase images")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/iris-segformer/checkpoints/best.pt")
    parser.add_argument("--test-dir", type=str, default="test")
    parser.add_argument("--num-batch", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--skip-augmentations",
        action="store_true",
        help="Skip generating augmentation_visualizations",
    )
    return parser.parse_args()


def _find_images(root: Path) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(root.glob(ext)))
    return files


def _predict_logits(model: torch.nn.Module, image_rgb: np.ndarray, input_size: tuple[int, int], device: str) -> np.ndarray:
    inp = preprocess_image(image_rgb, input_size).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    return probs


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask * 255).astype(np.uint8))


def _save_prob_map(path: Path, prob: np.ndarray) -> None:
    p = np.clip(prob, 0.0, 1.0)
    heat = (p * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), heat)


def _save_boundary(path: Path, image_bgr: np.ndarray, mask: np.ndarray) -> None:
    canvas = image_bgr.copy()
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def _save_comparison(path: Path, image_rgb: np.ndarray, before_mask: np.ndarray, after_mask: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(before_mask, cmap="gray")
    axes[1].set_title("Before Train")
    axes[1].axis("off")

    axes[2].imshow(after_mask, cmap="gray")
    axes[2].set_title("After Train")
    axes[2].axis("off")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _apply_overlay_style(image_rgb: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image_rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    color = np.array(color_rgb, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_overlay_examples(path: Path, image_rgb: np.ndarray, mask: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)

    clinical_red = _apply_overlay_style(image_rgb, mask, (255, 64, 64), 0.45)
    subtle_blue = _apply_overlay_style(image_rgb, mask, (70, 145, 255), 0.30)
    high_contrast = _apply_overlay_style(image_rgb, mask, (255, 215, 0), 0.65)
    scientific = _apply_overlay_style(image_rgb, mask, (40, 220, 140), 0.40)

    iris_only = np.zeros_like(image_rgb)
    iris_only[mask.astype(bool)] = image_rgb[mask.astype(bool)]

    cv2.imwrite(str(path / "clinical_red_overlay.png"), cv2.cvtColor(clinical_red, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(path / "subtle_blue_overlay.png"), cv2.cvtColor(subtle_blue, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(path / "high_contrast_overlay.png"), cv2.cvtColor(high_contrast, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(path / "scientific_overlay.png"), cv2.cvtColor(scientific, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(path / "iris_only_overlay.png"), cv2.cvtColor(iris_only, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    tiles = [
        ("Original", image_rgb),
        ("Clinical Red", clinical_red),
        ("Subtle Blue", subtle_blue),
        ("Scientific", scientific),
        ("High Contrast", high_contrast),
        ("Iris Only", iris_only),
    ]
    for ax, (title, img) in zip(axes.flatten(), tiles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path / "overlay_styles_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    compare = [
        ("Original", image_rgb),
        ("Clinical Red", clinical_red),
        ("Scientific", scientific),
        ("Iris Only", iris_only),
    ]
    for ax, (title, img) in zip(axes, compare):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path / "comprehensive_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    input_size = tuple(data_cfg.get("input_size", [512, 512]))

    root = Path(__file__).resolve().parents[1]
    aug_dir = root / "augmentation_visualizations"
    inference_dir = root / "inference_test_results"
    overlay_dir = root / "overlay_test_results"
    overlay_examples_dir = root / "overlay_examples"
    batch_dir = root / "batch_test_results"

    for folder in [aug_dir, inference_dir, overlay_dir, overlay_examples_dir, batch_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    print("Running with arguments:")
    print(f"- config: {args.config}")
    print(f"- checkpoint: {args.checkpoint}")
    print(f"- test_dir: {args.test_dir}")
    print(f"- num_batch: {args.num_batch}")
    print(f"- threshold: {args.threshold}")
    print(f"- device: {args.device}")
    print(f"- skip_augmentations: {args.skip_augmentations}")

    if not args.skip_augmentations:
        aug_script = root / "scripts" / "visualize_augmentations.py"
        cmd = [
            sys.executable,
            str(aug_script),
            "--output-dir",
            "augmentation_visualizations",
        ]
        subprocess.run(cmd, cwd=str(root), check=True)

    # Select source images: test/ first, fallback to dataset/images
    test_dir = (root / args.test_dir).resolve()
    images = _find_images(test_dir) if test_dir.exists() else []
    if not images:
        images = _find_images(root / "dataset" / "images")
    if not images:
        raise FileNotFoundError("No images found in test/ or dataset/images")

    image_path = images[0]
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # Before-train model (same architecture, random init)
    before_model = SegFormerCustom(
        backbone_name=model_cfg["backbone_name"],
        num_classes=model_cfg.get("num_classes", 1),
        decoder_channels=model_cfg.get("decoder_channels", 256),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(args.device)
    before_model.eval()

    before_prob_small = _predict_logits(before_model, image_rgb, input_size, args.device)
    before_prob = cv2.resize(before_prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
    before_mask = (before_prob > args.threshold).astype(np.uint8)

    # After-train model
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    after_model = load_segformer_checkpoint(
        checkpoint_path=args.checkpoint,
        backbone_name=model_cfg["backbone_name"],
        num_classes=model_cfg.get("num_classes", 1),
        decoder_channels=model_cfg.get("decoder_channels", 256),
        dropout=model_cfg.get("dropout", 0.1),
        device=args.device,
    )
    after_prob_small = _predict_logits(after_model, image_rgb, input_size, args.device)
    after_prob = cv2.resize(after_prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
    after_mask = (after_prob > args.threshold).astype(np.uint8)

    # inference_test_results
    _save_mask(inference_dir / "before_train_mask.png", before_mask)
    _save_prob_map(inference_dir / "before_train_iris_prob.png", before_prob)
    _save_boundary(inference_dir / "before_train_boundary.png", image_bgr, before_mask)

    _save_mask(inference_dir / "test_prediction_mask.png", after_mask)
    _save_prob_map(inference_dir / "test_prediction_iris_prob.png", after_prob)
    _save_boundary(inference_dir / "test_prediction_boundary.png", image_bgr, after_mask)

    # overlay_test_results
    save_overlay(image_rgb, before_mask, "before_train_overlay.png", output_dir=str(overlay_dir), alpha=0.5)
    save_overlay(image_rgb, after_mask, "full_prediction_overlay.png", output_dir=str(overlay_dir), alpha=0.5)
    _save_comparison(overlay_dir / "comparison.png", image_rgb, before_mask, after_mask)

    # overlay_examples
    _save_overlay_examples(overlay_examples_dir, image_rgb, after_mask)

    # batch_test_results
    batch_images = images[: max(1, args.num_batch)]
    lines = []
    for i, p in enumerate(batch_images, start=1):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hh, ww = rgb.shape[:2]
        prob_small = _predict_logits(after_model, rgb, input_size, args.device)
        prob = cv2.resize(prob_small, (ww, hh), interpolation=cv2.INTER_LINEAR)
        mask = (prob > args.threshold).astype(np.uint8)

        stem = p.stem
        _save_mask(batch_dir / f"result_{i:03d}_{stem}_mask.png", mask)
        _save_prob_map(batch_dir / f"result_{i:03d}_{stem}_iris_prob.png", prob)
        _save_boundary(batch_dir / f"result_{i:03d}_{stem}_boundary.png", bgr, mask)

        lines.append(f"{i:03d} | {stem} | mean_prob={prob.mean():.4f} | fg_ratio={mask.mean():.4f}")

    with (batch_dir / "batch_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Batch prediction summary\n")
        f.write("=" * 60 + "\n")
        for line in lines:
            f.write(line + "\n")

    print("Showcase results generated:")
    print(f"- {aug_dir}")
    print(f"- {inference_dir}")
    print(f"- {overlay_dir}")
    print(f"- {overlay_examples_dir}")
    print(f"- {batch_dir}")


if __name__ == "__main__":
    main()
