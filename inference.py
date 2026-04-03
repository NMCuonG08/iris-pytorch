from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.utils import load_yaml_config
from src.utils.visualization import load_segformer_checkpoint, predict_mask
from src.utils.visualize import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on test images and save boundary overlays")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="runs/iris-segformer/checkpoints/best.pt", help="Path to .pth/.pt model checkpoint")
    parser.add_argument("--test-dir", type=str, default="test", help="Folder containing test images")
    parser.add_argument("--num-images", type=int, default=10, help="Number of test images to run")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--boundary-output", type=str, default="inference_test_results", help="Folder to save boundary results")
    parser.add_argument("--overlay-output", type=str, default="overlay_test_results", help="Folder to save overlay results")
    return parser.parse_args()


def find_test_images(test_dir: Path) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(test_dir.glob(ext)))
    return files


def draw_boundary_overlay(image_bgr: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    contour_img = image_bgr.copy()
    mask_u8 = (pred_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_cfg = config["model"]
    data_cfg = config["data"]
    input_size = tuple(data_cfg.get("input_size", [512, 512]))

    model = load_segformer_checkpoint(
        checkpoint_path=args.checkpoint,
        backbone_name=model_cfg["backbone_name"],
        num_classes=model_cfg.get("num_classes", 1),
        decoder_channels=model_cfg.get("decoder_channels", 256),
        dropout=model_cfg.get("dropout", 0.1),
        device=args.device,
    )

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    boundary_dir = Path(args.boundary_output)
    overlay_dir = Path(args.overlay_output)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_test_images(test_dir)
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    selected = image_paths[: args.num_images]
    print(f"Running inference on {len(selected)} images from {test_dir}")

    for image_path in selected:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Skip unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        pred_small = predict_mask(
            model=model,
            image_rgb=image_rgb,
            input_size=input_size,
            device=args.device,
            threshold=args.threshold,
        )
        pred_mask = cv2.resize(pred_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        boundary_overlay = draw_boundary_overlay(image_bgr, pred_mask)
        boundary_path = boundary_dir / f"{image_path.stem}_boundary.png"
        cv2.imwrite(str(boundary_path), boundary_overlay)

        overlay_path = save_overlay(
            image_rgb=image_rgb,
            pred_mask=pred_mask,
            output_name=f"{image_path.stem}_overlay.png",
            output_dir=str(overlay_dir),
            alpha=0.5,
        )

        print(f"[OK] {image_path.name} -> {boundary_path.name}, {overlay_path.name}")

    print("Done")


if __name__ == "__main__":
    main()
