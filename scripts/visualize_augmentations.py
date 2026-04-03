from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.transforms import build_train_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmentation visualization artifacts")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    parser.add_argument("--mask", type=str, default="", help="Path to mask image")
    parser.add_argument("--size", type=int, default=512, help="Resize size (square)")
    parser.add_argument("--samples", type=int, default=6, help="Number of full-pipeline samples")
    parser.add_argument("--output-dir", type=str, default="augmentation_visualizations", help="Output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--create-extra-folders",
        action="store_true",
        help="Also create batch_test_results/inference_test_results/overlay_examples/overlay_test_results",
    )
    return parser.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (gray > 127).astype(np.uint8)


def _denormalize_tensor(t: torch.Tensor) -> np.ndarray:
    # t: [3, H, W]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0.0, 1.0)


def _overlay_mask(image_float: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    overlay = image_float.copy()
    m = mask.astype(bool)
    alpha = 0.35
    overlay[m] = (1.0 - alpha) * overlay[m] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def _boundary_from_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return ((dilated - eroded) > 0).astype(np.uint8)


def _resolve_default_paths(root: Path) -> tuple[Path, Path]:
    images_dir = root / "dataset" / "images"
    masks_dir = root / "dataset" / "masks"

    preferred_img = images_dir / "C1_S1_I1.tiff"
    preferred_mask = masks_dir / "OperatorA_C1_S1_I1.tiff"
    if preferred_img.exists() and preferred_mask.exists():
        return preferred_img, preferred_mask

    image_candidates = sorted(images_dir.glob("*.tiff"))
    if not image_candidates:
        image_candidates = sorted(images_dir.glob("*.png"))
    if not image_candidates:
        raise FileNotFoundError(f"No images found in {images_dir}")

    img_path = image_candidates[0]
    stem = img_path.stem

    mask_candidates = [
        masks_dir / f"OperatorA_{stem}.tiff",
        masks_dir / f"{stem}.tiff",
        masks_dir / f"OperatorA_{stem}.png",
        masks_dir / f"{stem}.png",
    ]

    for p in mask_candidates:
        if p.exists():
            return img_path, p

    any_masks = sorted(list(masks_dir.glob("*.tiff")) + list(masks_dir.glob("*.png")))
    if not any_masks:
        raise FileNotFoundError(f"No masks found in {masks_dir}")

    return img_path, any_masks[0]


def save_individual_augmentations(image: np.ndarray, mask: np.ndarray, size: int, save_path: Path) -> None:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transforms = {
        "Original": A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()]),
        "Horizontal Flip": A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
        "Vertical Flip": A.Compose([A.Resize(size, size), A.VerticalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
        "Rotation": A.Compose([A.Resize(size, size), A.Rotate(limit=(12, 12), p=1.0), A.Normalize(), ToTensorV2()]),
        "Brightness/Contrast": A.Compose([
            A.Resize(size, size),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ]),
        "CLAHE": A.Compose([A.Resize(size, size), A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), A.Normalize(), ToTensorV2()]),
        "Gaussian Blur": A.Compose([A.Resize(size, size), A.GaussianBlur(blur_limit=(3, 5), p=1.0), A.Normalize(), ToTensorV2()]),
        "Motion Blur": A.Compose([A.Resize(size, size), A.MotionBlur(blur_limit=(3, 5), p=1.0), A.Normalize(), ToTensorV2()]),
        "Noise": A.Compose([A.Resize(size, size), A.GaussNoise(std_range=(0.03, 0.08), p=1.0), A.Normalize(), ToTensorV2()]),
    }

    cols = 3
    rows = int(np.ceil(len(transforms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (name, tfm) in enumerate(transforms.items()):
        out = tfm(image=image, mask=mask)
        img = _denormalize_tensor(out["image"])
        m = out["mask"].detach().cpu().numpy()
        overlay = _overlay_mask(img, m, (1.0, 0.0, 0.0))

        axes[i].imshow(overlay)
        axes[i].set_title(name, fontsize=11)
        axes[i].axis("off")

    for j in range(len(transforms), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_progressive_augmentation(image: np.ndarray, mask: np.ndarray, size: int, save_path: Path) -> None:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    stages = [
        (
            "Original",
            A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()]),
        ),
        (
            "+ Geometry",
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=0, p=0.5),
                A.Resize(size, size),
                A.Normalize(),
                ToTensorV2(),
            ]),
        ),
        (
            "+ Color",
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.6),
                A.Resize(size, size),
                A.Normalize(),
                ToTensorV2(),
            ]),
        ),
        (
            "+ Enhancement",
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
                A.Resize(size, size),
                A.Normalize(),
                ToTensorV2(),
            ]),
        ),
        (
            "+ Blur/Noise",
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
                A.OneOf([A.MotionBlur(blur_limit=5, p=1.0), A.GaussianBlur(blur_limit=(3, 7), p=1.0)], p=0.35),
                A.GaussNoise(std_range=(0.03, 0.08), p=0.25),
                A.Resize(size, size),
                A.Normalize(),
                ToTensorV2(),
            ]),
        ),
    ]

    fig, axes = plt.subplots(2, len(stages), figsize=(4.2 * len(stages), 8))

    for i, (name, tfm) in enumerate(stages):
        out = tfm(image=image, mask=mask)
        img = _denormalize_tensor(out["image"])
        m = out["mask"].detach().cpu().numpy()

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{name}\nImage", fontsize=11)
        axes[0, i].axis("off")

        axes[1, i].imshow(m, cmap="gray")
        axes[1, i].set_title(f"{name}\nMask", fontsize=11)
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_boundary_visualization(image: np.ndarray, mask: np.ndarray, save_path: Path) -> None:
    ksizes = [1, 3, 5, 7]

    fig, axes = plt.subplots(2, len(ksizes) + 1, figsize=(4 * (len(ksizes) + 1), 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(mask, cmap="gray")
    axes[1, 0].set_title("Original Mask")
    axes[1, 0].axis("off")

    for i, k in enumerate(ksizes, start=1):
        boundary = _boundary_from_mask(mask, k)
        overlay = image.copy()
        overlay[boundary == 1] = [255, 0, 0]

        axes[0, i].imshow(overlay)
        axes[0, i].set_title(f"Boundary (k={k})")
        axes[0, i].axis("off")

        axes[1, i].imshow(boundary, cmap="gray")
        axes[1, i].set_title(f"Boundary Mask (k={k})")
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _letterbox(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)

    img_rs = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    msk_rs = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.zeros((size, size, 3), dtype=np.uint8)
    canvas_msk = np.zeros((size, size), dtype=np.uint8)

    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas_img[y0 : y0 + nh, x0 : x0 + nw] = img_rs
    canvas_msk[y0 : y0 + nh, x0 : x0 + nw] = msk_rs
    return canvas_img, canvas_msk


def save_aspect_ratio_comparison(image: np.ndarray, mask: np.ndarray, size: int, save_path: Path) -> None:
    standard_img = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    standard_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    aspect_img, aspect_mask = _letterbox(image, mask, size)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Original\n{image.shape[1]}x{image.shape[0]}")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(mask, cmap="gray")
    axes[1, 0].set_title("Original Mask")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(standard_img)
    axes[0, 1].set_title(f"Standard Resize\n{size}x{size}")
    axes[0, 1].axis("off")
    axes[1, 1].imshow(standard_mask, cmap="gray")
    axes[1, 1].set_title("Standard Mask")
    axes[1, 1].axis("off")

    axes[0, 2].imshow(aspect_img)
    axes[0, 2].set_title(f"Aspect Preserving\n{size}x{size}")
    axes[0, 2].axis("off")
    axes[1, 2].imshow(aspect_mask, cmap="gray")
    axes[1, 2].set_title("Aspect Preserving Mask")
    axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_full_pipeline_samples(image: np.ndarray, mask: np.ndarray, size: int, samples: int, save_path: Path) -> None:
    tfm = build_train_transforms((size, size))

    cols = 3
    rows = int(np.ceil(samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(samples):
        out = tfm(image=image, mask=mask)
        img = _denormalize_tensor(out["image"])
        m = out["mask"].detach().cpu().numpy()
        overlay = _overlay_mask(img, m, (1.0, 0.0, 0.0))

        axes[i].imshow(overlay)
        axes[i].set_title(f"Sample {i + 1}")
        axes[i].axis("off")

    for j in range(samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.create_extra_folders:
        for folder in [
            "batch_test_results",
            "inference_test_results",
            "overlay_examples",
            "overlay_test_results",
        ]:
            (root / folder).mkdir(parents=True, exist_ok=True)

    if args.image and args.mask:
        image_path = (root / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
        mask_path = (root / args.mask).resolve() if not Path(args.mask).is_absolute() else Path(args.mask)
    else:
        image_path, mask_path = _resolve_default_paths(root)

    image = _load_rgb(image_path)
    mask = _load_mask(mask_path)

    save_individual_augmentations(image, mask, args.size, out_dir / "individual_augmentations.png")
    save_progressive_augmentation(image, mask, args.size, out_dir / "progressive_augmentation.png")
    save_boundary_visualization(image, mask, out_dir / "boundary_visualization.png")
    save_aspect_ratio_comparison(image, mask, args.size, out_dir / "aspect_ratio_comparison.png")
    save_full_pipeline_samples(image, mask, args.size, args.samples, out_dir / "full_pipeline_samples.png")

    print("Done. Generated files:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"- {p}")


if __name__ == "__main__":
    main()
