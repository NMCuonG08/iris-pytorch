import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _opencv_denoise(image: np.ndarray, **kwargs) -> np.ndarray:
    """Apply non-local means denoising using OpenCV."""
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def build_train_transforms(input_size: tuple[int, int]) -> A.Compose:
    h, w = input_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.06,
                        scale_limit=0.10,
                        rotate_limit=18,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1.0,
                    ),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.05, 0.05),
                        rotate=(-20, 20),
                        shear=(-8, 8),
                        mode=cv2.BORDER_REFLECT_101,
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25,
                        contrast_limit=0.35,
                        p=1.0,
                    ),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomGamma(gamma_limit=(70, 140), p=1.0),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.03, 0.10), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
                    A.ImageCompression(quality_range=(35, 80), compression_type="jpeg", p=1.0),
                ],
                p=0.35,
            ),
            A.Lambda(image=_opencv_denoise, p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_val_transforms(input_size: tuple[int, int]) -> A.Compose:
    h, w = input_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
