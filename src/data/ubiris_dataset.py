from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UBIRISV2Dataset(Dataset):
    """UBIRIS V2 segmentation dataset.

    Expects image/mask pairs with identical stems.
    """

    def __init__(
        self,
        root_dir: str,
        image_dir: str,
        mask_dir: str,
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        transform: Callable | None = None,
        file_stems: List[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.mask_dir = self.root_dir / mask_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.transform = transform

        if file_stems is None:
            image_paths = sorted(self.image_dir.glob(f"*{self.image_ext}"))
            self.file_stems = [p.stem for p in image_paths]
        else:
            self.file_stems = sorted(file_stems)

        if not self.file_stems:
            raise ValueError(f"No files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.file_stems)

    def _load_pair(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.image_dir / f"{stem}{self.image_ext}"
        mask_path = self.mask_dir / f"{stem}{self.mask_ext}"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Try with OperatorA_ prefix if mask not found
        if mask is None:
            mask_path_alt = self.mask_dir / f"OperatorA_{stem}{self.mask_ext}"
            mask = cv2.imread(str(mask_path_alt), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_path = mask_path_alt

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)

        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.file_stems[idx]
        image, mask = self._load_pair(stem)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask
