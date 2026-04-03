import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import ensure_nchw


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = ensure_nchw(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        focal_term = (1.0 - pt).pow(self.gamma)
        focal = self.alpha * focal_term * bce
        return focal.mean()
