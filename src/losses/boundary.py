import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import ensure_nchw


def create_boundary_mask(mask: torch.Tensor, dilation_size: int = 3) -> torch.Tensor:
    """Create boundary mask via morphological gradient (dilate - erode) using torch ops."""
    if dilation_size < 1 or dilation_size % 2 == 0:
        raise ValueError("dilation_size must be an odd integer >= 1")

    mask = ensure_nchw(mask)
    pad = dilation_size // 2

    dilated = F.max_pool2d(mask, kernel_size=dilation_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=dilation_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return boundary


def prediction_boundary_map(probs: torch.Tensor) -> torch.Tensor:
    """Approximate boundary strength from prediction probabilities."""
    grad_x = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
    grad_y = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
    return (grad_x + grad_y).clamp(0.0, 1.0)


class BoundaryIoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, from_logits: bool = False) -> None:
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, boundary_pred: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
        boundary_target = ensure_nchw(boundary_target)
        boundary_pred = torch.sigmoid(boundary_pred) if self.from_logits else boundary_pred
        boundary_pred = boundary_pred.float().clamp(0.0, 1.0)

        intersection = (boundary_pred * boundary_target).sum(dim=(2, 3))
        union = boundary_pred.sum(dim=(2, 3)) + boundary_target.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class BoundaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, from_logits: bool = False) -> None:
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, boundary_pred: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
        boundary_target = ensure_nchw(boundary_target)
        boundary_pred = torch.sigmoid(boundary_pred) if self.from_logits else boundary_pred
        boundary_pred = boundary_pred.float().clamp(0.0, 1.0)

        pred_flat = boundary_pred.view(boundary_pred.shape[0], -1)
        target_flat = boundary_target.view(boundary_target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class ActiveContourLoss(nn.Module):
    def __init__(self, mu: float = 1.0) -> None:
        super().__init__()
        self.mu = mu

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        grad_x = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])
        grad_y = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])
        length = grad_x.mean() + grad_y.mean()
        return self.mu * length


class EdgeAwareLoss(nn.Module):
    def __init__(self, edge_weight: float = 2.0, dilation_size: int = 3) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.dilation_size = dilation_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = ensure_nchw(targets)
        edge_map = create_boundary_mask(targets, dilation_size=self.dilation_size)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = 1.0 + self.edge_weight * edge_map
        return (bce * weights).mean()
