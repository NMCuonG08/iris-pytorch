import torch
import torch.nn as nn

from .boundary import (
    ActiveContourLoss,
    BoundaryDiceLoss,
    BoundaryIoULoss,
    EdgeAwareLoss,
    create_boundary_mask,
    prediction_boundary_map,
)
from .dice import DiceLoss, ensure_nchw
from .focal import FocalLoss


class CombinedDiceFocalLoss(nn.Module):
    """Backwards-compatible combined loss with optional boundary-aware terms."""

    def __init__(
        self,
        dice_weight: float = 0.6,
        focal_weight: float = 0.4,
        focal_alpha: float = 0.8,
        focal_gamma: float = 2.0,
        boundary_iou_weight: float = 0.0,
        boundary_dice_weight: float = 0.0,
        contour_weight: float = 0.0,
        edge_weight: float = 0.0,
        boundary_dilation_size: int = 3,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_iou_weight = boundary_iou_weight
        self.boundary_dice_weight = boundary_dice_weight
        self.contour_weight = contour_weight
        self.edge_weight = edge_weight
        self.boundary_dilation_size = boundary_dilation_size

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_iou_loss = BoundaryIoULoss(from_logits=False)
        self.boundary_dice_loss = BoundaryDiceLoss(from_logits=False)
        self.active_contour_loss = ActiveContourLoss(mu=1.0)
        self.edge_aware_loss = EdgeAwareLoss(edge_weight=2.0, dilation_size=boundary_dilation_size)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = ensure_nchw(targets)

        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        total = self.dice_weight * dice + self.focal_weight * focal

        if self.boundary_iou_weight > 0.0 or self.boundary_dice_weight > 0.0:
            probs = torch.sigmoid(logits)
            pred_boundary = prediction_boundary_map(probs)
            target_boundary = create_boundary_mask(targets, dilation_size=self.boundary_dilation_size)

            if self.boundary_iou_weight > 0.0:
                total = total + self.boundary_iou_weight * self.boundary_iou_loss(pred_boundary, target_boundary)
            if self.boundary_dice_weight > 0.0:
                total = total + self.boundary_dice_weight * self.boundary_dice_loss(pred_boundary, target_boundary)

        if self.contour_weight > 0.0:
            total = total + self.contour_weight * self.active_contour_loss(logits)

        if self.edge_weight > 0.0:
            total = total + self.edge_weight * self.edge_aware_loss(logits, targets)

        return total


class CombinedIrisLoss(CombinedDiceFocalLoss):
    """Alias with defaults aligned to boundary-aware training presets."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        boundary_iou_weight: float = 0.25,
        boundary_dice_weight: float = 0.0,
        contour_weight: float = 0.0,
        edge_weight: float = 0.0,
        focal_alpha: float = 0.8,
        focal_gamma: float = 2.0,
        boundary_dilation_size: int = 3,
    ) -> None:
        super().__init__(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            boundary_iou_weight=boundary_iou_weight,
            boundary_dice_weight=boundary_dice_weight,
            contour_weight=contour_weight,
            edge_weight=edge_weight,
            boundary_dilation_size=boundary_dilation_size,
        )
