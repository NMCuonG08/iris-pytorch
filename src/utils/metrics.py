import torch


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    total = preds.sum(dim=1) + targets.sum(dim=1)
    union = total - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
