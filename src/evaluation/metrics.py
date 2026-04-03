"""Comprehensive evaluation metrics for iris segmentation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


class IrisSegmentationMetrics:
    def __init__(self, num_classes: int = 2, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.boundary_predictions: List[np.ndarray] = []
        self.boundary_targets: List[np.ndarray] = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        boundary_predictions: Optional[torch.Tensor] = None,
        boundary_targets: Optional[torch.Tensor] = None,
    ) -> None:
        if predictions.dim() == 4 and predictions.shape[1] > 1:
            predictions = torch.argmax(predictions, dim=1)
        elif predictions.dim() == 4:
            predictions = (torch.sigmoid(predictions).squeeze(1) > 0.5).long()

        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        self.predictions.extend(predictions.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())

        if boundary_predictions is not None and boundary_targets is not None:
            if boundary_predictions.dim() == 4:
                boundary_predictions = torch.sigmoid(boundary_predictions).squeeze(1)
            if boundary_targets.dim() == 4 and boundary_targets.shape[1] == 1:
                boundary_targets = boundary_targets.squeeze(1)
            self.boundary_predictions.extend(boundary_predictions.detach().cpu().numpy())
            self.boundary_targets.extend(boundary_targets.detach().cpu().numpy())

    def _flatten_valid(self) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        valid = targets != self.ignore_index
        return predictions[valid], targets[valid]

    def compute_pixel_accuracy(self) -> float:
        pred, tgt = self._flatten_valid()
        if tgt.size == 0:
            return 0.0
        return float((pred == tgt).mean())

    def compute_class_accuracy(self) -> Dict[str, float]:
        pred, tgt = self._flatten_valid()
        metrics: Dict[str, float] = {}
        for class_id in range(self.num_classes):
            class_mask = tgt == class_id
            if class_mask.sum() == 0:
                metrics[f"class_{class_id}_acc"] = 0.0
            else:
                metrics[f"class_{class_id}_acc"] = float((pred[class_mask] == class_id).mean())
        return metrics

    def compute_iou(self) -> Dict[str, float]:
        pred, tgt = self._flatten_valid()
        metrics: Dict[str, float] = {}
        values: List[float] = []
        for class_id in range(self.num_classes):
            pred_mask = pred == class_id
            tgt_mask = tgt == class_id
            intersection = np.logical_and(pred_mask, tgt_mask).sum()
            union = np.logical_or(pred_mask, tgt_mask).sum()
            iou = float(intersection / union) if union > 0 else 0.0
            metrics[f"class_{class_id}_iou"] = iou
            values.append(iou)
        metrics["mean_iou"] = float(np.mean(values)) if values else 0.0
        return metrics

    def compute_dice(self) -> Dict[str, float]:
        pred, tgt = self._flatten_valid()
        metrics: Dict[str, float] = {}
        values: List[float] = []
        for class_id in range(self.num_classes):
            pred_mask = pred == class_id
            tgt_mask = tgt == class_id
            intersection = np.logical_and(pred_mask, tgt_mask).sum()
            total = pred_mask.sum() + tgt_mask.sum()
            dice = float((2 * intersection) / total) if total > 0 else 0.0
            metrics[f"class_{class_id}_dice"] = dice
            values.append(dice)
        metrics["mean_dice"] = float(np.mean(values)) if values else 0.0
        return metrics

    def compute_precision_recall_f1(self) -> Dict[str, float]:
        pred, tgt = self._flatten_valid()
        metrics: Dict[str, float] = {}

        precision_vals: List[float] = []
        recall_vals: List[float] = []
        f1_vals: List[float] = []

        for class_id in range(self.num_classes):
            tp = np.logical_and(pred == class_id, tgt == class_id).sum()
            fp = np.logical_and(pred == class_id, tgt != class_id).sum()
            fn = np.logical_and(pred != class_id, tgt == class_id).sum()

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

            metrics[f"class_{class_id}_precision"] = precision
            metrics[f"class_{class_id}_recall"] = recall
            metrics[f"class_{class_id}_f1"] = f1

            precision_vals.append(precision)
            recall_vals.append(recall)
            f1_vals.append(f1)

        metrics["macro_precision"] = float(np.mean(precision_vals)) if precision_vals else 0.0
        metrics["macro_recall"] = float(np.mean(recall_vals)) if recall_vals else 0.0
        metrics["macro_f1"] = float(np.mean(f1_vals)) if f1_vals else 0.0
        return metrics

    def compute_boundary_f1(self, tolerance: int = 2) -> float:
        if not self.boundary_predictions or not self.boundary_targets:
            return 0.0

        scores: List[float] = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1))

        for pred, target in zip(self.boundary_predictions, self.boundary_targets):
            pred_bin = (pred > 0.5).astype(np.uint8)
            target_bin = (target > 0.5).astype(np.uint8)
            target_dilated = cv2.dilate(target_bin, kernel, iterations=1)

            tp = np.logical_and(pred_bin == 1, target_dilated == 1).sum()
            pred_pos = (pred_bin == 1).sum()
            tgt_pos = (target_bin == 1).sum()

            precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
            recall = float(tp / tgt_pos) if tgt_pos > 0 else 0.0
            f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            scores.append(f1)

        return float(np.mean(scores)) if scores else 0.0

    def compute_all_metrics(self) -> Dict[str, float]:
        if len(self.predictions) == 0:
            return {}

        metrics: Dict[str, float] = {}
        metrics["pixel_accuracy"] = self.compute_pixel_accuracy()
        metrics.update(self.compute_class_accuracy())
        metrics.update(self.compute_iou())
        metrics.update(self.compute_dice())
        metrics.update(self.compute_precision_recall_f1())

        if self.boundary_predictions and self.boundary_targets:
            metrics["boundary_f1"] = self.compute_boundary_f1()

        return metrics


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def benchmark_inference_speed(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: torch.device = torch.device("cpu"),
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_size, device=device)

    def _forward() -> None:
        try:
            _ = model(dummy_input)
        except TypeError:
            _ = model(dummy_input, return_boundary=False)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _forward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    import time

    times: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _forward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

    arr = np.array(times, dtype=np.float64)
    return {
        "mean_time": float(arr.mean()),
        "std_time": float(arr.std()),
        "min_time": float(arr.min()),
        "max_time": float(arr.max()),
        "fps": float(1.0 / arr.mean()),
        "throughput": float(input_size[0] / arr.mean()),
    }
