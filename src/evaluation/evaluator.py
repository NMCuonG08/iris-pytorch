"""Model evaluation orchestrator for iris segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import IrisSegmentationMetrics, benchmark_inference_speed


class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 2,
        save_predictions: bool = False,
        output_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.save_predictions = save_predictions
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "predictions").mkdir(exist_ok=True)
            (self.output_dir / "failed_cases").mkdir(exist_ok=True)

        self.metrics = IrisSegmentationMetrics(num_classes=num_classes)

    def evaluate(
        self,
        dataloader: DataLoader,
        save_failed_cases: bool = True,
        iou_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        self.model.eval()
        self.metrics.reset()

        failed_cases: List[Dict[str, Any]] = []
        all_predictions: List[Dict[str, Any]] = []

        print(f"Evaluating model on {len(dataloader)} batches...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                pixel_values, labels, image_paths = self._unpack_batch(batch)
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)

                outputs = self._forward_model(pixel_values)
                logits = outputs["logits"]
                predictions = self._to_predictions(logits)

                boundary_logits = outputs.get("boundary_logits")
                self.metrics.update(predictions, labels, boundary_logits, None)

                if save_failed_cases:
                    batch_ious = self._compute_batch_ious(predictions, labels)
                    for i, iou in enumerate(batch_ious):
                        if iou < iou_threshold:
                            failed_cases.append(
                                {
                                    "batch_idx": batch_idx,
                                    "sample_idx": i,
                                    "iou": iou,
                                    "image_path": image_paths[i] if image_paths else None,
                                    "prediction": predictions[i].detach().cpu(),
                                    "target": labels[i].detach().cpu(),
                                    "image": pixel_values[i].detach().cpu(),
                                }
                            )

                if self.save_predictions:
                    for i in range(predictions.shape[0]):
                        all_predictions.append(
                            {
                                "prediction": predictions[i].detach().cpu(),
                                "target": labels[i].detach().cpu(),
                                "image": pixel_values[i].detach().cpu(),
                                "batch_idx": batch_idx,
                                "sample_idx": i,
                            }
                        )

        final_metrics = self.metrics.compute_all_metrics()
        total_samples = len(dataloader.dataset)

        evaluation_results = {
            "metrics": final_metrics,
            "total_samples": total_samples,
            "failed_cases": len(failed_cases),
            "failed_rate": float(len(failed_cases) / max(total_samples, 1)),
            "evaluation_summary": self._create_evaluation_summary(final_metrics),
        }

        if save_failed_cases and failed_cases and self.output_dir:
            self._save_failed_cases(failed_cases, iou_threshold)

        if self.save_predictions and all_predictions and self.output_dir:
            self._save_predictions(all_predictions)

        return evaluation_results

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[List[str]]]:
        if isinstance(batch, dict):
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
            image_paths = batch.get("image_path")
            return pixel_values, labels, image_paths

        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError("Expected batch tuple/list to have at least (images, labels)")
            pixel_values = batch[0]
            labels = batch[1]
            image_paths = batch[2] if len(batch) > 2 else None
            return pixel_values, labels, image_paths

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _forward_model(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            outputs = self.model(pixel_values, return_boundary=True)
            if isinstance(outputs, dict) and "logits" in outputs:
                return outputs
        except TypeError:
            pass

        logits = self.model(pixel_values)
        return {"logits": logits}

    def _to_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4 and logits.shape[1] > 1:
            return torch.argmax(logits, dim=1)
        return (torch.sigmoid(logits).squeeze(1) > 0.5).long()

    def _compute_batch_ious(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        ious: List[float] = []
        for i in range(predictions.shape[0]):
            pred = predictions[i].detach().cpu().numpy()
            target = targets[i].detach().cpu().numpy()

            pred_iris = pred == 1
            target_iris = target == 1

            intersection = np.logical_and(pred_iris, target_iris).sum()
            union = np.logical_or(pred_iris, target_iris).sum()
            iou = float(intersection / union) if union > 0 else 0.0
            ious.append(iou)

        return ious

    def _save_failed_cases(self, failed_cases: List[Dict[str, Any]], iou_threshold: float) -> None:
        print(f"Saving {len(failed_cases)} failed cases (IoU < {iou_threshold})...")
        for i, case in enumerate(failed_cases[:20]):
            out = self.output_dir / "failed_cases" / f"failed_case_{i + 1}_iou_{case['iou']:.3f}.png"
            self._visualize_case(case, out)

    def _save_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        print(f"Saving {len(predictions)} predictions...")
        for i, pred_data in enumerate(predictions[::10]):
            out = self.output_dir / "predictions" / f"prediction_{i + 1}.png"
            self._visualize_case(pred_data, out)

    def _visualize_case(self, case_data: Dict[str, Any], save_path: Path) -> None:
        image = case_data["image"]
        prediction = case_data["prediction"]
        target = case_data["target"]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).numpy()

        pred_np = prediction.numpy()
        target_np = target.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(image_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(target_np, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        error_map = np.zeros((*pred_np.shape, 3), dtype=np.float32)
        error_map[pred_np != target_np] = [1.0, 0.0, 0.0]
        error_map[np.logical_and(pred_np == 1, target_np == 1)] = [0.0, 1.0, 0.0]

        axes[3].imshow(error_map)
        axes[3].set_title("Error Map\n(Red=Error, Green=Correct)")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_evaluation_summary(self, metrics: Dict[str, float]) -> str:
        return (
            "Evaluation Summary:\n"
            "==================\n"
            f"Pixel Accuracy: {metrics.get('pixel_accuracy', 0.0):.3f}\n"
            f"Mean IoU: {metrics.get('mean_iou', 0.0):.3f}\n"
            f"Mean Dice: {metrics.get('mean_dice', 0.0):.3f}\n"
            f"Iris IoU (class_1_iou): {metrics.get('class_1_iou', 0.0):.3f}\n"
            f"Iris Dice (class_1_dice): {metrics.get('class_1_dice', 0.0):.3f}\n"
        )

    def benchmark_speed(self, input_size: tuple = (1, 3, 512, 512), num_runs: int = 100) -> Dict[str, float]:
        return benchmark_inference_speed(
            self.model,
            input_size=input_size,
            device=self.device,
            num_runs=num_runs,
        )


class CrossValidationEvaluator:
    def __init__(
        self,
        model_factory: Any,
        n_folds: int = 5,
        device: torch.device = torch.device("cpu"),
        output_dir: Optional[str] = None,
    ) -> None:
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_fold(self, fold_idx: int, test_loader: DataLoader, checkpoint_path: str) -> Dict[str, Any]:
        print(f"Evaluating fold {fold_idx + 1}/{self.n_folds}...")
        model = self.model_factory()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        fold_output = self.output_dir / f"fold_{fold_idx}" if self.output_dir else None
        evaluator = ModelEvaluator(
            model=model,
            device=self.device,
            save_predictions=True,
            output_dir=str(fold_output) if fold_output else None,
        )

        test_results = evaluator.evaluate(test_loader, save_failed_cases=True)
        speed_results = evaluator.benchmark_speed()

        return {
            "fold_idx": fold_idx,
            "test_metrics": test_results["metrics"],
            "speed_metrics": speed_results,
            "evaluation_summary": test_results["evaluation_summary"],
            "checkpoint_path": checkpoint_path,
        }

    def evaluate_all_folds(self, fold_checkpoints: List[str], test_loaders: List[DataLoader]) -> Dict[str, Any]:
        if not (len(fold_checkpoints) == len(test_loaders) == self.n_folds):
            raise ValueError("fold_checkpoints and test_loaders must match n_folds")

        fold_results = [
            self.evaluate_fold(idx, test_loader, ckpt)
            for idx, (ckpt, test_loader) in enumerate(zip(fold_checkpoints, test_loaders))
        ]

        aggregated = self._aggregate_fold_results(fold_results)
        if self.output_dir:
            self._save_cv_results(aggregated)
        return aggregated

    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metric_names = fold_results[0]["test_metrics"].keys()
        speed_names = fold_results[0]["speed_metrics"].keys()

        aggregated_metrics: Dict[str, Dict[str, Any]] = {}
        for metric_name in metric_names:
            values = np.array([fold["test_metrics"][metric_name] for fold in fold_results], dtype=np.float64)
            aggregated_metrics[metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "values": [float(v) for v in values],
            }

        aggregated_speed: Dict[str, Dict[str, float]] = {}
        for speed_name in speed_names:
            values = np.array([fold["speed_metrics"][speed_name] for fold in fold_results], dtype=np.float64)
            aggregated_speed[speed_name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }

        summary = self._create_cv_summary(aggregated_metrics)
        return {
            "n_folds": self.n_folds,
            "aggregated_metrics": aggregated_metrics,
            "aggregated_speed": aggregated_speed,
            "fold_results": fold_results,
            "summary": summary,
        }

    def _create_cv_summary(self, aggregated_metrics: Dict[str, Dict[str, Any]]) -> str:
        miou_mean = aggregated_metrics.get("mean_iou", {}).get("mean", 0.0)
        miou_std = aggregated_metrics.get("mean_iou", {}).get("std", 0.0)
        dice_mean = aggregated_metrics.get("mean_dice", {}).get("mean", 0.0)
        dice_std = aggregated_metrics.get("mean_dice", {}).get("std", 0.0)

        return (
            f"Cross-Validation ({self.n_folds}-Fold)\n"
            "=====================================\n"
            f"Mean IoU: {miou_mean:.3f} +- {miou_std:.3f}\n"
            f"Mean Dice: {dice_mean:.3f} +- {dice_std:.3f}\n"
        )

    def _save_cv_results(self, results: Dict[str, Any]) -> None:
        import json

        results_path = self.output_dir / "cv_results.json"
        summary_path = self.output_dir / "cv_summary.txt"

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        with summary_path.open("w", encoding="utf-8") as f:
            f.write(results["summary"])
