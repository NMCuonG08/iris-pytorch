from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _derive_iou_from_dice(dice_value: float) -> float:
    if dice_value <= 0:
        return 0.0
    return dice_value / (2.0 - dice_value)


def load_metrics(metrics_file: str, summary_file: str | None = None) -> List[dict]:
    path = Path(metrics_file)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    if value in (None, ""):
                        parsed[key] = value
                    elif key == "epoch":
                        parsed[key] = int(float(value))
                    else:
                        parsed[key] = float(value)
                rows.append(parsed)
            return rows

    if summary_file is None:
        summary_file = str(path.with_name("epoch_summary.txt"))

    summary_path = Path(summary_file)
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find metrics file: {metrics_file} or summary file: {summary_file}")

    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/(?:\d+)\s+summary:\s+"
        r"loss_train=(?P<loss_train>[0-9.]+),\s+"
        r"acc_train=(?P<acc_train>[0-9.]+)%,\s+"
        r"loss_val=(?P<loss_val>[0-9.]+),\s+"
        r"acc_val=(?P<acc_val>[0-9.]+)%"
        r"(?:,\s+miou_train=(?P<miou_train>[0-9.]+)%,\s+miou_val=(?P<miou_val>[0-9.]+)%)?"
    )
    rows = []
    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            train_loss = float(match.group("loss_train"))
            train_dice = float(match.group("acc_train")) / 100.0
            val_loss = float(match.group("loss_val"))
            val_dice = float(match.group("acc_val")) / 100.0
            train_iou = match.group("miou_train")
            val_iou = match.group("miou_val")
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_dice,
                    "train_iou": float(train_iou) / 100.0 if train_iou is not None else _derive_iou_from_dice(train_dice),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_acc": val_dice,
                    "val_iou": float(val_iou) / 100.0 if val_iou is not None else _derive_iou_from_dice(val_dice),
                }
            )
    if not rows:
        raise ValueError(f"No metrics could be parsed from: {summary_path}")
    return rows


def plot_training_curves(
    metrics_file: str,
    output_path: str,
    summary_file: str | None = None,
):
    rows = load_metrics(metrics_file, summary_file=summary_file)
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_iou = [float(row["train_iou"]) * 100.0 for row in rows]
    val_iou = [float(row["val_iou"]) * 100.0 for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_iou, marker="o", label="Train mIoU")
    axes[1].plot(epochs, val_iou, marker="o", label="Val mIoU")
    axes[1].set_title("mIoU Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path