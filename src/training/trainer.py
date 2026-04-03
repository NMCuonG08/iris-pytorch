import csv
from pathlib import Path
from datetime import datetime

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import dice_score_from_logits, iou_score_from_logits


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader,
        val_loader,
        config: dict,
        logger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger

        train_cfg = config["training"]
        project_cfg = config["project"]
        log_cfg = config["logging"]

        requested_device = train_cfg.get("device", "cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.logger.warning("CUDA not available, falling back to CPU")
        else:
            self.device = torch.device(requested_device)

        self.model.to(self.device)

        self.epochs = train_cfg.get("epochs", 50)
        self.amp = train_cfg.get("amp", True) and self.device.type == "cuda"
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
        self.save_every = train_cfg.get("save_every", 1)
        self.log_interval = log_cfg.get("log_interval", 10)

        self.output_dir = Path(project_cfg.get("output_dir", "runs/default"))
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.summary_file = self.output_dir / "epoch_summary.txt"
        self.metrics_file = self.output_dir / "metrics.csv"
        self.scaler = GradScaler(enabled=self.amp)

        self.metric_to_monitor = log_cfg.get("metric_to_monitor", "val_dice")
        self.monitor_mode = log_cfg.get("monitor_mode", "max")
        self.best_metric = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> None:
        self.logger.info("Start training for %d epochs", self.epochs)
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, train_iou = self._train_one_epoch(epoch)
            val_loss, val_dice, val_iou = self._validate(epoch)
            val_acc = val_dice

            logs = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_acc": val_acc,
                "val_iou": val_iou,
            }
            self.logger.info("Epoch %d summary: %s", epoch, logs)

            if epoch % self.save_every == 0:
                self._save_latest(epoch, logs)

            current = logs.get(self.metric_to_monitor)
            if current is not None and self._is_best(current):
                self.best_metric = current
                self.best_epoch = epoch
                self._save_best(epoch, logs)

            self._write_epoch_summary(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_iou=train_iou,
                val_loss=val_loss,
                val_acc=val_acc,
                val_iou=val_iou,
            )
            self._write_metrics_row(logs)

    def _train_one_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for step, (images, masks) in enumerate(pbar, start=1):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            running_dice += dice_score_from_logits(logits, masks)
            running_iou += iou_score_from_logits(logits, masks)
            if step % self.log_interval == 0:
                self.logger.info(
                    "Epoch %d | Step %d/%d | Train Loss %.4f",
                    epoch,
                    step,
                    len(self.train_loader),
                    loss.item(),
                )

        denom = max(len(self.train_loader), 1)
        return running_loss / denom, running_dice / denom, running_iou / denom

    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple[float, float, float]:
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0

        pbar = tqdm(self.val_loader, desc=f"Val {epoch}", leave=False)
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            running_loss += loss.item()
            running_dice += dice_score_from_logits(logits, masks)
            running_iou += iou_score_from_logits(logits, masks)

        denom = max(len(self.val_loader), 1)
        return running_loss / denom, running_dice / denom, running_iou / denom

    def _is_best(self, current_value: float) -> bool:
        if self.monitor_mode == "max":
            return current_value > self.best_metric
        return current_value < self.best_metric

    def _save_latest(self, epoch: int, logs: dict) -> None:
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "logs": logs,
            },
            str(self.ckpt_dir / "latest.pt"),
        )

    def _save_best(self, epoch: int, logs: dict) -> None:
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "logs": logs,
            },
            str(self.ckpt_dir / "best.pt"),
        )
        self.logger.info("Saved best checkpoint at epoch %d", epoch)

    def _write_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_iou: float,
        val_loss: float,
        val_acc: float,
        val_iou: float,
    ) -> None:
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        best_acc_percent = self.best_metric * 100.0 if self.best_epoch > 0 else val_acc * 100.0
        best_epoch = self.best_epoch if self.best_epoch > 0 else epoch

        line = (
            f"{timestamp} Epoch {epoch}/{self.epochs} summary: "
            f"loss_train={train_loss:.5f}, "
            f"acc_train={train_acc * 100.0:.2f}%, "
            f"loss_val={val_loss:.5f}, "
            f"acc_val={val_acc * 100.0:.2f}%, "
            f"miou_train={train_iou * 100.0:.2f}%, "
            f"miou_val={val_iou * 100.0:.2f}% "
            f"(best: {best_acc_percent:.2f}% @ epoch {best_epoch})"
        )

        with self.summary_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        self.logger.info(line)

    def _write_metrics_row(self, logs: dict) -> None:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_acc",
            "train_iou",
            "val_loss",
            "val_dice",
            "val_acc",
            "val_iou",
        ]
        write_header = not self.metrics_file.exists()
        with self.metrics_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: logs.get(key) for key in fieldnames})
