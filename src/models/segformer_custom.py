"""Enhanced SegFormer implementations for iris segmentation."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from .heads import AuxiliaryHead, BoundaryRefinementHead


def _load_segformer_with_fallback(model_name: str, num_labels: int) -> SegformerForSemanticSegmentation:
    candidates = [
        model_name,
        "nvidia/segformer-b1-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512",
    ]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return SegformerForSemanticSegmentation.from_pretrained(
                candidate,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                token=False,
            )
        except Exception as e:  # pragma: no cover - runtime fallback path
            last_error = e

    # Final fallback: initialize an untrained SegFormer from config so training can still start offline.
    config = SegformerConfig(num_labels=num_labels)
    return SegformerForSemanticSegmentation(config)


class EnhancedSegFormer(nn.Module):
    """SegFormer with optional boundary refinement head."""

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        freeze_encoder: bool = False,
        freeze_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.segformer = _load_segformer_with_fallback(model_name=model_name, num_labels=num_labels)

        self.num_labels = num_labels
        self.add_boundary_head = add_boundary_head
        if add_boundary_head:
            self.boundary_head = BoundaryRefinementHead(in_channels=num_labels, hidden_channels=64)

        self.freeze_encoder = freeze_encoder
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for param in self.segformer.segformer.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        for param in self.segformer.segformer.encoder.parameters():
            param.requires_grad = True

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if self.freeze_encoder and epoch >= self.freeze_epochs:
            self._unfreeze_encoder()
            self.freeze_encoder = False

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        # HF SegformerForSemanticSegmentation expects class ids [B, H, W].
        if labels.ndim == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        if self.num_labels == 1:
            labels = labels.long()
        return labels.long()

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_boundary: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        hf_labels = self._prepare_labels(labels) if labels is not None else None
        outputs = self.segformer(pixel_values=pixel_values, labels=hf_labels)

        upsampled_logits = F.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        result: Dict[str, Optional[torch.Tensor]] = {
            "logits": upsampled_logits,
            "seg_loss": outputs.loss if labels is not None else None,
        }

        if self.add_boundary_head and return_boundary:
            result["boundary_logits"] = self.boundary_head(upsampled_logits)

        return result


class DeepSupervisionSegFormer(EnhancedSegFormer):
    """SegFormer with optional deep supervision branch."""

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        deep_supervision: bool = True,
        aux_loss_weight: float = 0.2,
        freeze_encoder: bool = False,
        freeze_epochs: int = 0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            add_boundary_head=add_boundary_head,
            freeze_encoder=freeze_encoder,
            freeze_epochs=freeze_epochs,
        )
        self.deep_supervision = deep_supervision
        self.aux_loss_weight = aux_loss_weight

        if deep_supervision:
            hidden_sizes = list(self.segformer.config.hidden_sizes)
            in_channels = hidden_sizes[2] if len(hidden_sizes) > 2 else hidden_sizes[-1]
            self.aux_classifier = AuxiliaryHead(in_channels=in_channels, num_classes=num_labels, dropout=0.1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_boundary: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        encoder_outputs = self.segformer.segformer(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = encoder_outputs.hidden_states

        result = super().forward(pixel_values=pixel_values, labels=labels, return_boundary=return_boundary)

        if self.deep_supervision and labels is not None and hidden_states is not None:
            aux_features = hidden_states[2] if len(hidden_states) > 2 else hidden_states[-1]
            aux_logits = self.aux_classifier(aux_features)
            aux_logits = F.interpolate(
                aux_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            if self.num_labels == 1:
                target = labels.float()
                if target.ndim == 3:
                    target = target.unsqueeze(1)
                aux_loss = F.binary_cross_entropy_with_logits(aux_logits, target)
            else:
                target = labels.squeeze(1).long() if labels.ndim == 4 else labels.long()
                aux_loss = F.cross_entropy(aux_logits, target)

            result["aux_logits"] = aux_logits
            result["aux_loss"] = self.aux_loss_weight * aux_loss

        return result


class SegFormerCustom(nn.Module):
    """Backward-compatible wrapper used by the current trainer."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _ = decoder_channels
        _ = dropout
        self.model = EnhancedSegFormer(
            model_name=backbone_name,
            num_labels=num_classes,
            add_boundary_head=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x, labels=None, return_boundary=False)
        logits = outputs["logits"]
        if logits is None:
            raise RuntimeError("Model did not produce logits")
        return logits


def create_model(
    model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
    num_labels: int = 2,
    model_type: str = "enhanced",
    **kwargs: Any,
) -> nn.Module:
    if model_type == "enhanced":
        return EnhancedSegFormer(model_name=model_name, num_labels=num_labels, **kwargs)
    if model_type == "deep_supervision":
        return DeepSupervisionSegFormer(model_name=model_name, num_labels=num_labels, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")


def load_pretrained_iris_model(checkpoint_path: str, model_config: Dict[str, Any]) -> nn.Module:
    model = create_model(**model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params
