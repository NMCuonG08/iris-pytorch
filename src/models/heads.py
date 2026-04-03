import torch
import torch.nn as nn


class BoundaryRefinementHead(nn.Module):
    """Lightweight boundary refinement head for sharper contours."""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 64) -> None:
        super().__init__()
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, seg_logits: torch.Tensor) -> torch.Tensor:
        return self.refine_conv(seg_logits)


class AuxiliaryHead(nn.Module):
    """Auxiliary classifier head for deep supervision."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, num_classes, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class AttentionRefinementHead(nn.Module):
    """Attention-based refinement head."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        reduced = max(1, in_channels // 4)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(features)
        return self.refiner(features * attention_weights)
