"""MobileNetV3-Large with a Lite R-ASPP segmentation head.

Adapted from Torchvision's BSD-3-Clause licensed MobileNetV3 and LR-ASPP
implementations. It uses only core PyTorch operators to avoid a binary
torchvision dependency and returns FM2Edge's common Student output.

Sources:
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py
License copy: third_party_licenses/TORCHVISION_LICENSE.txt
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from fm2edge.models.outputs import ModelOutput


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        activation: type[nn.Module] | None = nn.ReLU,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        ]
        if activation is not None:
            layers.append(activation(inplace=True))
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // 4, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.expand = nn.Conv2d(squeeze_channels, input_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.reduce(scale), inplace=True)
        scale = F.hardsigmoid(self.expand(scale), inplace=True)
        return x * scale


@dataclass(frozen=True)
class InvertedResidualConfig:
    input_channels: int
    kernel_size: int
    expanded_channels: int
    out_channels: int
    use_se: bool
    use_hardswish: bool
    stride: int
    dilation: int = 1


class InvertedResidual(nn.Module):
    def __init__(self, config: InvertedResidualConfig) -> None:
        super().__init__()
        activation = nn.Hardswish if config.use_hardswish else nn.ReLU
        layers: list[nn.Module] = []
        if config.expanded_channels != config.input_channels:
            layers.append(
                ConvNormActivation(
                    config.input_channels,
                    config.expanded_channels,
                    kernel_size=1,
                    activation=activation,
                )
            )
        actual_stride = 1 if config.dilation > 1 else config.stride
        layers.append(
            ConvNormActivation(
                config.expanded_channels,
                config.expanded_channels,
                kernel_size=config.kernel_size,
                stride=actual_stride,
                groups=config.expanded_channels,
                dilation=config.dilation,
                activation=activation,
            )
        )
        if config.use_se:
            layers.append(SqueezeExcitation(config.expanded_channels))
        layers.append(
            ConvNormActivation(
                config.expanded_channels,
                config.out_channels,
                kernel_size=1,
                activation=None,
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = config.stride == 1 and config.input_channels == config.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        return result + x if self.use_residual else result


class MobileNetV3LargeBackbone(nn.Module):
    """Dilated MobileNetV3-Large exposing stride-8 and stride-16 features."""

    low_channels = 40
    high_channels = 960

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvNormActivation(3, 16, stride=2, activation=nn.Hardswish)
        settings = (
            InvertedResidualConfig(16, 3, 16, 16, False, False, 1),
            InvertedResidualConfig(16, 3, 64, 24, False, False, 2),
            InvertedResidualConfig(24, 3, 72, 24, False, False, 1),
            InvertedResidualConfig(24, 5, 72, 40, True, False, 2),
            InvertedResidualConfig(40, 5, 120, 40, True, False, 1),
            InvertedResidualConfig(40, 5, 120, 40, True, False, 1),
            InvertedResidualConfig(40, 3, 240, 80, False, True, 2),
            InvertedResidualConfig(80, 3, 200, 80, False, True, 1),
            InvertedResidualConfig(80, 3, 184, 80, False, True, 1),
            InvertedResidualConfig(80, 3, 184, 80, False, True, 1),
            InvertedResidualConfig(80, 3, 480, 112, True, True, 1),
            InvertedResidualConfig(112, 3, 672, 112, True, True, 1),
            InvertedResidualConfig(112, 5, 672, 160, True, True, 2, dilation=2),
            InvertedResidualConfig(160, 5, 960, 160, True, True, 1, dilation=2),
            InvertedResidualConfig(160, 5, 960, 160, True, True, 1, dilation=2),
        )
        self.blocks = nn.ModuleList(InvertedResidual(config) for config in settings)
        self.final = ConvNormActivation(160, 960, kernel_size=1, activation=nn.Hardswish)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low: torch.Tensor | None = None
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 3:
                low = x
        if low is None:
            raise RuntimeError("MobileNetV3 low-level feature was not produced")
        return low, self.final(x)


class LiteRASPPHead(nn.Module):
    def __init__(self, num_classes: int, intermediate_channels: int = 128) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(960, intermediate_channels, 1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, intermediate_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(40, num_classes, 1)
        self.high_classifier = nn.Conv2d(intermediate_channels, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context(high) * self.scale(high)
        context = F.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.low_classifier(low) + self.high_classifier(context)
        return logits, context


class MobileNetV3LRASPP(nn.Module):
    """Torchvision-structure MobileNetV3-Large + LR-ASPP Student."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = MobileNetV3LargeBackbone()
        self.head = LiteRASPPHead(num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> ModelOutput:
        low, high = self.backbone(image)
        logits, kd_feature = self.head(low, high)
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return ModelOutput(logits=logits, features={"kd": kd_feature})
