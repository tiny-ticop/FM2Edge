"""PyTorch PP-LiteSeg with the STDC1 backbone.

Adapted from PaddleSeg's Apache-2.0 licensed PP-LiteSeg and STDCNet
implementations. The topology and channel dimensions follow the official
STDC1 Cityscapes configuration; the return value follows FM2Edge's common
Student interface.

Sources:
https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/paddleseg/models/pp_liteseg.py
https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/paddleseg/models/backbones/stdcnet.py
License copy: third_party_licenses/PADDLESEG_LICENSE.txt
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fm2edge.models.outputs import ModelOutput


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class CatBottleneck(nn.Module):
    """Short-Term Dense Concatenate block used by the official STDC1."""

    def __init__(
        self, in_channels: int, out_channels: int, block_num: int = 4, stride: int = 1
    ) -> None:
        super().__init__()
        if block_num <= 1:
            raise ValueError("block_num must be greater than one")
        self.stride = stride
        first_channels = out_channels // 2
        self.convs = nn.ModuleList([ConvBNReLU(in_channels, first_channels, kernel_size=1)])
        for index in range(1, block_num):
            input_channels = out_channels // (2**index)
            output_channels = (
                out_channels // (2**index)
                if index == block_num - 1
                else out_channels // (2 ** (index + 1))
            )
            self.convs.append(ConvBNReLU(input_channels, output_channels))
        if stride == 2:
            self.downsample_first = nn.Sequential(
                nn.Conv2d(
                    first_channels,
                    first_channels,
                    3,
                    stride=2,
                    padding=1,
                    groups=first_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(first_channels),
            )
            self.skip = nn.AvgPool2d(3, stride=2, padding=1)
        else:
            self.downsample_first = nn.Identity()
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.convs[0](x)
        output = self.convs[1](self.downsample_first(first))
        branches = [self.skip(first), output]
        for conv in self.convs[2:]:
            output = conv(output)
            branches.append(output)
        return torch.cat(branches, dim=1)


class STDC1Backbone(nn.Module):
    """STDCNet1 feature extractor returning x2, x4, x8, x16 and x32 maps."""

    feature_channels = (32, 64, 256, 512, 1024)

    def __init__(self) -> None:
        super().__init__()
        self.stem2 = ConvBNReLU(3, 32, stride=2)
        self.stem4 = ConvBNReLU(32, 64, stride=2)
        self.stage8 = self._make_stage(64, 256, blocks=2)
        self.stage16 = self._make_stage(256, 512, blocks=2)
        self.stage32 = self._make_stage(512, 1024, blocks=2)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        layers: list[nn.Module] = [CatBottleneck(in_channels, out_channels, stride=2)]
        layers.extend(CatBottleneck(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x2 = self.stem2(x)
        x4 = self.stem4(x2)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return [x2, x4, x8, x16, x32]


class SimplePyramidPooling(nn.Module):
    """Official PP-LiteSeg simple pyramid pooling context module."""

    def __init__(self, in_channels: int, out_channels: int, bin_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    ConvBNReLU(in_channels, out_channels, kernel_size=1),
                )
                for size in bin_sizes
            ]
        )
        self.output = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [
            F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.output(torch.stack(pooled, dim=0).sum(dim=0))


class UnifiedAttentionFusion(nn.Module):
    """Spatial mean/max UAFM from the default PP-LiteSeg decoder."""

    def __init__(self, low_channels: int, high_channels: int, out_channels: int) -> None:
        super().__init__()
        self.low_projection = ConvBNReLU(low_channels, high_channels)
        self.attention = nn.Sequential(ConvBNReLU(4, 2), ConvBN(2, 1))
        self.output = ConvBNReLU(high_channels, out_channels)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        low = self.low_projection(low)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        statistics = torch.cat(
            [
                low.mean(dim=1, keepdim=True),
                low.amax(dim=1, keepdim=True),
                high.mean(dim=1, keepdim=True),
                high.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        attention = torch.sigmoid(self.attention(statistics))
        return self.output(low * attention + high * (1.0 - attention))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, intermediate_channels: int, num_classes: int) -> None:
        super().__init__(
            ConvBNReLU(in_channels, intermediate_channels),
            nn.Conv2d(intermediate_channels, num_classes, 1, bias=False),
        )


class PPLiteSegSTDC1(nn.Module):
    """PP-LiteSeg-T using STDC1 and the official lightweight channel settings."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = STDC1Backbone()
        self.context = SimplePyramidPooling(1024, 128, (1, 2, 4))
        self.fusion32 = UnifiedAttentionFusion(1024, 128, 128)
        self.fusion16 = UnifiedAttentionFusion(512, 128, 64)
        self.fusion8 = UnifiedAttentionFusion(256, 64, 32)
        self.head8 = SegmentationHead(32, 32, num_classes)
        self.head16 = SegmentationHead(64, 64, num_classes)
        self.head32 = SegmentationHead(128, 64, num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _upsample(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> ModelOutput:
        output_size = image.shape[-2:]
        _, _, x8, x16, x32 = self.backbone(image)
        context = self.context(x32)
        decoded32 = self.fusion32(x32, context)
        decoded16 = self.fusion16(x16, decoded32)
        decoded8 = self.fusion8(x8, decoded16)
        return ModelOutput(
            logits=self._upsample(self.head8(decoded8), output_size),
            features={"kd": decoded8},
            aux_logits={
                "x16": self._upsample(self.head16(decoded16), output_size),
                "x32": self._upsample(self.head32(decoded32), output_size),
            },
        )
