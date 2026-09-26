"""PIDNet-S adapted from Jiacong Xu's official MIT-licensed implementation.

Architecture/module names follow the reference implementation so compatible
weights can be imported. Only the return type and final upsampling are adapted
to FM2Edge's common Student interface.

Source: https://github.com/XuJiacong/PIDNet
License copy: third_party_licenses/PIDNET_LICENSE.txt
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fm2edge.models.outputs import ModelOutput

BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        output = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + residual
        return output if self.no_relu else self.relu(output)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output)) + residual
        return output if self.no_relu else self.relu(output)


class SegmentHead(nn.Module):
    def __init__(self, inplanes: int, interplanes: int, outplanes: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu(self.bn1(x)))
        return self.conv2(self.relu(self.bn2(x)))


class PagFM(nn.Module):
    def __init__(self, channels: int, mid_channels: int) -> None:
        super().__init__()
        self.f_x = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_query = F.interpolate(
            self.f_y(y), size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        similarity = torch.sigmoid((self.f_x(x) * y_query).sum(dim=1, keepdim=True))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return (1.0 - similarity) * x + similarity * y


class PAPPM(nn.Module):
    def __init__(self, inplanes: int, branch_planes: int, outplanes: int) -> None:
        super().__init__()

        def scale(pool: nn.Module) -> nn.Sequential:
            return nn.Sequential(
                pool,
                nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, branch_planes, 1, bias=False),
            )

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False),
        )
        self.scale1 = scale(nn.AvgPool2d(5, 2, 2))
        self.scale2 = scale(nn.AvgPool2d(9, 4, 4))
        self.scale3 = scale(nn.AvgPool2d(17, 8, 8))
        self.scale4 = scale(nn.AdaptiveAvgPool2d((1, 1)))
        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4, branch_planes * 4, 3, padding=1, groups=4, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        base = self.scale0(x)
        scales = [
            F.interpolate(layer(x), size=size, mode="bilinear", align_corners=False) + base
            for layer in (self.scale1, self.scale2, self.scale3, self.scale4)
        ]
        processed = self.scale_process(torch.cat(scales, dim=1))
        return self.compression(torch.cat([base, processed], dim=1)) + self.shortcut(x)


class LightBag(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        edge = torch.sigmoid(d)
        return self.conv_p((1.0 - edge) * i + p) + self.conv_i(i + edge * p)


class PIDNetSmall(nn.Module):
    """Official PIDNet-S dimensions: m=2, n=3, planes=32."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        m, n, planes, ppm_planes, head_planes = 2, 3, 32, 96, 128
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, 2, 1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 2, 1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, 2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, 2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, 2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, 2)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.pag3, self.pag4 = PagFM(planes * 2, planes), PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
        self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        self.diff3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
        self.dfm = LightBag(planes * 4)
        self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
        self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)
        self._initialize()

    @staticmethod
    def _downsample(inplanes: int, outplanes: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM),
        )

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        outplanes = planes * block.expansion
        downsample = (
            self._downsample(inplanes, outplanes, stride)
            if stride != 1 or inplanes != outplanes
            else None
        )
        layers: list[nn.Module] = [block(inplanes, planes, stride, downsample)]
        layers.extend(
            block(outplanes, planes, no_relu=index == blocks - 1) for index in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _make_single_layer(
        self, block: type[BasicBlock | Bottleneck], inplanes: int, planes: int, stride: int = 1
    ) -> nn.Module:
        outplanes = planes * block.expansion
        downsample = (
            self._downsample(inplanes, outplanes, stride)
            if stride != 1 or inplanes != outplanes
            else None
        )
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> ModelOutput:
        output_size = image.shape[-2:]
        feature_size = (image.shape[-2] // 8, image.shape[-1] // 8)
        x = self.layer1(self.conv1(image))
        x = self.relu(self.layer2(self.relu(x)))
        p, d = self.layer3_(x), self.layer3_d(x)
        x = self.relu(self.layer3(x))
        p = self.pag3(p, self.compression3(x))
        d = d + F.interpolate(
            self.diff3(x), size=feature_size, mode="bilinear", align_corners=False
        )
        auxiliary_feature = p
        x = self.relu(self.layer4(x))
        p, d = self.layer4_(self.relu(p)), self.layer4_d(self.relu(d))
        p = self.pag4(p, self.compression4(x))
        d = d + F.interpolate(
            self.diff4(x), size=feature_size, mode="bilinear", align_corners=False
        )
        boundary_feature = d
        p, d = self.layer5_(self.relu(p)), self.layer5_d(self.relu(d))
        x = F.interpolate(
            self.spp(self.layer5(x)), size=feature_size, mode="bilinear", align_corners=False
        )
        fused = self.dfm(p, x, d)
        logits = F.interpolate(
            self.final_layer(fused), size=output_size, mode="bilinear", align_corners=False
        )
        auxiliary = F.interpolate(
            self.seghead_p(auxiliary_feature),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        boundary = F.interpolate(
            self.seghead_d(boundary_feature), size=output_size, mode="bilinear", align_corners=False
        )
        return ModelOutput(
            logits=logits,
            features={"kd": fused},
            aux_logits={"segmentation": auxiliary, "boundary": boundary},
        )
