"""Shared multi-scale image encoder used by PUMA-MV."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from multiview_detector.models.resnet import resnet18


def _group_count(channels, maximum=8):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResNet18FPNEncoder(nn.Module):
    """Return stride-8/16/32 feature maps with top-down semantic fusion."""

    def __init__(self, out_channels=32, pretrained=False):
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        backbone = resnet18(
            pretrained=pretrained,
            replace_stride_with_dilation=[False, False, False],
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lateral3 = nn.Conv2d(128, out_channels, 1)
        self.lateral4 = nn.Conv2d(256, out_channels, 1)
        self.lateral5 = nn.Conv2d(512, out_channels, 1)
        groups = _group_count(out_channels)
        self.output3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.output4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.output5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, image):
        feature = self.layer1(self.stem(image))
        c3 = self.layer2(feature)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode="bilinear", align_corners=False
        )
        p3 = self.lateral3(c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.output3(p3), self.output4(p4), self.output5(p5)
