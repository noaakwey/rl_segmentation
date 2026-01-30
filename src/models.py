"""
Neural network models for supervised segmentation.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UNet(nn.Module):
    """
    Lightweight U-Net for supervised segmentation.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            ConvBlock(base_channels * 16, base_channels * 16)
        )

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ConvBlock(base_channels * 16, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels)
        )

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)
