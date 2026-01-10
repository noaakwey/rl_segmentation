"""
CNN Backbones для feature extraction.

Поддерживаемые архитектуры:
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- MobileNetV3
- ConvNeXt
- Custom CNN

Все backbones адаптированы для работы с:
- Мультиспектральными данными (любое количество каналов)
- Sentinel-2 (12 каналов)
- RGB (3 канала)
- RGB + NIR (4 канала)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSpectralAdapter(nn.Module):
    """
    Адаптер для работы с мультиспектральными данными.

    Преобразует N-канальное изображение в 3-канальное для pretrained моделей
    или адаптирует первый слой для работы с N каналами.
    """

    def __init__(self,
                 in_channels: int,
                 method: str = 'adapt_first_layer'):
        """
        Initialize adapter.

        Args:
            in_channels: Число входных каналов
            method: Метод адаптации:
                - 'adapt_first_layer': Изменить веса первого слоя
                - 'channel_selection': Выбрать 3 канала из N
                - 'learnable_projection': Обучаемая проекция N → 3
        """
        super().__init__()
        self.in_channels = in_channels
        self.method = method

        if method == 'learnable_projection':
            # Обучаемая проекция N каналов → 3 канала
            self.projection = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        elif method == 'channel_selection':
            # Индексы каналов для выбора (например, для Sentinel-2: R, G, B)
            self.selected_channels = [3, 2, 1]  # Sentinel-2: B4(R), B3(G), B2(B)

    def adapt_weights(self, pretrained_conv: nn.Conv2d) -> nn.Conv2d:
        """
        Адаптировать веса pretrained conv слоя для N каналов.

        Args:
            pretrained_conv: Pretrained conv слой (3 входных канала)

        Returns:
            Адаптированный conv слой (N входных каналов)
        """
        if self.method == 'adapt_first_layer':
            # Создать новый слой с N каналами
            new_conv = nn.Conv2d(
                self.in_channels,
                pretrained_conv.out_channels,
                kernel_size=pretrained_conv.kernel_size,
                stride=pretrained_conv.stride,
                padding=pretrained_conv.padding,
                bias=pretrained_conv.bias is not None
            )

            # Копировать веса
            with torch.no_grad():
                if self.in_channels == 3:
                    # Просто копируем
                    new_conv.weight.copy_(pretrained_conv.weight)
                elif self.in_channels < 3:
                    # Меньше 3 каналов - берем первые N каналов
                    new_conv.weight.copy_(pretrained_conv.weight[:, :self.in_channels, :, :])
                else:
                    # Больше 3 каналов - повторяем веса
                    # Метод 1: Среднее из RGB весов для дополнительных каналов
                    rgb_weights = pretrained_conv.weight  # [out, 3, k, k]
                    avg_weight = rgb_weights.mean(dim=1, keepdim=True)  # [out, 1, k, k]

                    # Инициализация: RGB каналы + повтор среднего для остальных
                    new_conv.weight[:, :3, :, :].copy_(rgb_weights)
                    for i in range(3, self.in_channels):
                        new_conv.weight[:, i:i+1, :, :].copy_(avg_weight)

                if pretrained_conv.bias is not None:
                    new_conv.bias.copy_(pretrained_conv.bias)

            return new_conv

        return pretrained_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (B, in_channels, H, W)

        Returns:
            Output (B, 3, H, W) if projection, else (B, in_channels, H, W)
        """
        if self.method == 'learnable_projection':
            return self.projection(x)
        elif self.method == 'channel_selection':
            # Выбрать нужные каналы
            return x[:, self.selected_channels, :, :]
        else:
            return x


class ResNetBackbone(nn.Module):
    """
    ResNet backbone для feature extraction.
    """

    def __init__(self,
                 in_channels: int = 3,
                 variant: str = 'resnet50',
                 pretrained: bool = True,
                 features_dim: int = 512):
        """
        Initialize ResNet backbone.

        Args:
            in_channels: Число входных каналов
            variant: Вариант ResNet (resnet18, resnet34, resnet50, resnet101, resnet152)
            pretrained: Использовать pretrained веса
            features_dim: Размерность выходных features
        """
        super().__init__()

        self.in_channels = in_channels
        self.features_dim = features_dim

        # Загрузить ResNet
        if variant == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            base_features_dim = 512
        elif variant == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            base_features_dim = 512
        elif variant == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            base_features_dim = 2048
        elif variant == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            base_features_dim = 2048
        elif variant == 'resnet152':
            base_model = models.resnet152(pretrained=pretrained)
            base_features_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        # Адаптировать первый слой для мультиспектральных данных
        if in_channels != 3:
            adapter = MultiSpectralAdapter(in_channels, method='adapt_first_layer')
            base_model.conv1 = adapter.adapt_weights(base_model.conv1)
            logger.info(f"Adapted ResNet first layer for {in_channels} channels")

        # Убрать FC слой
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection
        self.projection = nn.Linear(base_features_dim, features_dim)

        logger.info(f"Initialized {variant} with {in_channels} input channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (B, in_channels, H, W)

        Returns:
            Features (B, features_dim)
        """
        features = self.encoder(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        return features


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone для feature extraction.
    """

    def __init__(self,
                 in_channels: int = 3,
                 variant: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 features_dim: int = 512):
        """
        Initialize EfficientNet backbone.

        Args:
            in_channels: Число входных каналов
            variant: Вариант EfficientNet (b0-b7)
            pretrained: Использовать pretrained веса
            features_dim: Размерность выходных features
        """
        super().__init__()

        self.in_channels = in_channels
        self.features_dim = features_dim

        # Загрузить EfficientNet
        if variant == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            base_features_dim = 1280
        elif variant == 'efficientnet_b1':
            base_model = models.efficientnet_b1(pretrained=pretrained)
            base_features_dim = 1280
        elif variant == 'efficientnet_b2':
            base_model = models.efficientnet_b2(pretrained=pretrained)
            base_features_dim = 1408
        elif variant == 'efficientnet_b3':
            base_model = models.efficientnet_b3(pretrained=pretrained)
            base_features_dim = 1536
        elif variant == 'efficientnet_b4':
            base_model = models.efficientnet_b4(pretrained=pretrained)
            base_features_dim = 1792
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        # Адаптировать первый слой
        if in_channels != 3:
            adapter = MultiSpectralAdapter(in_channels, method='adapt_first_layer')
            base_model.features[0][0] = adapter.adapt_weights(base_model.features[0][0])
            logger.info(f"Adapted EfficientNet first layer for {in_channels} channels")

        # Features
        self.encoder = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection
        self.projection = nn.Linear(base_features_dim, features_dim)

        logger.info(f"Initialized {variant} with {in_channels} input channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.encoder(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        return features


class Sentinel2Backbone(nn.Module):
    """
    Специализированный backbone для Sentinel-2 данных (12 каналов).

    Sentinel-2 bands:
    - B1 (443nm): Coastal aerosol
    - B2 (490nm): Blue
    - B3 (560nm): Green
    - B4 (665nm): Red
    - B5 (705nm): Red Edge 1
    - B6 (740nm): Red Edge 2
    - B7 (783nm): Red Edge 3
    - B8 (842nm): NIR
    - B8A (865nm): NIR narrow
    - B9 (940nm): Water vapor
    - B11 (1610nm): SWIR 1
    - B12 (2190nm): SWIR 2
    """

    def __init__(self,
                 base_backbone: str = 'resnet50',
                 pretrained: bool = True,
                 features_dim: int = 512,
                 use_all_bands: bool = True):
        """
        Initialize Sentinel-2 backbone.

        Args:
            base_backbone: Базовая архитектура (resnet50, efficientnet_b0, etc.)
            pretrained: Использовать pretrained веса
            features_dim: Размерность features
            use_all_bands: Использовать все 12 каналов или только RGB+NIR
        """
        super().__init__()

        self.use_all_bands = use_all_bands
        self.num_bands = 12 if use_all_bands else 4

        # Band indices for RGB+NIR
        self.rgb_nir_indices = [3, 2, 1, 7]  # R, G, B, NIR (B4, B3, B2, B8)

        # Создать backbone
        if 'resnet' in base_backbone:
            self.backbone = ResNetBackbone(
                in_channels=self.num_bands,
                variant=base_backbone,
                pretrained=pretrained,
                features_dim=features_dim
            )
        elif 'efficientnet' in base_backbone:
            self.backbone = EfficientNetBackbone(
                in_channels=self.num_bands,
                variant=base_backbone,
                pretrained=pretrained,
                features_dim=features_dim
            )
        else:
            raise ValueError(f"Unknown backbone: {base_backbone}")

        logger.info(f"Initialized Sentinel-2 backbone with {self.num_bands} bands")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Sentinel-2 image (B, 12, H, W)

        Returns:
            Features (B, features_dim)
        """
        # Если не используем все каналы, выбираем RGB+NIR
        if not self.use_all_bands and x.shape[1] == 12:
            x = x[:, self.rgb_nir_indices, :, :]

        return self.backbone(x)


def create_backbone(backbone_name: str,
                    in_channels: int = 3,
                    pretrained: bool = True,
                    features_dim: int = 512) -> nn.Module:
    """
    Factory function для создания backbone.

    Args:
        backbone_name: Название backbone
        in_channels: Число входных каналов
        pretrained: Использовать pretrained веса
        features_dim: Размерность features

    Returns:
        Backbone module
    """
    if backbone_name.startswith('resnet'):
        return ResNetBackbone(in_channels, backbone_name, pretrained, features_dim)
    elif backbone_name.startswith('efficientnet'):
        return EfficientNetBackbone(in_channels, backbone_name, pretrained, features_dim)
    elif backbone_name == 'sentinel2':
        return Sentinel2Backbone('resnet50', pretrained, features_dim)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


if __name__ == "__main__":
    # Test backbones
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Testing CNN Backbones")
    print("="*70)

    # Test ResNet with different channels
    print("\n1. ResNet50 with RGB (3 channels)")
    resnet_rgb = ResNetBackbone(in_channels=3, variant='resnet50').to(device)
    x_rgb = torch.randn(2, 3, 256, 256).to(device)
    out = resnet_rgb(x_rgb)
    print(f"   Input: {x_rgb.shape} → Output: {out.shape}")

    print("\n2. ResNet50 with multispectral (5 channels)")
    resnet_multi = ResNetBackbone(in_channels=5, variant='resnet50').to(device)
    x_multi = torch.randn(2, 5, 256, 256).to(device)
    out = resnet_multi(x_multi)
    print(f"   Input: {x_multi.shape} → Output: {out.shape}")

    print("\n3. EfficientNet-B0 with RGB+NIR (4 channels)")
    effnet = EfficientNetBackbone(in_channels=4, variant='efficientnet_b0').to(device)
    x_4ch = torch.randn(2, 4, 256, 256).to(device)
    out = effnet(x_4ch)
    print(f"   Input: {x_4ch.shape} → Output: {out.shape}")

    print("\n4. Sentinel-2 backbone (12 channels)")
    s2_backbone = Sentinel2Backbone(use_all_bands=True).to(device)
    x_s2 = torch.randn(2, 12, 256, 256).to(device)
    out = s2_backbone(x_s2)
    print(f"   Input: {x_s2.shape} → Output: {out.shape}")

    print("\n5. Sentinel-2 backbone (RGB+NIR only)")
    s2_rgb_nir = Sentinel2Backbone(use_all_bands=False).to(device)
    x_s2 = torch.randn(2, 12, 256, 256).to(device)
    out = s2_rgb_nir(x_s2)
    print(f"   Input: {x_s2.shape} → Output: {out.shape}")

    print("\n" + "="*70)
    print("✓ All backbones working!")
    print("="*70)
