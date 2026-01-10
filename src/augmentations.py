"""
Data augmentation для спутниковых снимков.

Поддерживаются:
- Geometric transformations (rotate, flip, crop)
- Color augmentations (brightness, contrast, gamma)
- Noise augmentations (gaussian, speckle)
- Advanced augmentations (mixup, cutmix)
- Sentinel-specific augmentations
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple
import torch


class SatelliteAugmentation:
    """
    Augmentation pipeline для спутниковых снимков.
    """

    def __init__(self,
                 mode: str = 'train',
                 image_size: int = 256,
                 normalize: bool = True):
        """
        Initialize augmentation pipeline.

        Args:
            mode: 'train' или 'val'
            image_size: Размер выходного изображения
            normalize: Нормализовать ли данные
        """
        self.mode = mode
        self.image_size = image_size
        self.normalize = normalize

        if mode == 'train':
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()

    def _get_train_transforms(self) -> A.Compose:
        """Get training augmentations."""
        transforms = [
            # Geometric transforms
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=0,
                p=0.5
            ),

            # Crops and pads
            A.RandomCrop(height=self.image_size, width=self.image_size, p=1.0),

            # Color augmentations (осторожно с мультиспектральными!)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=1.0
                ),
            ], p=0.3),

            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.2),

            # Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.1),

            # Advanced
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.2
            ),
        ]

        if self.normalize:
            transforms.append(A.Normalize(mean=0.0, std=1.0))

        return A.Compose(transforms)

    def _get_val_transforms(self) -> A.Compose:
        """Get validation transforms (только crop/resize)."""
        transforms = [
            A.CenterCrop(height=self.image_size, width=self.image_size, p=1.0),
        ]

        if self.normalize:
            transforms.append(A.Normalize(mean=0.0, std=1.0))

        return A.Compose(transforms)

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Apply augmentations.

        Args:
            image: Image array (H, W, C) или (C, H, W)
            mask: Mask array (H, W)

        Returns:
            Dictionary with augmented image and mask
        """
        # Convert to (H, W, C) if needed
        if image.ndim == 3 and image.shape[0] <= 12:  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))

        # Apply transforms
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            result = {
                'image': augmented['image'],
                'mask': augmented['mask']
            }
        else:
            augmented = self.transform(image=image)
            result = {'image': augmented['image']}

        # Convert back to (C, H, W)
        if result['image'].ndim == 3:
            result['image'] = np.transpose(result['image'], (2, 0, 1))

        return result


class Sentinel2Augmentation:
    """
    Специализированная аугментация для Sentinel-2.

    Учитывает особенности мультиспектральных данных:
    - Не применяет color augmentation к ИК каналам
    - Сохраняет физический смысл спектральных индексов
    """

    def __init__(self,
                 mode: str = 'train',
                 image_size: int = 256,
                 use_all_bands: bool = True):
        """
        Initialize Sentinel-2 augmentation.

        Args:
            mode: 'train' или 'val'
            image_size: Размер изображения
            use_all_bands: Использовать все 12 каналов
        """
        self.mode = mode
        self.image_size = image_size
        self.use_all_bands = use_all_bands

        # Индексы RGB каналов в Sentinel-2
        self.rgb_indices = [3, 2, 1]  # B4 (R), B3 (G), B2 (B)

        if mode == 'train':
            self.transform = self._get_train_transforms()
        else:
            self.transform = self._get_val_transforms()

    def _get_train_transforms(self) -> A.Compose:
        """Get training transforms for Sentinel-2."""
        # Только geometric transforms - безопасны для всех каналов
        transforms = [
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=0,
                p=0.5
            ),
            A.RandomCrop(height=self.image_size, width=self.image_size, p=1.0),

            # Mild noise (применяется ко всем каналам)
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),

            # Коррупция отдельных каналов (имитация облаков, теней)
            A.CoarseDropout(
                max_holes=5,
                max_height=64,
                max_width=64,
                fill_value=0,
                p=0.15
            ),
        ]

        return A.Compose(transforms)

    def _get_val_transforms(self) -> A.Compose:
        """Get validation transforms."""
        return A.Compose([
            A.CenterCrop(height=self.image_size, width=self.image_size, p=1.0),
        ])

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Apply Sentinel-2 specific augmentations.

        Args:
            image: Sentinel-2 image (12, H, W) или (H, W, 12)
            mask: Mask (H, W)

        Returns:
            Augmented data
        """
        # Convert to (H, W, C)
        if image.ndim == 3 and image.shape[0] <= 12:
            image = np.transpose(image, (1, 2, 0))

        # Apply geometric transforms
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            result = {
                'image': augmented['image'],
                'mask': augmented['mask']
            }
        else:
            augmented = self.transform(image=image)
            result = {'image': augmented['image']}

        # Если нужно, применить color augmentation ТОЛЬКО к RGB каналам
        if self.mode == 'train' and np.random.random() < 0.3:
            # Extract RGB channels
            rgb = result['image'][:, :, self.rgb_indices]

            # Apply color augmentation
            color_aug = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=1.0
                )
            ])

            rgb_augmented = color_aug(image=rgb)['image']

            # Put back RGB channels
            result['image'][:, :, self.rgb_indices] = rgb_augmented

        # Convert back to (C, H, W)
        result['image'] = np.transpose(result['image'], (2, 0, 1))

        return result


class MixUpAugmentation:
    """
    MixUp augmentation для улучшения генерализации.

    MixUp создает виртуальные обучающие примеры путем
    линейной интерполяции между парами примеров.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.

        Args:
            alpha: Параметр Beta distribution
        """
        self.alpha = alpha

    def __call__(self,
                 image1: np.ndarray,
                 mask1: np.ndarray,
                 image2: np.ndarray,
                 mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply MixUp.

        Args:
            image1, mask1: Первый пример
            image2, mask2: Второй пример

        Returns:
            Mixed image and mask
        """
        # Sample lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_mask = lam * mask1 + (1 - lam) * mask2

        return mixed_image.astype(image1.dtype), mixed_mask.astype(mask1.dtype)


class CutMixAugmentation:
    """
    CutMix augmentation - вырезает регион из одного изображения
    и вставляет в другое.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.

        Args:
            alpha: Параметр Beta distribution
        """
        self.alpha = alpha

    def __call__(self,
                 image1: np.ndarray,
                 mask1: np.ndarray,
                 image2: np.ndarray,
                 mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix.

        Args:
            image1, mask1: Первый пример
            image2, mask2: Второй пример

        Returns:
            CutMix image and mask
        """
        # Get image size
        _, H, W = image1.shape

        # Sample lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Sample random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)

        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Box boundaries
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Copy
        mixed_image = image1.copy()
        mixed_mask = mask1.copy()

        mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

        return mixed_image, mixed_mask


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) для улучшения inference.

    Применяет несколько аугментаций, делает предсказания
    и усредняет результаты.
    """

    def __init__(self, n_augments: int = 8):
        """
        Initialize TTA.

        Args:
            n_augments: Число аугментаций
        """
        self.n_augments = n_augments

        # Define TTA transforms
        self.transforms = [
            A.Compose([]),  # Identity
            A.Compose([A.HorizontalFlip(p=1.0)]),
            A.Compose([A.VerticalFlip(p=1.0)]),
            A.Compose([A.Rotate(limit=90, p=1.0)]),
            A.Compose([A.Rotate(limit=180, p=1.0)]),
            A.Compose([A.Rotate(limit=270, p=1.0)]),
            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=90, p=1.0)]),
            A.Compose([A.VerticalFlip(p=1.0), A.Rotate(limit=90, p=1.0)]),
        ]

    def augment_batch(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate augmented versions.

        Args:
            image: Input image (C, H, W)

        Returns:
            List of augmented images
        """
        augmented = []

        # Convert to (H, W, C)
        if image.ndim == 3 and image.shape[0] <= 12:
            image = np.transpose(image, (1, 2, 0))

        for transform in self.transforms[:self.n_augments]:
            aug_image = transform(image=image)['image']

            # Convert back to (C, H, W)
            if aug_image.ndim == 3:
                aug_image = np.transpose(aug_image, (2, 0, 1))

            augmented.append(aug_image)

        return augmented

    def deaugment_predictions(self,
                             predictions: List[np.ndarray],
                             method: str = 'mean') -> np.ndarray:
        """
        Aggregate predictions from augmented images.

        Args:
            predictions: List of predictions (each H, W)
            method: Aggregation method ('mean', 'max', 'vote')

        Returns:
            Aggregated prediction (H, W)
        """
        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # (N, H, W)

        if method == 'mean':
            return predictions.mean(axis=0)
        elif method == 'max':
            return predictions.max(axis=0)
        elif method == 'vote':
            # Binary voting
            return (predictions > 0.5).mean(axis=0) > 0.5
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


def get_augmentation_pipeline(data_type: str = 'rgb',
                              mode: str = 'train',
                              image_size: int = 256) -> callable:
    """
    Factory function для создания augmentation pipeline.

    Args:
        data_type: Тип данных ('rgb', 'multispectral', 'sentinel2')
        mode: 'train' или 'val'
        image_size: Размер изображения

    Returns:
        Augmentation callable
    """
    if data_type == 'rgb':
        return SatelliteAugmentation(mode=mode, image_size=image_size)
    elif data_type == 'sentinel2':
        return Sentinel2Augmentation(mode=mode, image_size=image_size)
    elif data_type == 'multispectral':
        return SatelliteAugmentation(mode=mode, image_size=image_size)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


if __name__ == "__main__":
    print("="*70)
    print("Testing Augmentations")
    print("="*70)

    # Test standard augmentation
    print("\n1. Testing SatelliteAugmentation (RGB)")
    aug_rgb = SatelliteAugmentation(mode='train', image_size=256)

    image_rgb = np.random.rand(3, 512, 512).astype(np.float32)
    mask = (np.random.rand(512, 512) > 0.5).astype(np.float32)

    result = aug_rgb(image_rgb, mask)
    print(f"   Input: {image_rgb.shape} → Output: {result['image'].shape}")
    print(f"   Mask: {mask.shape} → Output: {result['mask'].shape}")

    # Test Sentinel-2 augmentation
    print("\n2. Testing Sentinel2Augmentation (12 channels)")
    aug_s2 = Sentinel2Augmentation(mode='train', image_size=256)

    image_s2 = np.random.rand(12, 512, 512).astype(np.float32)
    result = aug_s2(image_s2, mask)
    print(f"   Input: {image_s2.shape} → Output: {result['image'].shape}")

    # Test MixUp
    print("\n3. Testing MixUp")
    mixup = MixUpAugmentation(alpha=0.2)

    image1 = np.random.rand(3, 256, 256).astype(np.float32)
    mask1 = np.random.rand(256, 256).astype(np.float32)
    image2 = np.random.rand(3, 256, 256).astype(np.float32)
    mask2 = np.random.rand(256, 256).astype(np.float32)

    mixed_img, mixed_mask = mixup(image1, mask1, image2, mask2)
    print(f"   Mixed image: {mixed_img.shape}, mask: {mixed_mask.shape}")

    # Test CutMix
    print("\n4. Testing CutMix")
    cutmix = CutMixAugmentation(alpha=1.0)

    cut_img, cut_mask = cutmix(image1, mask1, image2, mask2)
    print(f"   CutMix image: {cut_img.shape}, mask: {cut_mask.shape}")

    # Test TTA
    print("\n5. Testing Test-Time Augmentation")
    tta = TestTimeAugmentation(n_augments=4)

    test_image = np.random.rand(3, 256, 256).astype(np.float32)
    augmented_batch = tta.augment_batch(test_image)
    print(f"   Generated {len(augmented_batch)} augmented versions")

    # Simulate predictions
    predictions = [np.random.rand(256, 256) for _ in range(len(augmented_batch))]
    final_pred = tta.deaugment_predictions(predictions, method='mean')
    print(f"   Aggregated prediction: {final_pred.shape}")

    print("\n" + "="*70)
    print("✓ All augmentations working!")
    print("="*70)
