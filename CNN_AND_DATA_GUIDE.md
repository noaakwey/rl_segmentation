# 🎯 CNN Backbones и работа с данными - Полное руководство

Ответы на ключевые вопросы: какая CNN в основе, форматы данных, поддержка мультиспектральных данных, аугментация.

## 1. 🧠 CNN в основе - гибкий выбор

### Поддерживаемые архитектуры

Система поддерживает **любые современные CNN** в качестве feature extractor:

#### ResNet (рекомендуется)

```python
from src.backbones import ResNetBackbone

# ResNet-18 (легкий)
backbone = ResNetBackbone(
    in_channels=4,
    variant='resnet18',
    pretrained=True,
    features_dim=512
)

# ResNet-50 (стандарт)
backbone = ResNetBackbone(
    in_channels=4,
    variant='resnet50',
    pretrained=True
)

# ResNet-101 (мощный)
backbone = ResNetBackbone(
    in_channels=4,
    variant='resnet101',
    pretrained=True
)
```

#### EfficientNet (эффективный)

```python
from src.backbones import EfficientNetBackbone

# EfficientNet-B0 (быстрый)
backbone = EfficientNetBackbone(
    in_channels=4,
    variant='efficientnet_b0',
    pretrained=True
)

# EfficientNet-B4 (баланс)
backbone = EfficientNetBackbone(
    in_channels=4,
    variant='efficientnet_b4',
    pretrained=True
)
```

#### Специализированный для Sentinel-2

```python
from src.backbones import Sentinel2Backbone

# Автоматически адаптирован для 12 каналов Sentinel-2
backbone = Sentinel2Backbone(
    base_backbone='resnet50',
    pretrained=True,
    use_all_bands=True  # Все 12 каналов
)
```

### Интеграция в RL агент

```python
from src.advanced_agent import AdvancedRLAgent
from src.hierarchical_agent import HierarchicalSegmentationPolicy

# Вариант 1: Автоматический выбор backbone
agent = AdvancedRLAgent(
    in_channels=12,  # Sentinel-2
    device='cuda'
)

# Вариант 2: Кастомный backbone
from src.backbones import create_backbone

custom_backbone = create_backbone(
    backbone_name='resnet101',
    in_channels=12,
    pretrained=True
)

# Использовать в hierarchical policy
policy = HierarchicalSegmentationPolicy(
    in_channels=12,
    num_coarse_actions=16
)
# Policy.feature_extractor можно заменить на custom_backbone
```

### Сравнение backbones

| Backbone | Params | Speed | Quality | Рекомендуется для |
|----------|--------|-------|---------|-------------------|
| ResNet-18 | 11M | ⚡⚡⚡ | ⭐⭐ | Быстрые эксперименты |
| ResNet-50 | 25M | ⚡⚡ | ⭐⭐⭐ | **Стандарт (рекомендуется)** |
| ResNet-101 | 44M | ⚡ | ⭐⭐⭐⭐ | Максимальное качество |
| EfficientNet-B0 | 5M | ⚡⚡⚡ | ⭐⭐⭐ | Мобильные устройства |
| EfficientNet-B4 | 19M | ⚡⚡ | ⭐⭐⭐⭐ | Баланс качества/скорости |

---

## 2. 📁 Форматы входных данных

### Поддерживаемые форматы изображений

#### GeoTIFF (.tif, .tiff) - основной формат

```python
from src.data_loader import GeoDataLoader

loader = GeoDataLoader(
    image_path="satellite_image.tif",
    shapefile_path="crop_boundaries.shp",
    patch_size=256,
    stride=128,
    normalize=True
)
```

**Поддерживаются:**
- ✅ 1-20+ каналов (RGB, multispectral, hyperspectral)
- ✅ uint8, uint16, float32, float64
- ✅ Любые проекции (автоматическая репроекция)
- ✅ Compressed (LZW, DEFLATE, JPEG)
- ✅ Cloud Optimized GeoTIFF (COG)

#### Shapefile (.shp) - векторные границы

**Требуется полный набор файлов:**
```
boundaries/
├── boundaries.shp      # Геометрия
├── boundaries.shx      # Индекс
├── boundaries.dbf      # Атрибуты
└── boundaries.prj      # Проекция
```

**Автоматическая конвертация:**
```python
# GeoDataLoader автоматически:
# 1. Репроектирует shapefile в CRS изображения
# 2. Конвертирует векторные полигоны в растровую маску
# 3. Выравнивает bounds

loader = GeoDataLoader(image_path="image.tif", shapefile_path="fields.shp")
mask = loader.create_mask()  # Растровая маска
```

### Структура данных (рекомендуемая)

```
data/
├── raw/
│   ├── region1/
│   │   ├── sentinel2_20230615.tif      # 12 каналов
│   │   └── fields_2023.shp
│   ├── region2/
│   │   ├── landsat8_20230701.tif       # 6 каналов
│   │   └── fields_2023.shp
│   └── aerial/
│       ├── drone_rgb_nir.tif           # 4 канала (RGB+NIR)
│       └── fields.shp
│
└── processed/
    ├── patches/        # Извлеченные патчи
    └── augmented/      # Аугментированные данные
```

---

## 3. 🛰️ Мультиспектральные данные - полная поддержка

### Sentinel-2 (12 каналов)

**Автоматическая поддержка всех 12 каналов:**

```python
from src.backbones import Sentinel2Backbone
from src.augmentations import Sentinel2Augmentation

# Backbone для Sentinel-2
backbone = Sentinel2Backbone(
    base_backbone='resnet50',
    use_all_bands=True,  # Все 12 каналов
    pretrained=True
)

# Sentinel-2 specific augmentation
aug = Sentinel2Augmentation(
    mode='train',
    image_size=256,
    use_all_bands=True
)

# Загрузка Sentinel-2
loader = GeoDataLoader(
    image_path="sentinel2_L2A.tif",  # 12 каналов
    shapefile_path="fields.shp"
)

image, mask = loader.load_all()
print(f"Sentinel-2 shape: {image.shape}")  # (12, H, W)
```

**Sentinel-2 bands:**
```
B1  (443nm)  - Coastal aerosol - 60m
B2  (490nm)  - Blue            - 10m
B3  (560nm)  - Green           - 10m
B4  (665nm)  - Red             - 10m
B5  (705nm)  - Red Edge 1      - 20m
B6  (740nm)  - Red Edge 2      - 20m
B7  (783nm)  - Red Edge 3      - 20m
B8  (842nm)  - NIR             - 10m
B8A (865nm)  - NIR narrow      - 20m
B9  (940nm)  - Water vapor     - 60m
B11 (1610nm) - SWIR 1          - 20m
B12 (2190nm) - SWIR 2          - 20m
```

### Landsat 8/9 (6-11 каналов)

```python
# Обычно используют B2-B7 (6 каналов)
from src.backbones import ResNetBackbone

backbone = ResNetBackbone(
    in_channels=6,  # B2, B3, B4, B5, B6, B7
    variant='resnet50',
    pretrained=True
)
```

### Planet / Aerial (4 канала: RGB+NIR)

```python
backbone = ResNetBackbone(
    in_channels=4,  # RGB + NIR
    variant='resnet50',
    pretrained=True
)
```

### Любое количество каналов

```python
# Система поддерживает ЛЮБОЕ количество каналов
backbone = ResNetBackbone(
    in_channels=20,  # Даже гиперспектральные данные!
    variant='resnet50',
    pretrained=True
)
```

### Адаптация pretrained весов

**Автоматическая адаптация для N каналов:**

```python
# Pretrained на ImageNet (3 канала) → N каналов

# Метод 1: Среднее из RGB для доп. каналов (используется по умолчанию)
backbone = ResNetBackbone(in_channels=12, pretrained=True)
# RGB веса копируются напрямую
# Каналы 4-12: инициализируются средним из RGB весов

# Метод 2: Learnable projection (альтернатива)
from src.backbones import MultiSpectralAdapter

adapter = MultiSpectralAdapter(
    in_channels=12,
    method='learnable_projection'  # Обучаемая проекция 12→3
)
```

---

## 4. 🎨 Аугментация данных - обязательно!

### Базовая аугментация (RGB)

```python
from src.augmentations import SatelliteAugmentation

# Training augmentation
train_aug = SatelliteAugmentation(
    mode='train',
    image_size=256,
    normalize=True
)

# Validation (только crop)
val_aug = SatelliteAugmentation(
    mode='val',
    image_size=256
)

# Применить
augmented = train_aug(image, mask)
aug_image = augmented['image']
aug_mask = augmented['mask']
```

**Включенные augmentations:**
- ✅ RandomRotate90
- ✅ Flip (horizontal/vertical)
- ✅ ShiftScaleRotate
- ✅ RandomCrop
- ✅ RandomBrightnessContrast
- ✅ RandomGamma
- ✅ GaussNoise
- ✅ GaussianBlur
- ✅ CoarseDropout

### Sentinel-2 augmentation (специализированная)

```python
from src.augmentations import Sentinel2Augmentation

# Учитывает особенности мультиспектральных данных
s2_aug = Sentinel2Augmentation(
    mode='train',
    image_size=256,
    use_all_bands=True
)

# Применяет:
# - Geometric transforms ко ВСЕМ каналам (безопасно)
# - Color augmentation только к RGB каналам (безопасно для индексов)
# - Не портит физический смысл спектральных данных

augmented = s2_aug(sentinel_image, mask)
```

### MixUp & CutMix (advanced)

```python
from src.augmentations import MixUpAugmentation, CutMixAugmentation

# MixUp: linear interpolation между примерами
mixup = MixUpAugmentation(alpha=0.2)
mixed_img, mixed_mask = mixup(image1, mask1, image2, mask2)

# CutMix: вырезать и вставить регион
cutmix = CutMixAugmentation(alpha=1.0)
cut_img, cut_mask = cutmix(image1, mask1, image2, mask2)
```

### Test-Time Augmentation (TTA)

```python
from src.augmentations import TestTimeAugmentation

# TTA для улучшения inference
tta = TestTimeAugmentation(n_augments=8)

# Создать аугментированные версии
augmented_batch = tta.augment_batch(test_image)

# Получить предсказания
predictions = [model(aug_img) for aug_img in augmented_batch]

# Усреднить
final_prediction = tta.deaugment_predictions(predictions, method='mean')
```

### Интеграция в training loop

```python
from src.data_loader import GeoDataLoader
from src.augmentations import get_augmentation_pipeline

# Загрузить данные
loader = GeoDataLoader(
    image_path="sentinel2.tif",
    shapefile_path="fields.shp",
    patch_size=512,  # Больше для crop augmentation
    stride=256
)

patches = loader.extract_patches()

# Augmentation pipeline
train_aug = get_augmentation_pipeline(
    data_type='sentinel2',  # 'rgb', 'multispectral', 'sentinel2'
    mode='train',
    image_size=256
)

# Training loop
for epoch in range(num_epochs):
    for patch in patches:
        # Применить аугментацию
        augmented = train_aug(patch['image'], patch['mask'])

        # Обучение
        loss = train_step(
            augmented['image'],
            augmented['mask']
        )
```

---

## 5. 📊 Полный пример: Sentinel-2 → RL Agent

```python
# 1. Загрузка Sentinel-2 данных
from src.data_loader import GeoDataLoader

loader = GeoDataLoader(
    image_path="data/raw/sentinel2_L2A_20230615.tif",  # 12 каналов
    shapefile_path="data/raw/crop_fields.shp",
    patch_size=512,
    stride=256,
    normalize=True
)

image, mask = loader.load_all()
patches = loader.extract_patches()

print(f"Loaded Sentinel-2 image: {image.shape}")  # (12, H, W)
print(f"Extracted {len(patches)} patches")

# 2. Разделить на train/val/test
from src.data_loader import DatasetSplitter

train_patches, val_patches, test_patches = DatasetSplitter.spatial_split(
    patches,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# 3. Создать augmentation
from src.augmentations import Sentinel2Augmentation

train_aug = Sentinel2Augmentation(mode='train', image_size=256)
val_aug = Sentinel2Augmentation(mode='val', image_size=256)

# 4. Создать backbone для Sentinel-2
from src.backbones import Sentinel2Backbone

backbone = Sentinel2Backbone(
    base_backbone='resnet50',
    pretrained=True,
    use_all_bands=True  # Все 12 каналов
)

# 5. Создать Advanced RL Agent
from src.advanced_agent import AdvancedRLAgent

agent = AdvancedRLAgent(
    in_channels=12,  # Sentinel-2
    device='cuda',
    use_contrastive=True,
    use_meta_learning=True,
    use_active_learning=True
)

# 6. Training loop с аугментацией
for epoch in range(num_epochs):
    for patch in train_patches:
        # Аугментация
        augmented = train_aug(patch['image'], patch['mask'])

        # Конвертировать в tensor
        image_tensor = torch.FloatTensor(augmented['image']).unsqueeze(0)
        mask_tensor = torch.FloatTensor(augmented['mask']).unsqueeze(0)

        # Training step
        losses = agent.train_step(image_tensor, mask_tensor, epoch=epoch)

    # Validation (без аугментации)
    for patch in val_patches:
        val_data = val_aug(patch['image'], patch['mask'])
        # ... evaluation

print("Training complete!")
```

---

## 6. 🔧 Troubleshooting

### Проблема: "CNN не поддерживает 12 каналов"

**Решение:**

```python
# Используйте адаптированный backbone
from src.backbones import Sentinel2Backbone

# ✅ Правильно - автоматическая адаптация
backbone = Sentinel2Backbone(use_all_bands=True)

# ❌ Неправильно - стандартный ResNet ожидает 3 канала
backbone = torchvision.models.resnet50(pretrained=True)
```

### Проблема: "Аугментация портит спектральные индексы"

**Решение:**

```python
# Используйте Sentinel2Augmentation
# Она применяет color augmentation только к RGB каналам!

from src.augmentations import Sentinel2Augmentation

aug = Sentinel2Augmentation(mode='train')  # ✅ Безопасно

# ❌ Не используйте стандартную aug для мультиспектральных данных
# aug = A.RandomBrightnessContrast()  # Портит NIR, SWIR каналы!
```

### Проблема: "Разные разрешения bands в Sentinel-2"

**Решение:**

```python
# Предварительно ресемплируйте все bands к одному разрешению

import rasterio
from rasterio.enums import Resampling

# Resample 20m bands → 10m
with rasterio.open("B05_20m.jp2") as src:
    data = src.read(
        out_shape=(src.height * 2, src.width * 2),
        resampling=Resampling.bilinear
    )

# Или используйте готовые продукты (Sen2Cor создает 10m все bands)
```

### Проблема: "Out of memory при аугментации"

**Решение:**

```python
# 1. Уменьшите patch_size
loader = GeoDataLoader(..., patch_size=128)  # Вместо 512

# 2. Применяйте аугментацию on-the-fly, не храните в памяти
for patch in patches:
    augmented = aug(patch['image'], patch['mask'])
    train_step(augmented)
    # augmented удаляется после использования

# 3. Используйте батчи меньшего размера
```

---

## 7. 📚 Резюме

### Ответы на вопросы:

**1. Какая CNN в основе?**
- ✅ Любая! ResNet, EfficientNet, и др.
- ✅ Автоматическая адаптация для N каналов
- ✅ Специализированный Sentinel2Backbone

**2. В каком виде загружать исходники?**
- ✅ GeoTIFF для изображений (любые каналы, любые форматы)
- ✅ Shapefile для векторных границ
- ✅ Автоматическая репроекция и конвертация

**3. Поддержка мультиспектральных данных?**
- ✅ Sentinel-2 (12 каналов) - полная поддержка
- ✅ Landsat (6-11 каналов) - поддержка
- ✅ Любое количество каналов (1-20+)
- ✅ Адаптация pretrained весов

**4. Аугментация данных?**
- ✅ Да, обязательна!
- ✅ Стандартная для RGB
- ✅ Специализированная для Sentinel-2
- ✅ MixUp, CutMix, TTA
- ✅ Безопасна для мультиспектральных данных

---

**Для детальной информации см.:**
- `DATA_FORMATS.md` - Подробно о форматах данных
- `src/backbones.py` - Код CNN backbones
- `src/augmentations.py` - Код аугментаций
- `notebooks/example_usage.py` - Примеры использования

---

**Готово к использованию! 🚀**
