# 📁 Форматы данных и загрузка

Подробное руководство по форматам входных данных, поддерживаемым источникам и способам загрузки.

## 🛰️ Поддерживаемые форматы изображений

### 1. GeoTIFF (.tif, .tiff)

**Основной формат** для спутниковых снимков.

```python
from src.data_loader import GeoDataLoader

loader = GeoDataLoader(
    image_path="data/raw/satellite_image.tif",
    shapefile_path="data/raw/crop_boundaries.shp",
    patch_size=256,
    stride=128,
    normalize=True
)

image, mask = loader.load_all()
```

**Поддерживаемые характеристики:**
- ✅ Любое количество каналов (1-20+)
- ✅ Различные битности: uint8, uint16, float32
- ✅ Различные проекции (автоматическая репроекция)
- ✅ Cloud Optimized GeoTIFF (COG)
- ✅ Сжатые файлы (LZW, DEFLATE, JPEG)

**Структура GeoTIFF:**
```
satellite_image.tif
├── Spatial Reference (CRS)
├── Geotransform (координаты)
├── Bands (каналы):
│   ├── Band 1
│   ├── Band 2
│   ├── ...
│   └── Band N
└── Metadata
```

### 2. Shapefile (.shp)

**Формат для векторных границ полей.**

```python
# Автоматическая загрузка и конвертация в растр
loader = GeoDataLoader(
    image_path="image.tif",
    shapefile_path="boundaries.shp"  # Автоматически конвертируется
)
```

**Требования:**
- ✅ Должен содержать полигоны (Polygon или MultiPolygon)
- ✅ Должен иметь CRS (автоматическая репроекция если отличается)
- ✅ Может содержать атрибуты (игнорируются при базовой сегментации)

**Файлы shapefile** (должны быть все вместе):
```
boundaries/
├── boundaries.shp      # Основной файл
├── boundaries.shx      # Индекс
├── boundaries.dbf      # Атрибуты
├── boundaries.prj      # Проекция
└── boundaries.cpg      # Кодировка (опционально)
```

### 3. GeoJSON (.geojson, .json)

```python
import geopandas as gpd

# Загрузить GeoJSON
gdf = gpd.read_file("boundaries.geojson")

# Сохранить как shapefile
gdf.to_file("boundaries.shp")

# Использовать в loader
loader = GeoDataLoader(..., shapefile_path="boundaries.shp")
```

## 🛰️ Специфика различных спутников

### Sentinel-2 (ESA)

**12 спектральных каналов:**

| Band | Wavelength (nm) | Resolution | Description |
|------|----------------|------------|-------------|
| B1   | 443           | 60m        | Coastal aerosol |
| B2   | 490           | 10m        | Blue |
| B3   | 560           | 10m        | Green |
| B4   | 665           | 10m        | Red |
| B5   | 705           | 20m        | Red Edge 1 |
| B6   | 740           | 20m        | Red Edge 2 |
| B7   | 783           | 20m        | Red Edge 3 |
| B8   | 842           | 10m        | NIR |
| B8A  | 865           | 20m        | NIR narrow |
| B9   | 940           | 60m        | Water vapor |
| B11  | 1610          | 20m        | SWIR 1 |
| B12  | 2190          | 20m        | SWIR 2 |

**Загрузка Sentinel-2:**

```python
from src.backbones import Sentinel2Backbone
from src.advanced_agent import AdvancedRLAgent

# Вариант 1: Использовать все 12 каналов
backbone = Sentinel2Backbone(use_all_bands=True)

# Вариант 2: Только RGB + NIR (4 канала)
backbone = Sentinel2Backbone(use_all_bands=False)

# Интеграция в агент
agent = AdvancedRLAgent(
    in_channels=12,  # или 4 для RGB+NIR
    device='cuda'
)
```

**Загрузка через rasterio:**

```python
import rasterio
import numpy as np

# Sentinel-2 обычно идет как .SAFE директория
# Нужно загрузить отдельные bands и стакнуть

bands = []
for band_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]:  # 8A пропущен
    with rasterio.open(f"S2_MSIL2A/.../B{band_num:02d}.jp2") as src:
        bands.append(src.read(1))

# Stack bands
sentinel_image = np.stack(bands, axis=0)  # (12, H, W)
print(f"Sentinel-2 shape: {sentinel_image.shape}")

# Сохранить как единый GeoTIFF
with rasterio.open(
    "sentinel2_stacked.tif",
    'w',
    driver='GTiff',
    height=sentinel_image.shape[1],
    width=sentinel_image.shape[2],
    count=12,
    dtype=sentinel_image.dtype,
    crs=src.crs,
    transform=src.transform
) as dst:
    dst.write(sentinel_image)
```

**Augmentation для Sentinel-2:**

```python
from src.augmentations import Sentinel2Augmentation

# Создать Sentinel-2 specific augmentation
aug = Sentinel2Augmentation(
    mode='train',
    image_size=256,
    use_all_bands=True
)

# Применить
augmented = aug(sentinel_image, mask)
```

### Landsat 8/9 (NASA/USGS)

**11 спектральных каналов:**

| Band | Wavelength (nm) | Resolution | Description |
|------|----------------|------------|-------------|
| B1   | 433-453       | 30m        | Coastal/Aerosol |
| B2   | 450-515       | 30m        | Blue |
| B3   | 525-600       | 30m        | Green |
| B4   | 630-680       | 30m        | Red |
| B5   | 845-885       | 30m        | NIR |
| B6   | 1560-1660     | 30m        | SWIR 1 |
| B7   | 2100-2300     | 30m        | SWIR 2 |
| B8   | 500-680       | 15m        | Panchromatic |
| B9   | 1360-1390     | 30m        | Cirrus |
| B10  | 10600-11200   | 100m       | TIRS 1 |
| B11  | 11500-12500   | 100m       | TIRS 2 |

```python
# Загрузка Landsat
from src.backbones import create_backbone

# Обычно используют B2-B7 (6 каналов)
backbone = create_backbone(
    backbone_name='resnet50',
    in_channels=6,  # B2, B3, B4, B5, B6, B7
    pretrained=True
)
```

### Planet Scope (Planet Labs)

**4 канала:**
- Blue (455-515 nm)
- Green (500-590 nm)
- Red (590-670 nm)
- NIR (780-860 nm)

```python
# Planet: 4 канала (RGB + NIR)
agent = AdvancedRLAgent(in_channels=4)
```

### Aerial/Drone imagery

**Обычно RGB или RGB+NIR:**

```python
# RGB (3 канала)
agent = AdvancedRLAgent(in_channels=3)

# RGB + NIR (4 канала)
agent = AdvancedRLAgent(in_channels=4)

# RGB + NIR + RedEdge (5 каналов)
agent = AdvancedRLAgent(in_channels=5)
```

## 📦 Пример структуры данных

### Минимальная структура

```
data/
└── raw/
    ├── satellite_image.tif    # Спутниковый снимок
    └── crop_boundaries.shp    # Границы полей (+ .shx, .dbf, .prj)
```

### Полная структура

```
data/
├── raw/
│   ├── region1/
│   │   ├── sentinel2_L2A_2023-06-15.tif
│   │   └── fields_2023.shp
│   ├── region2/
│   │   ├── landsat8_2023-07-01.tif
│   │   └── fields_2023.shp
│   └── aerial/
│       ├── drone_rgb_nir.tif
│       └── fields.shp
│
└── processed/
    ├── patches_256x256/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── augmented/
```

## 💻 Примеры загрузки

### 1. Базовая загрузка RGB

```python
from src.data_loader import GeoDataLoader

loader = GeoDataLoader(
    image_path="data/raw/rgb_orthophoto.tif",
    shapefile_path="data/raw/fields.shp",
    patch_size=256,
    stride=128,
    normalize=True
)

# Загрузить все
image, mask = loader.load_all()  # image: (3, H, W), mask: (H, W)

# Извлечь патчи
patches = loader.extract_patches()
print(f"Extracted {len(patches)} patches")

# Получить статистику
stats = loader.get_statistics()
print(f"Crop percentage: {stats['crop_percentage']:.2f}%")
```

### 2. Загрузка Sentinel-2

```python
# Предполагаем, что вы уже стакнули bands в один GeoTIFF

loader = GeoDataLoader(
    image_path="data/raw/sentinel2_all_bands.tif",  # 12 каналов
    shapefile_path="data/raw/fields.shp",
    patch_size=256,
    stride=128,
    normalize=True
)

image, mask = loader.load_all()
print(f"Sentinel-2 image shape: {image.shape}")  # (12, H, W)

# Использовать специализированный backbone
from src.backbones import Sentinel2Backbone

backbone = Sentinel2Backbone(
    base_backbone='resnet50',
    use_all_bands=True  # Все 12 каналов
)
```

### 3. Загрузка с аугментацией

```python
from src.data_loader import GeoDataLoader
from src.augmentations import get_augmentation_pipeline

# Загрузить данные
loader = GeoDataLoader(
    image_path="data/raw/image.tif",
    shapefile_path="data/raw/fields.shp",
    patch_size=512,  # Больший размер для аугментации
    stride=256
)

image, mask = loader.load_all()
patches = loader.extract_patches()

# Создать augmentation pipeline
aug = get_augmentation_pipeline(
    data_type='sentinel2',  # или 'rgb', 'multispectral'
    mode='train',
    image_size=256  # Crop до нужного размера
)

# Применить к патчу
patch = patches[0]
augmented = aug(patch['image'], patch['mask'])

print(f"Original: {patch['image'].shape}")
print(f"Augmented: {augmented['image'].shape}")
```

### 4. Загрузка из нескольких источников

```python
from src.data_loader import GeoDataLoader
import glob

# Найти все GeoTIFF в директории
image_paths = glob.glob("data/raw/region1/*.tif")

all_patches = []

for image_path in image_paths:
    # Соответствующий shapefile
    shapefile_path = image_path.replace('.tif', '_fields.shp')

    loader = GeoDataLoader(
        image_path=image_path,
        shapefile_path=shapefile_path,
        patch_size=256
    )

    patches = loader.extract_patches()
    all_patches.extend(patches)

print(f"Loaded {len(all_patches)} patches from {len(image_paths)} images")
```

## 🔧 Конвертация форматов

### Google Earth Engine → GeoTIFF

```python
# Экспорт из GEE (JavaScript)
var image = ee.Image('COPERNICUS/S2_SR/20230615T...')
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']);

Export.image.toDrive({
    image: image,
    description: 'sentinel2_export',
    scale: 10,
    region: geometry,
    fileFormat: 'GeoTIFF'
});
```

### Shapefile → GeoJSON

```python
import geopandas as gpd

gdf = gpd.read_file("fields.shp")
gdf.to_file("fields.geojson", driver='GeoJSON')
```

### Multiple TIFFs → Stacked

```python
import rasterio
import numpy as np

# Загрузить отдельные bands
bands = []
for band_file in ['B2.tif', 'B3.tif', 'B4.tif', 'B8.tif']:
    with rasterio.open(band_file) as src:
        bands.append(src.read(1))
        meta = src.meta.copy()

# Stack
stacked = np.stack(bands, axis=0)

# Сохранить
meta.update(count=len(bands))
with rasterio.open('stacked.tif', 'w', **meta) as dst:
    dst.write(stacked)
```

## ⚠️ Важные моменты

### 1. CRS (Coordinate Reference System)

**Shapefile и GeoTIFF должны иметь одинаковую проекцию**

```python
import rasterio
import geopandas as gpd

# Проверить CRS
with rasterio.open("image.tif") as src:
    image_crs = src.crs

gdf = gpd.read_file("fields.shp")
shapefile_crs = gdf.crs

print(f"Image CRS: {image_crs}")
print(f"Shapefile CRS: {shapefile_crs}")

# Если разные - репроект shapefile
if image_crs != shapefile_crs:
    gdf = gdf.to_crs(image_crs)
    gdf.to_file("fields_reprojected.shp")
```

**GeoDataLoader делает это автоматически!**

### 2. Нормализация

**Различные спутники имеют разные диапазоны значений:**

```python
# uint8 (0-255) - обычные RGB изображения
# uint16 (0-65535) - Sentinel-2, Landsat
# float32 - уже нормализованные

# GeoDataLoader автоматически нормализует если normalize=True:
loader = GeoDataLoader(..., normalize=True)
# uint8 → [0, 1]
# uint16 → [0, 1]
# float → percentile normalization
```

### 3. Разрешение

**Патчи должны покрывать достаточную площадь:**

```python
# Sentinel-2: 10m/pixel
# patch_size=256 → 2.56 km × 2.56 km

# Landsat: 30m/pixel
# patch_size=256 → 7.68 km × 7.68 km

# Aerial: 0.5m/pixel
# patch_size=256 → 128 m × 128 m

# Выбирайте patch_size в зависимости от разрешения!
```

### 4. Память

**Большие изображения могут не поместиться в память:**

```python
# Плохо: загружать все сразу
image, mask = loader.load_all()  # Может быть 20GB+

# Хорошо: использовать патчи
patches = loader.extract_patches()  # Загружает по одному
for patch in patches:
    process(patch)  # Обрабатывает постепенно
```

## 📚 Дополнительная информация

- Для Sentinel-2: https://sentinels.copernicus.eu/web/sentinel/user-guides
- Для Landsat: https://www.usgs.gov/landsat-missions
- Rasterio docs: https://rasterio.readthedocs.io/
- GeoPandas docs: https://geopandas.org/

## 🆘 Частые проблемы

### "CRS mismatch"

```python
# Решение: репроект
gdf = gdf.to_crs(image_crs)
```

### "Out of memory"

```python
# Решение: использовать меньшие патчи или stride
loader = GeoDataLoader(..., patch_size=128, stride=64)
```

### "No bands found"

```python
# Проверить количество каналов
with rasterio.open("image.tif") as src:
    print(f"Bands: {src.count}")
```

### "Shapefile empty"

```python
# Проверить bbox overlap
with rasterio.open("image.tif") as src:
    image_bounds = src.bounds

gdf = gpd.read_file("fields.shp")
shapefile_bounds = gdf.total_bounds

print(f"Image bounds: {image_bounds}")
print(f"Shapefile bounds: {shapefile_bounds}")
# Должны пересекаться!
```

---

**Для быстрого старта см. `README.md` и `notebooks/example_usage.py`**
