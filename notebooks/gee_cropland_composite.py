# ============================================================================
# Стриминг композита из GEE через xee для сегментации пашни (U-Net)
# Проекция EPSG:3035 (LAEA Europe) — оптимально для ЕЧР
#
# Каналы (8 шт):
#   1. NDVI_stdDev   — сезонная вариабельность (пашня > луг)
#   2. NDVI_median   — общий уровень вегетации
#   3. NDVI_p90      — пиковая продуктивность
#   4. NDVI_p10      — минимум (голая почва весной = пашня)
#   5. NDVI_amp      — амплитуда (p90 - p10), ключевой признак пашни
#   6. SWIR1_median  — B11, структура почвы/влажность
#   7. SWIR2_median  — B12, разделение влажных/сухих поверхностей
#   8. GREEN_median  — B3, дополнительный спектральный контекст
#
# Автор: Artur (Kazan Federal University)
# Проект GEE: ee-landeco
# ============================================================================

import ee
import xarray as xr
import rioxarray
import xee
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import json

# ============================================================================
# Конфигурация
# ============================================================================

GEE_PROJECT = 'ee-landeco'
START_DATE = '2021-05-01'
END_DATE = '2025-10-31'
SCALE = 20                      # м/пиксель
CRS = 'EPSG:3035'               # LAEA Europe
NODATA = 0                      # nodata значение (uint8)
CLOUD_THRESH_MEDIAN = 0.65      # мягкий порог для медианы
CLOUD_THRESH_PEAK = 0.85        # жёсткий порог для p90
TILE_SIZE_M = 50000             # 50 км — оптимально для xee на Windows
OVERLAP_PX = 256                # пикселей перекрытия для U-Net
OVERLAP_M = OVERLAP_PX * SCALE  # 5120 м
MAX_RETRIES = 3                 # повторных попыток при сбое
RETRY_DELAY = 10                # секунд между попытками
S2_SCALE_FACTOR = 10000         # SR_HARMONIZED: reflectance × 10000

# Windows-совместимый путь
OUTPUT_DIR = Path(r'D:\ecr_cropland')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Каналы композита — порядок и параметры нормализации
# name: имя канала
# source: ключ в словаре при сборке
# norm_min, norm_max: входной диапазон для линейной нормализации в 0-255
BAND_CONFIG = [
    {'name': 'stdDev',  'norm_min': 0.0,  'norm_max': 0.35},
    {'name': 'median',  'norm_min': -0.1, 'norm_max': 0.9},
    {'name': 'p90',     'norm_min': 0.0,  'norm_max': 1.0},
    {'name': 'p10',     'norm_min': -0.2, 'norm_max': 0.6},
    {'name': 'amp',     'norm_min': 0.0,  'norm_max': 0.8},
    {'name': 'SWIR1',   'norm_min': 0.0,  'norm_max': 0.45},
    {'name': 'SWIR2',   'norm_min': 0.0,  'norm_max': 0.35},
    {'name': 'GREEN',   'norm_min': 0.0,  'norm_max': 0.15},
]

BAND_NAMES = [b['name'] for b in BAND_CONFIG]

# ============================================================================
# Геометрия ЕЧР
# ============================================================================

GEOMETRY_ECR = ee.Geometry.Polygon([
    [[44.84085558514128, 43.54760885494628],
     [44.797578913686756, 44.751121042909666],
     [44.90146645165875, 46.6255285864593],
     [45.92433808163401, 48.52623498170353],
     [47.13141607897388, 48.953157775771224],
     [49.0265288290429, 49.73314143201927],
     [51.81871963983705, 50.58483555464868],
     [53.89449496075653, 50.66989719586453],
     [55.30179494342321, 50.751001011597076],
     [57.55164797690707, 50.795990929429784],
     [57.56644584645435, 51.12122392097962],
     [57.463353470948626, 52.23276262145823],
     [57.865291605639584, 53.52684582393732],
     [59.56149510686957, 55.494200948409784],
     [58.717330303839674, 56.57511589328021],
     [57.122154085038815, 57.19878623860552],
     [50.933140811511905, 58.72007378447314],
     [46.583833667165194, 57.49258959204147],
     [43.867332688965995, 56.26883849257055],
     [40.30059873077071, 56.39429639806621],
     [37.851859721976695, 56.12270476879584],
     [36.826871487533836, 55.33599704082388],
     [35.63663575995269, 54.851443336556805],
     [35.49743450504397, 54.06516947088925],
     [35.06276028054044, 53.963811950471296],
     [34.54018695669403, 53.65699688392337],
     [34.79424836970688, 52.243342815805256],
     [35.789825391433936, 51.27333839161009],
     [36.60640758165285, 49.915469015527925],
     [38.038424821953015, 48.78018999762029],
     [38.4472715246393, 46.970147796256576],
     [36.82912858936574, 44.308461048636964],
     [41.22845532858114, 43.42436442343679]]
])

# ============================================================================
# Инициализация GEE
# ============================================================================

print(f"Инициализация GEE (проект: {GEE_PROJECT})...")
ee.Initialize(project=GEE_PROJECT)
print("✓ GEE инициализирован\n")

# ============================================================================
# Функции для создания композита
# ============================================================================

def _normalize_band(image, band_name, norm_min, norm_max, output_name):
    """Линейная нормализация канала в uint8 [1..255], 0 = nodata."""
    band = image.select(band_name)
    # Линейное растяжение norm_min..norm_max -> 1..255 (0 зарезервирован под nodata)
    normalized = band.subtract(norm_min)\
        .divide(norm_max - norm_min)\
        .multiply(254).add(1)\
        .clamp(1, 255)\
        .toByte()\
        .rename(output_name)
    return normalized


def create_composite(geometry, start_date, end_date):
    """
    Создает 8-канальный композит uint8 для сегментации пашни.
    
    Два прохода фильтрации облаков:
    - Мягкий (cs_cdf >= 0.65) для медианных статистик
    - Жёсткий (cs_cdf >= 0.85) для пиковых значений (p90)
    """
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    
    # Базовая фильтрация
    collection = s2.filterBounds(geometry)\
        .filterDate(start_date, end_date)\
        .filter(ee.Filter.calendarRange(4, 10, 'month'))  # апрель-октябрь для юга ЕЧР
    
    # Join с Cloud Score+
    joined = ee.Join.inner().apply(
        collection, cs,
        ee.Filter.equals(leftField='system:index', rightField='system:index')
    )
    
    def _merge_bands(feat):
        img = ee.Image(feat.get('primary'))
        cs_img = ee.Image(feat.get('secondary'))
        return img.addBands(cs_img.select('cs_cdf'))
    
    collection = ee.ImageCollection(joined.map(_merge_bands))
    
    # --- Вычисление NDVI ---
    def _add_ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return img.addBands(ndvi)
    
    collection = collection.map(_add_ndvi)
    
    # --- Мягкая фильтрация (для медианы, stdDev, p10) ---
    def _mask_soft(img):
        mask = img.select('cs_cdf').gte(CLOUD_THRESH_MEDIAN)\
            .And(img.select('B8A').mask())
        return img.updateMask(mask)
    
    coll_soft = collection.map(_mask_soft)
    
    # --- Жёсткая фильтрация (для p90) ---
    def _mask_strict(img):
        mask = img.select('cs_cdf').gte(CLOUD_THRESH_PEAK)\
            .And(img.select('B8A').mask())
        return img.updateMask(mask)
    
    coll_strict = collection.map(_mask_strict)
    
    # --- NDVI статистики ---
    ndvi_soft = coll_soft.select('NDVI').reduce(
        ee.Reducer.median()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.percentile([10]), sharedInputs=True)
    )
    
    ndvi_p90 = coll_strict.select('NDVI').reduce(
        ee.Reducer.percentile([90])
    )
    
    # Амплитуда NDVI
    ndvi_amp = ndvi_p90.select('NDVI_p90')\
        .subtract(ndvi_soft.select('NDVI_p10'))\
        .rename('NDVI_amp')
    
    # --- Спектральные медианы (с конверсией в reflectance 0-1) ---
    def _to_reflectance(img):
        """Конвертация B3, B11, B12 из SR (×10000) в reflectance (0-1)."""
        bands_sr = img.select(['B3', 'B11', 'B12'])
        bands_refl = bands_sr.divide(S2_SCALE_FACTOR)
        return img.addBands(bands_refl, overwrite=True)
    
    coll_refl = coll_soft.map(_to_reflectance)
    
    swir1_med = coll_refl.select('B11').median().rename('SWIR1_med')
    swir2_med = coll_refl.select('B12').median().rename('SWIR2_med')
    green_med = coll_refl.select('B3').median().rename('GREEN_med')
    
    # --- Объединяем все каналы в один Image ---
    all_bands = ndvi_soft.select('NDVI_stdDev')\
        .addBands(ndvi_soft.select('NDVI_median'))\
        .addBands(ndvi_p90.select('NDVI_p90'))\
        .addBands(ndvi_soft.select('NDVI_p10'))\
        .addBands(ndvi_amp)\
        .addBands(swir1_med)\
        .addBands(swir2_med)\
        .addBands(green_med)
    
    # --- Нормализация каждого канала в uint8 ---
    source_names = [
        'NDVI_stdDev', 'NDVI_median', 'NDVI_p90', 'NDVI_p10',
        'NDVI_amp', 'SWIR1_med', 'SWIR2_med', 'GREEN_med'
    ]
    
    channels = []
    for src_name, cfg in zip(source_names, BAND_CONFIG):
        ch = _normalize_band(all_bands, src_name, cfg['norm_min'], cfg['norm_max'], cfg['name'])
        channels.append(ch)
    
    composite = ee.Image.cat(channels)
    
    # Применяем единую маску (если хоть один канал пуст — nodata)
    valid_mask = composite.select(0).mask()
    for i in range(1, len(channels)):
        valid_mask = valid_mask.And(composite.select(i).mask())
    
    composite = composite.updateMask(valid_mask)\
        .unmask(NODATA)\
        .toByte()
    
    return composite


# ============================================================================
# Вспомогательные функции
# ============================================================================

def save_tile(ds_tile, output_path, band_names):
    """Сохраняет xarray Dataset как GeoTIFF uint8 с метаданными."""
    da = ds_tile[band_names].to_array(dim='band')
    da = da.assign_coords(band=band_names)
    
    da.rio.to_raster(
        str(output_path),
        compress='LZW',
        tiled=True,
        blockxsize=256,
        blockysize=256,
        dtype='uint8',
        nodata=NODATA,
    )


def load_with_retry(composite, crs, scale, geometry, retries=MAX_RETRIES):
    """Открытие и загрузка данных через xee с retry-логикой."""
    for attempt in range(1, retries + 1):
        try:
            ds = xr.open_dataset(
                composite,
                engine='ee',
                crs=crs,
                scale=scale,
                geometry=geometry,
                ee_mask_value=NODATA,
            )
            ds = ds.load()
            return ds
        except Exception as e:
            if attempt < retries:
                print(f"  ⚠ Попытка {attempt}/{retries} не удалась: {e}")
                print(f"    Повтор через {RETRY_DELAY} сек...")
                time.sleep(RETRY_DELAY)
            else:
                raise
    return None


def validate_tile(ds, band_names):
    """Проверяет, что тайл содержит валидные данные."""
    if ds.x.size == 0 or ds.y.size == 0:
        return False
    
    # Хотя бы 1% пикселей должны быть != nodata
    for name in band_names:
        data = ds[name].values
        valid_ratio = np.count_nonzero(data != NODATA) / data.size
        if valid_ratio < 0.01:
            return False
    return True


# ============================================================================
# ШАГ 1: Тестирование на маленькой области
# ============================================================================

print("=" * 70)
print("ШАГ 1: Тестирование на области вокруг Казани")
print("=" * 70)

test_geometry = ee.Geometry.Rectangle([48.5, 55.5, 49.5, 56.5])

print("\nСоздаём тестовый 8-канальный композит...")
test_composite = create_composite(test_geometry, START_DATE, END_DATE)

# Проверка метаданных в GEE
print("\nМетаданные композита в GEE:")
band_info = test_composite.bandTypes().getInfo()
band_names_gee = test_composite.bandNames().getInfo()
print(f"  Каналы ({len(band_names_gee)}): {band_names_gee}")
for name, info in band_info.items():
    print(f"  {name}: {info}")

# ============================================================================
# ШАГ 2: Загрузка через xee
# ============================================================================

print("\n" + "=" * 70)
print("ШАГ 2: Загрузка через xee в EPSG:3035")
print("=" * 70)

ds = xr.open_dataset(
    test_composite,
    engine='ee',
    crs=CRS,
    scale=SCALE,
    geometry=test_geometry,
    ee_mask_value=NODATA,
)

print(f"\nDataset:")
print(f"  Переменные: {list(ds.data_vars)}")
print(f"  Проекция:   {ds.rio.crs}")
print(f"  Разрешение: {ds.rio.resolution()}")
for var in ds.data_vars:
    print(f"  {var}: shape={ds[var].shape}, dtype={ds[var].dtype}")

# ============================================================================
# ШАГ 3: Валидация данных
# ============================================================================

print("\n" + "=" * 70)
print("ШАГ 3: Валидация данных (сэмпл 200×200 px)")
print("=" * 70)

sample = ds.isel(x=slice(0, 200), y=slice(0, 200)).load()

all_ok = True
for var in BAND_NAMES:
    if var not in sample.data_vars:
        print(f"\n  ✗ ОШИБКА: канал '{var}' отсутствует в Dataset!")
        all_ok = False
        continue
    
    data = sample[var].values
    valid = data[data != NODATA]
    
    print(f"\n  {var}:")
    print(f"    dtype={data.dtype}, shape={data.shape}")
    print(f"    all:   min={np.min(data)}, max={np.max(data)}")
    
    if valid.size > 0:
        print(f"    valid: min={np.min(valid)}, max={np.max(valid)}, "
              f"mean={np.mean(valid):.1f}, count={valid.size}")
    else:
        print(f"    ⚠ Нет валидных пикселей!")
    
    if data.dtype == np.uint8:
        print(f"    ✓ uint8")
    else:
        print(f"    ✗ Ожидался uint8, получен {data.dtype}")
        all_ok = False

if all_ok:
    print(f"\n✓ Все {len(BAND_NAMES)} каналов корректны")
else:
    print(f"\n✗ Обнаружены проблемы — проверьте конфигурацию")

# ============================================================================
# ШАГ 4: Сохранение тестового тайла
# ============================================================================

print("\n" + "=" * 70)
print("ШАГ 4: Сохранение тестового тайла")
print("=" * 70)

test_output = OUTPUT_DIR / 'test_tile_8ch_epsg3035.tif'

save_tile(sample, test_output, BAND_NAMES)
print(f"\n✓ Сохранён: {test_output}")

# Верификация
saved = rioxarray.open_rasterio(test_output)
print(f"\n  Верификация сохранённого файла:")
print(f"    Shape:      {saved.shape}")
print(f"    Dtype:      {saved.dtype}")
print(f"    CRS:        {saved.rio.crs}")
print(f"    Resolution: {saved.rio.resolution()}")
print(f"    Bounds:     {saved.rio.bounds()}")
print(f"    NoData:     {saved.rio.nodata}")

for i, bname in enumerate(BAND_NAMES):
    bd = saved.isel(band=i).values
    valid = bd[bd != NODATA]
    if valid.size > 0:
        print(f"    Band {i+1} ({bname}): min={np.min(valid)}, max={np.max(valid)}, "
              f"mean={np.mean(valid):.1f}")
    else:
        print(f"    Band {i+1} ({bname}): нет данных")

saved.close()

# Сохраняем метаданные нормализации (нужны для инференса U-Net!)
meta_path = OUTPUT_DIR / 'band_config.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump({
        'bands': BAND_CONFIG,
        'band_order': BAND_NAMES,
        'crs': CRS,
        'scale_m': SCALE,
        'nodata': NODATA,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'cloud_thresh_median': CLOUD_THRESH_MEDIAN,
        'cloud_thresh_peak': CLOUD_THRESH_PEAK,
        's2_scale_factor': S2_SCALE_FACTOR,
        'calendar_months': '4-10',
        'description': (
            '8-channel composite for cropland segmentation (U-Net). '
            'Values 1-255 (linear stretch), 0 = nodata.'
        ),
    }, f, indent=2, ensure_ascii=False)
print(f"✓ Метаданные нормализации: {meta_path}")

print(f"\n{'='*70}")
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")
print(f"{'='*70}")


# ============================================================================
# Функция обработки всей ЕЧР с перекрытиями
# ============================================================================

def process_full_region(
    geometry,
    output_dir,
    tile_size_m=TILE_SIZE_M,
    overlap_m=OVERLAP_M,
):
    """
    Обработка ЕЧР по тайлам 50×50 км с перекрытием 256 px
    для бесшовной сегментации U-Net.
    
    Тайлы сохраняются с координатами в имени файла (в км EPSG:3035).
    Перекрытия обрезаются при сборке мозаики.
    """
    
    print(f"\n{'='*70}")
    print("ОБРАБОТКА ПОЛНОЙ ТЕРРИТОРИИ ЕЧР")
    print(f"{'='*70}")
    print(f"  Размер тайла:   {tile_size_m/1000:.0f} × {tile_size_m/1000:.0f} км")
    print(f"  Перекрытие:     {overlap_m:.0f} м ({OVERLAP_PX} px)")
    print(f"  Разрешение:     {SCALE} м/px")
    print(f"  Каналы:         {len(BAND_NAMES)}: {BAND_NAMES}")
    print(f"  CRS:            {CRS}")
    print(f"  NoData:         {NODATA}\n")
    
    # Композит
    print("Создаём композит в GEE...")
    composite = create_composite(geometry, START_DATE, END_DATE)
    
    # Bbox в EPSG:3035
    geometry_3035 = geometry.transform(CRS, ee.ErrorMargin(1))
    bounds = geometry_3035.bounds(ee.ErrorMargin(1)).getInfo()['coordinates'][0]
    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    print(f"  Границы EPSG:3035:")
    print(f"    X: {x_min:,.0f} — {x_max:,.0f} м")
    print(f"    Y: {y_min:,.0f} — {y_max:,.0f} м")
    
    # Сетка тайлов (шаг = tile_size - overlap для покрытия перекрытий)
    step = tile_size_m  # шаг без вычета overlap — overlap добавляется сверху
    x_origins = np.arange(x_min, x_max, step)
    y_origins = np.arange(y_min, y_max, step)
    total_tiles = len(x_origins) * len(y_origins)
    
    print(f"  Тайлов в сетке: {total_tiles} ({len(x_origins)} × {len(y_origins)})")
    
    tile_dir = output_dir / 'tiles'
    tile_dir.mkdir(exist_ok=True, parents=True)
    
    saved_count = 0
    skipped_count = 0
    error_count = 0
    
    log_path = output_dir / 'processing_log.json'
    log_entries = []
    
    for i, x0 in enumerate(x_origins):
        for j, y0 in enumerate(y_origins):
            tile_id = f"x{int(x0/1000):05d}_y{int(y0/1000):05d}"
            tile_path = tile_dir / f"tile_{tile_id}.tif"
            
            # Пропускаем уже обработанные
            if tile_path.exists():
                saved_count += 1
                continue
            
            # Геометрия тайла с перекрытием
            x1 = min(x0 + tile_size_m + overlap_m, x_max + overlap_m)
            y1 = min(y0 + tile_size_m + overlap_m, y_max + overlap_m)
            
            tile_rect = ee.Geometry.Rectangle(
                [x0, y0, x1, y1],
                proj=CRS,
                geodesic=False,
            )
            
            # Пересечение с исходной геометрией
            tile_geom = tile_rect.intersection(geometry_3035, ee.ErrorMargin(1))
            
            try:
                ds_tile = load_with_retry(composite, CRS, SCALE, tile_geom)
                
                if not validate_tile(ds_tile, BAND_NAMES):
                    skipped_count += 1
                    continue
                
                save_tile(ds_tile, tile_path, BAND_NAMES)
                saved_count += 1
                
                log_entries.append({
                    'tile_id': tile_id,
                    'status': 'ok',
                    'x0': float(x0), 'y0': float(y0),
                    'x1': float(x1), 'y1': float(y1),
                    'shape': [int(ds_tile.dims.get('x', 0)), int(ds_tile.dims.get('y', 0))],
                })
                
                if saved_count % 10 == 0:
                    print(f"  [{saved_count:4d} сохранено | {skipped_count} пусто | "
                          f"{error_count} ошибок] → {tile_path.name}")
                
            except Exception as e:
                error_count += 1
                log_entries.append({
                    'tile_id': tile_id,
                    'status': 'error',
                    'error': str(e),
                })
                print(f"  ✗ Ошибка {tile_id}: {e}")
                continue
    
    # Сохраняем лог
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_grid': total_tiles,
            'saved': saved_count,
            'skipped_empty': skipped_count,
            'errors': error_count,
            'tile_size_m': tile_size_m,
            'overlap_m': overlap_m,
            'tiles': log_entries,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Обработка завершена!")
    print(f"  Сохранено: {saved_count} тайлов")
    print(f"  Пропущено (пусто): {skipped_count}")
    print(f"  Ошибки: {error_count}")
    print(f"  Каталог: {tile_dir}")
    print(f"  Лог: {log_path}")
    print(f"{'='*70}")


# ============================================================================
# ЗАПУСК (раскомментируйте после успешного тестирования)
# ============================================================================

# process_full_region(GEOMETRY_ECR, OUTPUT_DIR)
