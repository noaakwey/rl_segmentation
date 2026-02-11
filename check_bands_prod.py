import rasterio
import numpy as np

img = r"D:\Cache\Yandex.Disk-amgafurov@kpfu.ru\Загрузки\midVolga_mosaic.tif"
with rasterio.open(img) as src:
    print(f"Bands: {src.count}")
    print(f"Dtype: {src.dtypes}")
    print(f"Descriptions: {src.descriptions}")
    print(f"ColorInterp: {[str(c) for c in src.colorinterp]}")
    print(f"Size: {src.width} x {src.height}")
    # Sample small region for statistics (full image too large)
    from rasterio.windows import Window
    w = Window(src.width // 2, src.height // 2, 2048, 2048)
    for b in range(1, src.count + 1):
        data = src.read(b, window=w)
        print(f"  Band {b}: min={data.min()}, max={data.max()}, mean={data.mean():.2f}, std={data.std():.2f}")
