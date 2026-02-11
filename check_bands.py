import rasterio
import numpy as np

img = r"d:\Cache\Yandex.Disk\РАЗРАБОТКА\code\rl_segmentation\data\raw\satellite_image.tif"
with rasterio.open(img) as src:
    print(f"Bands: {src.count}")
    print(f"Dtype: {src.dtypes}")
    print(f"Descriptions: {src.descriptions}")
    print(f"ColorInterp: {[str(c) for c in src.colorinterp]}")
    print(f"Size: {src.width} x {src.height}")
    for b in range(1, src.count + 1):
        data = src.read(b)
        print(f"  Band {b}: min={data.min()}, max={data.max()}, mean={data.mean():.2f}, std={data.std():.2f}, zeros={np.sum(data==0)}")
