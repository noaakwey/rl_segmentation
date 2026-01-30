"""
Production inference for large rasters using supervised U-Net.

Key ideas for very large GeoTIFFs:
- Stream tiles from disk (no full-image loads)
- Add context padding, but write only the center (no huge accumulators)
- Optional TTA (flips) for more stable predictions
- Save both raster mask and vector polygons
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from tqdm import tqdm
import yaml
import geopandas as gpd
from shapely.geometry import shape as shapely_shape

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import UNet


def _load_config(path: str) -> Dict:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _normalize(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return image.astype(np.float32)


def _tta_variants(image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    return [
        (image, "none"),
        (np.flip(image, axis=2).copy(), "hflip"),
        (np.flip(image, axis=1).copy(), "vflip"),
        (np.flip(np.flip(image, axis=1), axis=2).copy(), "hvflip"),
    ]


def _invert_tta(mask: np.ndarray, tag: str) -> np.ndarray:
    if tag == "hflip":
        return np.flip(mask, axis=1)
    if tag == "vflip":
        return np.flip(mask, axis=0)
    if tag == "hvflip":
        return np.flip(np.flip(mask, axis=0), axis=1)
    return mask


def _predict_prob(model: UNet, image: np.ndarray, device: torch.device, use_tta: bool) -> np.ndarray:
    """
    Predict probability mask for a single tile (C, H, W) in [0, 1].
    """
    use_amp = device.type == "cuda"
    _, h, w = image.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16

    if not use_tta:
        x = torch.from_numpy(image).float().unsqueeze(0).to(device)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        prob = prob[:h, :w]
        return prob

    probs = []
    for aug, tag in _tta_variants(image):
        x = torch.from_numpy(aug).float().unsqueeze(0).to(device)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            aug_prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        aug_prob = aug_prob[:h, :w]
        probs.append(_invert_tta(aug_prob, tag))
    return np.mean(probs, axis=0)


def _predict_prob_batch(model: UNet, images: List[np.ndarray], device: torch.device) -> List[np.ndarray]:
    """Predict probability masks for a batch of tiles (no TTA)."""
    if not images:
        return []

    use_amp = device.type == "cuda"
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    c = images[0].shape[0]
    pad_h = (16 - (max_h % 16)) % 16
    pad_w = (16 - (max_w % 16)) % 16
    padded_h = max_h + pad_h
    padded_w = max_w + pad_w

    batch = np.zeros((len(images), c, padded_h, padded_w), dtype=np.float32)
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        batch[i, :, :h, :w] = img

    x = torch.from_numpy(batch).float().to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

    # Crop back to original tile sizes
    out: List[np.ndarray] = []
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        out.append(probs[i, :h, :w])
    return out


def _iter_windows(width: int, height: int, tile_size: int) -> Iterable[Tuple[int, int, Window]]:
    """
    Iterate non-overlapping write windows that cover the full raster.
    """
    for top in range(0, height, tile_size):
        h = min(tile_size, height - top)
        for left in range(0, width, tile_size):
            w = min(tile_size, width - left)
            yield left, top, Window(left, top, w, h)


def run_production_inference(
    image_path: str,
    checkpoint_path: str,
    output_mask_path: str,
    output_vector_path: str,
    config: Dict,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    prod_cfg = config.get("production", {})
    tile_size = int(prod_cfg.get("tile_size", 512))
    context = int(prod_cfg.get("context", 64))
    batch_size = int(prod_cfg.get("batch_size", 4))
    threshold = float(prod_cfg.get("threshold", config.get("supervised_eval", {}).get("threshold_max", 0.7)))
    use_tta = bool(prod_cfg.get("use_tta", config.get("supervised_eval", {}).get("use_tta", True)))
    min_area_pixels = int(prod_cfg.get("min_area_pixels", 1024))
    do_vectorize = bool(prod_cfg.get("vectorize", True))

    # Load model
    with rasterio.open(image_path) as src:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        required_in_channels = int(checkpoint["model_state_dict"]["enc1.0.conv.weight"].shape[1])

        # Resolve which bands to use. Prefer config, but auto-fix mismatches.
        bands = prod_cfg.get("bands")
        if bands is None:
            bands = list(range(1, required_in_channels + 1))
        bands = [int(b) for b in bands]

        if len(bands) != required_in_channels:
            print(
                f"Configured bands={bands} (len={len(bands)}) do not match checkpoint "
                f"in_channels={required_in_channels}. Auto-selecting first {required_in_channels} bands."
            )
            bands = list(range(1, required_in_channels + 1))

        if src.count < required_in_channels:
            raise ValueError(
                f"Input raster has {src.count} bands, but checkpoint expects {required_in_channels} channels."
            )

        in_channels = required_in_channels
        model = UNet(
            in_channels=in_channels, base_channels=int(config.get("supervised", {}).get("base_channels", 32))
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            compress="lzw",
            tiled=True,
            blockxsize=tile_size,
            blockysize=tile_size,
            BIGTIFF="YES",
            nodata=0,
        )

        output_mask_path = str(Path(output_mask_path))
        Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)

        width, height = src.width, src.height
        print(f"Raster size: {width} x {height}")
        print(
            f"Tile size: {tile_size}, context: {context}, threshold: {threshold:.3f}, "
            f"TTA: {use_tta}, batch_size: {batch_size}, bands: {bands}, device: {device}"
        )

        with rasterio.open(output_mask_path, "w", **profile) as dst:
            total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
            tiles_buffer: List[np.ndarray] = []
            meta_buffer: List[Tuple[Window, int, int]] = []

            def flush_buffer():
                if not tiles_buffer:
                    return
                probs = _predict_prob_batch(model, tiles_buffer, device)
                for prob_full, (write_window, write_h, write_w) in zip(probs, meta_buffer):
                    prob_center = prob_full[context : context + write_h, context : context + write_w]
                    mask_center = (prob_center > threshold).astype(np.uint8)
                    dst.write(mask_center, 1, window=write_window)
                tiles_buffer.clear()
                meta_buffer.clear()

            for left, top, write_window in tqdm(
                _iter_windows(width, height, tile_size), total=total_tiles, desc="Tiles"
            ):
                write_w = int(write_window.width)
                write_h = int(write_window.height)

                read_left = left - context
                read_top = top - context
                read_w = write_w + 2 * context
                read_h = write_h + 2 * context
                read_window = Window(read_left, read_top, read_w, read_h)

                tile = src.read(indexes=bands, window=read_window, boundless=True, fill_value=0)

                if src.count >= 4 and 4 in bands:
                    alpha_idx = bands.index(4)
                    if np.max(tile[alpha_idx]) == 0:
                        dst.write(np.zeros((write_h, write_w), dtype=np.uint8), 1, window=write_window)
                        continue

                tile = _normalize(tile)

                if use_tta:
                    prob_full = _predict_prob(model, tile, device, use_tta=True)
                    prob_center = prob_full[context : context + write_h, context : context + write_w]
                    mask_center = (prob_center > threshold).astype(np.uint8)
                    dst.write(mask_center, 1, window=write_window)
                    continue

                tiles_buffer.append(tile)
                meta_buffer.append((write_window, write_h, write_w))
                if len(tiles_buffer) >= batch_size:
                    flush_buffer()

            flush_buffer()

    print(f"Mask saved to {output_mask_path}")

    # Vectorization step
    if do_vectorize:
        vectorize_mask(
            mask_path=output_mask_path,
            output_vector_path=output_vector_path,
            min_area_pixels=min_area_pixels,
        )
    else:
        print("Vectorization disabled by config (production.vectorize=false).")


def vectorize_mask(mask_path: str, output_vector_path: str, min_area_pixels: int = 1024):
    """
    Vectorize a binary mask GeoTIFF into polygons.

    Note: on very large rasters this can produce many features. The min_area filter
    reduces noise and output size.
    """
    output_vector_path = str(Path(output_vector_path))
    Path(output_vector_path).parent.mkdir(parents=True, exist_ok=True)

    geoms: List[Tuple[object, int]] = []

    with rasterio.open(mask_path) as src:
        print("Vectorizing mask...")
        for _, window in tqdm(src.block_windows(1), desc="Blocks"):
            data = src.read(1, window=window)
            if data.max() == 0:
                continue
            transform = src.window_transform(window)
            for geom, value in shapes(data.astype(np.uint8), mask=data.astype(bool), transform=transform):
                if value != 1:
                    continue
                geoms.append((shapely_shape(geom), int(value)))

        if not geoms:
            print("No positive polygons found; saving empty vector.")
            gdf = gpd.GeoDataFrame({"value": []}, geometry=[], crs=src.crs)
            gdf.to_file(output_vector_path, driver="GPKG")
            print(f"Vector saved to {output_vector_path}")
            return

        gdf = gpd.GeoDataFrame({"value": [v for _, v in geoms]}, geometry=[g for g, _ in geoms], crs=src.crs)

    # Filter tiny polygons in pixel units by approximating via raster resolution
    # This keeps vector sizes manageable for production workflows.
    if min_area_pixels > 1 and len(gdf) > 0:
        # Convert pixel area to projected area using approximate pixel size
        # Assumes square-ish pixels.
        with rasterio.open(mask_path) as src:
            px_w = abs(src.transform.a)
            px_h = abs(src.transform.e)
        min_area = float(min_area_pixels) * px_w * px_h
        gdf = gdf[gdf.geometry.area >= min_area]

    # Dissolve to reduce fragmentation
    if len(gdf) > 0:
        gdf["dissolve_key"] = 1
        gdf = gdf.dissolve(by="dissolve_key").explode(index_parts=False).reset_index(drop=True)
        gdf = gdf.drop(columns=["dissolve_key"], errors="ignore")

    gdf.to_file(output_vector_path, driver="GPKG")
    print(f"Vector saved to {output_vector_path} (features={len(gdf)})")


def main():
    parser = argparse.ArgumentParser(description="Production U-Net inference on large rasters")
    parser.add_argument("--image", type=str, required=True, help="Path to large input GeoTIFF")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_unet.pt checkpoint")
    parser.add_argument(
        "--config", type=str, default="configs/custom_config.yaml", help="Path to configuration YAML"
    )
    parser.add_argument(
        "--output-mask",
        type=str,
        default="experiments/production/midvolga_mask.tif",
        help="Path to output mask GeoTIFF",
    )
    parser.add_argument(
        "--output-vector",
        type=str,
        default="experiments/production/midvolga_mask.gpkg",
        help="Path to output vector (GPKG)",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    run_production_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_mask_path=args.output_mask,
        output_vector_path=args.output_vector,
        config=config,
    )


if __name__ == "__main__":
    main()
