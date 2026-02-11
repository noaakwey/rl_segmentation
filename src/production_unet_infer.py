"""
Production inference for large rasters using supervised U-Net.

Key ideas:
- Stream tiles from disk (no full-image loads)
- Overlapping tiles with cosine-taper blending → no seams
- Optional morphological opening → preserves field boundaries
- Prefetch tiles in background thread to keep GPU utilized
- Multi-GPU support via DataParallel
"""

import os
import sys
import argparse
import threading
import queue
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import UNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _normalize_tile(
    tile: np.ndarray,
    normalize_mode: str,
    global_p1: Optional[np.ndarray],
    global_p99: Optional[np.ndarray],
    p_low: float,
    p_high: float,
) -> np.ndarray:
    """Normalize a single tile (C, H, W). Vectorized when global percentiles available."""
    if normalize_mode == "percentile":
        tile = tile.astype(np.float32)
        if global_p1 is not None and global_p99 is not None:
            lo = global_p1[:, None, None]
            hi = global_p99[:, None, None]
            tile = np.clip(tile, lo, hi)
            tile = (tile - lo) / (hi - lo + 1e-8)
        else:
            for b in range(tile.shape[0]):
                v_lo = np.percentile(tile[b], p_low)
                v_hi = np.percentile(tile[b], p_high)
                if v_hi <= v_lo:
                    continue
                band = np.clip(tile[b], v_lo, v_hi)
                tile[b] = (band - v_lo) / (v_hi - v_lo + 1e-8)
        return tile
    return _normalize(tile)


def _make_blend_weights(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """Create 2D cosine-taper blending weights. Center=1, tapers to 0 at edges."""
    def _taper(size: int, olap: int) -> np.ndarray:
        w = np.ones(size, dtype=np.float32)
        if olap > 0 and olap < size:
            ramp = np.sin(np.linspace(0, np.pi / 2, olap, dtype=np.float32)) ** 2
            w[:olap] = ramp
            w[-olap:] = ramp[::-1]
        return w
    return _taper(tile_h, overlap)[:, None] * _taper(tile_w, overlap)[None, :]


def _morph_open(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening (erosion → dilation) to break thin connections."""
    from scipy.ndimage import binary_opening
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    struct = (x ** 2 + y ** 2) <= radius ** 2
    return binary_opening(mask, structure=struct).astype(np.uint8)


# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict_prob(model, image: np.ndarray, device: torch.device, use_tta: bool) -> np.ndarray:
    """Predict probability mask for a single tile (C, H, W)."""
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
        return prob[:h, :w]

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


def _predict_prob_batch(model, images: List[np.ndarray], device: torch.device) -> List[np.ndarray]:
    """Predict probability masks for a batch of tiles. Uses pinned memory."""
    if not images:
        return []

    use_amp = device.type == "cuda"
    pin = device.type == "cuda"
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    c = images[0].shape[0]
    pad_h = (16 - (max_h % 16)) % 16
    pad_w = (16 - (max_w % 16)) % 16
    padded_h = max_h + pad_h
    padded_w = max_w + pad_w

    batch = torch.zeros((len(images), c, padded_h, padded_w), dtype=torch.float32, pin_memory=pin)
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        batch[i, :, :h, :w] = torch.from_numpy(img)

    x = batch.to(device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        logits = model(x)
        probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

    out: List[np.ndarray] = []
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        out.append(probs[i, :h, :w])
    return out


# ---------------------------------------------------------------------------
# Window iteration
# ---------------------------------------------------------------------------

def _iter_windows(width: int, height: int, tile_size: int, stride: Optional[int] = None):
    """Iterate write windows. stride < tile_size → overlapping tiles."""
    if stride is None:
        stride = tile_size
    for top in range(0, height, stride):
        h = min(tile_size, height - top)
        for left in range(0, width, stride):
            w = min(tile_size, width - left)
            yield left, top, Window(left, top, w, h)


def _parse_window(window_str: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in window_str.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("window must be 'left,top,width,height'")
    left, top, width, height = [int(p) for p in parts]
    if width <= 0 or height <= 0:
        raise ValueError("window width/height must be positive")
    return left, top, width, height


def _compute_global_percentiles(
    src: rasterio.io.DatasetReader,
    bands: List[int],
    p_low: float,
    p_high: float,
    downsample: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    h = min(downsample, src.height)
    w = min(downsample, src.width)
    data = src.read(indexes=bands, out_shape=(len(bands), h, w))
    p1 = []
    p99 = []
    for b in range(data.shape[0]):
        band = data[b].astype(np.float32)
        p1.append(np.percentile(band, p_low))
        p99.append(np.percentile(band, p_high))
    return np.array(p1, dtype=np.float32), np.array(p99, dtype=np.float32)


# ---------------------------------------------------------------------------
# Prefetch: background thread reads and normalizes tiles
# ---------------------------------------------------------------------------

def _prefetch_worker(
    image_path: str,
    bands: List[int],
    windows: List[Tuple[int, int, Window]],
    crop_left: int,
    crop_top: int,
    crop_width: int,
    context: int,
    normalize_mode: str,
    global_p1: Optional[np.ndarray],
    global_p99: Optional[np.ndarray],
    p_low: float,
    p_high: float,
    use_alpha_mask: bool,
    out_queue: queue.Queue,
):
    """Background thread: read horizontal strips (sequential HDD I/O),
    then slice individual tiles from memory.

    For each row of tiles we issue ONE large sequential read spanning the full
    crop width (+context).  Individual tiles are sliced from the in-memory
    strip with zero disk I/O — dramatically reducing seek overhead on HDDs.
    """
    try:
        with rasterio.open(image_path) as src:
            strip_read_left = crop_left - context
            strip_read_width = crop_width + 2 * context
            prev_row_top = -1
            strip = None

            for left, top, write_window in windows:
                write_w = int(write_window.width)
                write_h = int(write_window.height)
                read_h = write_h + 2 * context
                read_w = write_w + 2 * context

                # New row → one big sequential read for the whole strip
                if top != prev_row_top:
                    read_top = crop_top + top - context
                    s_win = Window(strip_read_left, read_top,
                                   strip_read_width, read_h)
                    strip = src.read(indexes=bands, window=s_win,
                                     boundless=True, fill_value=0)
                    prev_row_top = top

                # Slice tile from the in-memory strip (no disk I/O)
                col_start = left  # = (crop_left+left-context) - strip_read_left
                tile = strip[:, :read_h, col_start:col_start + read_w].copy()

                if use_alpha_mask and src.count >= 4 and 4 in bands:
                    alpha_idx = bands.index(4)
                    if np.max(tile[alpha_idx]) == 0:
                        out_queue.put((None, write_window, write_h, write_w))
                        continue

                tile = _normalize_tile(tile, normalize_mode, global_p1, global_p99, p_low, p_high)
                out_queue.put((tile, write_window, write_h, write_w))
    finally:
        out_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def run_production_inference(
    image_path: str,
    checkpoint_path: str,
    output_mask_path: str,
    output_vector_path: str,
    config: Dict,
    window: Tuple[int, int, int, int] = None,
    quadrant: str = None,
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
    use_alpha_mask = bool(prod_cfg.get("use_alpha_mask", True))
    global_percentile = bool(prod_cfg.get("global_percentile", False))
    downsample = int(prod_cfg.get("percentile_downsample", 1024))
    prefetch_count = int(prod_cfg.get("prefetch", batch_size * 4))
    overlap = int(prod_cfg.get("overlap", 0))
    morph_radius = int(prod_cfg.get("morph_open_radius", 0))
    normalize_cfg = config.get("normalization", {})
    normalize_mode = normalize_cfg.get("mode", "percentile")
    p_low = float(normalize_cfg.get("p_low", 1.0))
    p_high = float(normalize_cfg.get("p_high", 99.0))

    use_blend = overlap > 0
    stride = tile_size - overlap if use_blend else tile_size

    # Load model
    with rasterio.open(image_path) as src:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        required_in_channels = int(checkpoint["model_state_dict"]["enc1.0.conv.weight"].shape[1])
        saved_percentiles = checkpoint.get("norm_percentiles")

        bands = prod_cfg.get("bands")
        if bands is None:
            bands = list(range(1, required_in_channels + 1))
        if isinstance(bands, str):
            if bands.strip().lower() == "all":
                bands = list(range(1, required_in_channels + 1))
            else:
                parts = [p.strip() for p in bands.split(",") if p.strip()]
                bands = [int(p) for p in parts]
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
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
        use_multi_gpu = bool(prod_cfg.get("multi_gpu", False))
        if num_gpus > 1 and use_multi_gpu:
            model = torch.nn.DataParallel(model)
            print(f"Using {num_gpus} GPUs (batch_size={batch_size} split across GPUs)")
        else:
            print(f"Using single GPU: {device}")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff", count=1, dtype="uint8", compress="lzw",
            tiled=True, blockxsize=tile_size, blockysize=tile_size,
            BIGTIFF="YES", nodata=0,
        )

        output_mask_path = str(Path(output_mask_path))
        Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)

        src_width, src_height = src.width, src.height
        crop_left, crop_top = 0, 0
        crop_width, crop_height = src_width, src_height

        if quadrant:
            q = quadrant.strip().lower()
            half_w, half_h = src_width // 2, src_height // 2
            offsets = {
                "ul": (0, 0, half_w, half_h),
                "ur": (half_w, 0, src_width - half_w, half_h),
                "ll": (0, half_h, half_w, src_height - half_h),
                "lr": (half_w, half_h, src_width - half_w, src_height - half_h),
            }
            if q not in offsets:
                raise ValueError("quadrant must be one of: ul, ur, ll, lr")
            crop_left, crop_top, crop_width, crop_height = offsets[q]

        if window:
            crop_left, crop_top, crop_width, crop_height = window

        crop_left = max(0, min(crop_left, src_width - 1))
        crop_top = max(0, min(crop_top, src_height - 1))
        crop_width = max(1, min(crop_width, src_width - crop_left))
        crop_height = max(1, min(crop_height, src_height - crop_top))

        width, height = crop_width, crop_height
        print(f"Raster size: {src_width} x {src_height}")
        if window or quadrant:
            print(f"Crop window: left={crop_left}, top={crop_top}, w={width}, h={height}")
        print(
            f"Tile: {tile_size}, context: {context}, threshold: {threshold:.3f}, "
            f"TTA: {use_tta}, batch: {batch_size}, bands: {bands}, device: {device}"
        )
        if use_blend:
            print(f"Blend: overlap={overlap}, stride={stride}, morph_radius={morph_radius}")
        strip_mem_mb = (width + 2 * context) * (tile_size + 2 * context) * len(bands) * 2 / 1024**2
        print(f"Strip-read mode: ~{strip_mem_mb:.0f} MB per strip (sequential HDD I/O)")

        if window or quadrant:
            profile.update(
                width=width, height=height,
                transform=rasterio.windows.transform(
                    Window(crop_left, crop_top, width, height), src.transform
                ),
            )

        # Normalization percentiles
        global_p1: Optional[np.ndarray] = None
        global_p99: Optional[np.ndarray] = None
        if normalize_mode == "percentile" and saved_percentiles is not None:
            sp_low = saved_percentiles["p_low"]
            sp_high = saved_percentiles["p_high"]
            if len(sp_low) == len(bands):
                global_p1 = np.array(sp_low, dtype=np.float32)
                global_p99 = np.array(sp_high, dtype=np.float32)
                print("Using normalization percentiles from training checkpoint")
            else:
                print(
                    f"Warning: checkpoint percentiles have {len(sp_low)} bands, "
                    f"but {len(bands)} requested. Falling back."
                )
        if normalize_mode == "percentile" and global_p1 is None and global_percentile:
            global_p1, global_p99 = _compute_global_percentiles(
                src, bands, p_low, p_high, downsample=downsample
            )
        if normalize_mode == "percentile" and global_p1 is None:
            print("Warning: per-tile percentile normalization. Re-train to save percentiles.")

    # --- Build tile list and accumulators ---
    all_windows = list(_iter_windows(width, height, tile_size, stride))
    total_tiles = len(all_windows)

    temp_dir = None
    prob_acc = None
    weight_acc = None
    blend_w = None

    if use_blend:
        blend_w = _make_blend_weights(tile_size, tile_size, overlap)
        temp_base = prod_cfg.get("temp_dir", None)
        temp_dir = tempfile.mkdtemp(prefix="unet_blend_", dir=temp_base)
        prob_acc = np.memmap(
            os.path.join(temp_dir, "prob.dat"), dtype=np.float32, mode="w+", shape=(height, width)
        )
        weight_acc = np.memmap(
            os.path.join(temp_dir, "weight.dat"), dtype=np.float32, mode="w+", shape=(height, width)
        )
        print(f"Blend accumulators: {height}x{width} float32 (temp: {temp_dir})")

    # --- Prefetch thread ---
    tile_queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        kwargs=dict(
            image_path=image_path, bands=bands, windows=all_windows,
            crop_left=crop_left, crop_top=crop_top, crop_width=width,
            context=context,
            normalize_mode=normalize_mode, global_p1=global_p1, global_p99=global_p99,
            p_low=p_low, p_high=p_high, use_alpha_mask=use_alpha_mask,
            out_queue=tile_queue,
        ),
        daemon=True,
    )

    # --- Inference loop ---
    with rasterio.open(output_mask_path, "w", **profile) as dst:
        tiles_buf: List[np.ndarray] = []
        meta_buf: List[Tuple[Window, int, int]] = []

        def flush():
            if not tiles_buf:
                return
            if use_tta:
                probs = [_predict_prob(model, t, device, True) for t in tiles_buf]
            else:
                probs = _predict_prob_batch(model, tiles_buf, device)

            for prob_full, (wwin, wh, ww) in zip(probs, meta_buf):
                pc = prob_full[context: context + wh, context: context + ww]
                if use_blend:
                    y, x = wwin.row_off, wwin.col_off
                    bw = blend_w[:wh, :ww]
                    prob_acc[y: y + wh, x: x + ww] += pc * bw
                    weight_acc[y: y + wh, x: x + ww] += bw
                else:
                    mask = (pc > threshold).astype(np.uint8)
                    dst.write(mask, 1, window=wwin)

            tiles_buf.clear()
            meta_buf.clear()

        prefetch_thread.start()
        pbar = tqdm(total=total_tiles, desc="Tiles")

        while True:
            item = tile_queue.get()
            if item is None:
                break

            tile, wwin, wh, ww = item
            pbar.update(1)

            if tile is None:  # alpha-skip
                if not use_blend:
                    dst.write(np.zeros((wh, ww), dtype=np.uint8), 1, window=wwin)
                continue

            tiles_buf.append(tile)
            meta_buf.append((wwin, wh, ww))
            if len(tiles_buf) >= batch_size:
                flush()

        flush()
        pbar.close()

        # --- Finalize blended predictions ---
        if use_blend:
            print("Finalizing blended predictions...")
            valid = weight_acc > 0
            prob_acc[valid] /= weight_acc[valid]

            block_h = tile_size
            pad = morph_radius  # overlap to avoid boundary artifacts
            for row in tqdm(range(0, height, block_h), desc="Writing"):
                bh = min(block_h, height - row)
                # Read with padding so morph_open sees neighbours across blocks
                r0 = max(0, row - pad)
                r1 = min(height, row + bh + pad)
                block = (prob_acc[r0:r1, :] > threshold).astype(np.uint8)
                if morph_radius > 0:
                    block = _morph_open(block, morph_radius)
                # Crop back to the inner region
                top_pad = row - r0
                block = block[top_pad:top_pad + bh, :]
                dst.write(block, 1, window=Window(0, row, width, bh))

    prefetch_thread.join()

    # Cleanup temp
    if temp_dir:
        del prob_acc, weight_acc
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Mask saved to {output_mask_path}")

    if do_vectorize:
        vectorize_mask(
            mask_path=output_mask_path,
            output_vector_path=output_vector_path,
            min_area_pixels=min_area_pixels,
        )
    else:
        print("Vectorization disabled.")


def vectorize_mask(mask_path: str, output_vector_path: str, min_area_pixels: int = 1024):
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

        gdf = gpd.GeoDataFrame(
            {"value": [v for _, v in geoms]}, geometry=[g for g, _ in geoms], crs=src.crs
        )

    if min_area_pixels > 1 and len(gdf) > 0:
        with rasterio.open(mask_path) as src:
            px_w = abs(src.transform.a)
            px_h = abs(src.transform.e)
        min_area = float(min_area_pixels) * px_w * px_h
        gdf = gdf[gdf.geometry.area >= min_area]

    if len(gdf) > 0:
        gdf["dissolve_key"] = 1
        gdf = gdf.dissolve(by="dissolve_key").explode(index_parts=False).reset_index(drop=True)
        gdf = gdf.drop(columns=["dissolve_key"], errors="ignore")

    gdf.to_file(output_vector_path, driver="GPKG")
    print(f"Vector saved to {output_vector_path} (features={len(gdf)})")


def main():
    parser = argparse.ArgumentParser(description="Production U-Net inference on large rasters")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/custom_config.yaml")
    parser.add_argument("--output-mask", type=str, default="experiments/production/midvolga_mask.tif")
    parser.add_argument("--output-vector", type=str, default="experiments/production/midvolga_mask.gpkg")
    parser.add_argument("--window", type=str, default=None,
                        help="Crop window as 'left,top,width,height'")
    parser.add_argument("--quadrant", type=str, default=None,
                        help="Quadrant shortcut: ul, ur, ll, lr")
    args = parser.parse_args()

    config = _load_config(args.config)
    win = _parse_window(args.window) if args.window else None
    run_production_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_mask_path=args.output_mask,
        output_vector_path=args.output_vector,
        config=config,
        window=win,
        quadrant=args.quadrant,
    )


if __name__ == "__main__":
    main()
