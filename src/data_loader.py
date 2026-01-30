"""
Data loader for GeoTIFF satellite imagery and SHP crop field boundaries.
Handles both multispectral and RGB imagery with proper georeferencing.
"""

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
from typing import Tuple, Optional, List, Dict
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeoDataLoader:
    """
    Loader for geospatial data: satellite imagery (GeoTIFF) and crop boundaries (SHP).
    """

    def __init__(self,
                 image_path: str,
                 shapefile_path: str,
                 patch_size: int = 256,
                 stride: int = 128,
                 normalize: bool = True,
                 bands: Optional[List[int]] = None,
                 normalize_mode: str = "percentile",
                 p_low: float = 1.0,
                 p_high: float = 99.0):
        """
        Initialize data loader.

        Args:
            image_path: Path to GeoTIFF file
            shapefile_path: Path to shapefile with crop boundaries
            patch_size: Size of image patches for training
            stride: Stride for sliding window
            normalize: Whether to normalize image data
        """
        self.image_path = image_path
        self.shapefile_path = shapefile_path
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize
        self.bands = bands
        self.normalize_mode = normalize_mode
        self.p_low = p_low
        self.p_high = p_high

        # Load data
        self.image_data = None
        self.mask_data = None
        self.transform = None
        self.crs = None
        self.metadata = {}

    def load_image(self) -> np.ndarray:
        """
        Load GeoTIFF image.

        Returns:
            Image array with shape (C, H, W)
        """
        logger.info(f"Loading image from {self.image_path}")

        with rasterio.open(self.image_path) as src:
            # Read all bands
            if self.bands:
                image = src.read(self.bands)
            else:
                image = src.read()
            self.transform = src.transform
            self.crs = src.crs
            self.metadata = {
                'width': src.width,
                'height': src.height,
                'count': image.shape[0],
                'dtype': src.dtypes[0],
                'bounds': src.bounds
            }

            logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")
            logger.info(f"Number of bands: {src.count}")

            # Normalize if requested
            if self.normalize:
                image = self._normalize_image(image)

            self.image_data = image
            return image

    def load_shapefile(self) -> gpd.GeoDataFrame:
        """
        Load shapefile with crop boundaries.

        Returns:
            GeoDataFrame with crop boundaries
        """
        if self.shapefile_path is None or not os.path.exists(self.shapefile_path):
            logger.info("No shapefile provided or file not found. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        logger.info(f"Loading shapefile from {self.shapefile_path}")
        gdf = gpd.read_file(self.shapefile_path)

        # Reproject to match image CRS if needed
        if self.crs is not None and gdf.crs != self.crs:
            logger.info(f"Reprojecting shapefile from {gdf.crs} to {self.crs}")
            gdf = gdf.to_crs(self.crs)

        logger.info(f"Loaded {len(gdf)} crop field polygons")
        return gdf

    def create_mask(self, gdf: Optional[gpd.GeoDataFrame] = None) -> np.ndarray:
        """
        Create binary mask from shapefile polygons.

        Args:
            gdf: GeoDataFrame with polygons. If None, loads from shapefile_path

        Returns:
            Binary mask with shape (H, W)
        """
        if gdf is None:
            gdf = self.load_shapefile()

        # Get image shape
        height = self.metadata['height']
        width = self.metadata['width']

        if len(gdf) == 0:
            logger.info("Empty GeoDataFrame. Creating zero mask.")
            self.mask_data = np.zeros((height, width), dtype=np.uint8)
            return self.mask_data

        logger.info("Creating binary mask from polygons")

        # Rasterize polygons
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=self.transform,
            fill=0,
            dtype=np.uint8
        )

        self.mask_data = mask
        logger.info(f"Mask shape: {mask.shape}, positive pixels: {np.sum(mask)}")
        return mask

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        image = image.astype(np.float32)

        if self.normalize_mode == "percentile":
            # Per-band percentile clipping to [0,1]
            for b in range(image.shape[0]):
                p_low = np.percentile(image[b], self.p_low)
                p_high = np.percentile(image[b], self.p_high)
                if p_high <= p_low:
                    continue
                band = np.clip(image[b], p_low, p_high)
                image[b] = (band - p_low) / (p_high - p_low + 1e-8)
            return image

        # Fallback: min/max by dtype
        if image.dtype == np.uint8:
            return image / 255.0
        if image.dtype == np.uint16:
            return image / 65535.0
        return image

    def extract_patches(self,
                       image: Optional[np.ndarray] = None,
                       mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Extract patches from image and mask using sliding window.

        Args:
            image: Image array (C, H, W). If None, uses loaded image
            mask: Mask array (H, W). If None, uses loaded mask

        Returns:
            List of dictionaries with 'image' and 'mask' patches
        """
        if image is None:
            image = self.image_data
        if mask is None:
            mask = self.mask_data

        if image is None or mask is None:
            raise ValueError("Image and mask must be loaded first")

        patches = []
        C, H, W = image.shape

        # Calculate number of patches
        n_rows = (H - self.patch_size) // self.stride + 1
        n_cols = (W - self.patch_size) // self.stride + 1

        logger.info(f"Extracting {n_rows * n_cols} patches...")

        for i in range(n_rows):
            for j in range(n_cols):
                y = i * self.stride
                x = j * self.stride

                # Extract patch
                img_patch = image[:, y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

                crop_fraction = float(np.mean(mask_patch > 0.5))
                has_crop = crop_fraction > 0.0

                # Skip patches with no crop fields (optional)
                # if np.sum(mask_patch) > 0:
                patches.append({
                    'image': img_patch,
                    'mask': mask_patch,
                    'position': (y, x),
                    'has_crop': has_crop,
                    'crop_fraction': crop_fraction
                })

        logger.info(f"Extracted {len(patches)} patches")
        logger.info(f"Patches with crops: {sum(p['has_crop'] for p in patches)}")

        return patches

    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load both image and mask.

        Returns:
            Tuple of (image, mask)
        """
        image = self.load_image()
        mask = self.create_mask()
        return image, mask

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        if self.image_data is None or self.mask_data is None:
            self.load_all()

        stats = {
            'image_shape': self.image_data.shape,
            'mask_shape': self.mask_data.shape,
            'num_bands': self.image_data.shape[0],
            'image_dtype': str(self.image_data.dtype),
            'mask_dtype': str(self.mask_data.dtype),
            'crop_pixels': int(np.sum(self.mask_data)),
            'total_pixels': int(self.mask_data.size),
            'crop_percentage': float(np.sum(self.mask_data) / self.mask_data.size * 100),
            'bounds': self.metadata.get('bounds'),
            'crs': str(self.crs)
        }

        # Band statistics
        for i in range(self.image_data.shape[0]):
            band = self.image_data[i]
            stats[f'band_{i}_mean'] = float(np.mean(band))
            stats[f'band_{i}_std'] = float(np.std(band))
            stats[f'band_{i}_min'] = float(np.min(band))
            stats[f'band_{i}_max'] = float(np.max(band))

        return stats


class DatasetSplitter:
    """
    Split dataset into train/val/test sets.
    """

    @staticmethod
    def spatial_split(patches: List[Dict],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """
        Spatially split patches into train/val/test.

        Args:
            patches: List of patch dictionaries
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set

        Returns:
            Tuple of (train_patches, val_patches, test_patches)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        n_total = len(patches)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Sort patches by position for spatial splitting
        patches_sorted = sorted(patches, key=lambda x: (x['position'][0], x['position'][1]))

        train_patches = patches_sorted[:n_train]
        val_patches = patches_sorted[n_train:n_train+n_val]
        test_patches = patches_sorted[n_train+n_val:]

        logger.info(f"Split dataset: train={len(train_patches)}, "
                   f"val={len(val_patches)}, test={len(test_patches)}")

        return train_patches, val_patches, test_patches


if __name__ == "__main__":
    # Example usage
    loader = GeoDataLoader(
        image_path="data/raw/satellite_image.tif",
        shapefile_path="data/raw/crop_boundaries.shp",
        patch_size=256,
        stride=128
    )

    # Load data
    image, mask = loader.load_all()

    # Get statistics
    stats = loader.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Extract patches
    patches = loader.extract_patches()

    # Split dataset
    train, val, test = DatasetSplitter.spatial_split(patches)
