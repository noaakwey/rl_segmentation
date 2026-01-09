"""
Inference script for trained RL segmentation model on new satellite images.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import GeoDataLoader
from src.environment import CropSegmentationEnv
from src.agent import PPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationInference:
    """
    Inference engine for crop segmentation.
    """

    def __init__(self,
                 model_path: str,
                 in_channels: int = 4,
                 num_actions: int = 16,
                 patch_size: int = 256,
                 stride: int = 128):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint
            in_channels: Number of input channels
            num_actions: Number of actions
            patch_size: Size of patches
            stride: Stride for sliding window
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.stride = stride

        # Load agent
        logger.info(f"Loading model from {model_path}")
        self.agent = PPOAgent(
            in_channels=in_channels,
            num_actions=num_actions,
            device=str(self.device)
        )
        self.agent.load(model_path)
        self.agent.network.eval()

        logger.info("Model loaded successfully")

    def predict_patch(self, image: np.ndarray) -> np.ndarray:
        """
        Predict segmentation for a single patch.

        Args:
            image: Image patch (C, H, W)

        Returns:
            Predicted mask (H, W)
        """
        # Create dummy mask for environment
        dummy_mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)

        # Create environment
        env = CropSegmentationEnv(
            image_patches=[image],
            mask_patches=[dummy_mask],
            patch_size=self.patch_size,
            action_mode='patch',
            reward_type='dice',
            num_actions=self.agent.network.policy_net.network[-1].out_features // 2
        )

        # Run prediction
        obs, _ = env.reset()
        done = False

        with torch.no_grad():
            while not done:
                action, _, _ = self.agent.select_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        return env.predicted_mask

    def predict_image(self,
                     image_path: str,
                     output_path: str,
                     threshold: float = 0.5) -> np.ndarray:
        """
        Predict segmentation for entire image.

        Args:
            image_path: Path to input GeoTIFF
            output_path: Path to save prediction
            threshold: Threshold for binary mask

        Returns:
            Predicted mask
        """
        logger.info(f"Loading image from {image_path}")

        # Load image
        with rasterio.open(image_path) as src:
            image = src.read()
            transform = src.transform
            crs = src.crs
            height, width = image.shape[1], image.shape[2]

            logger.info(f"Image size: {width}x{height}, bands: {image.shape[0]}")

            # Normalize
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0

        # Initialize prediction arrays
        full_pred = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        # Extract patches
        n_rows = (height - self.patch_size) // self.stride + 1
        n_cols = (width - self.patch_size) // self.stride + 1
        total_patches = n_rows * n_cols

        logger.info(f"Processing {total_patches} patches...")

        # Process patches
        for i in tqdm(range(n_rows)):
            for j in range(n_cols):
                y = i * self.stride
                x = j * self.stride

                # Extract patch
                patch = image[:, y:y+self.patch_size, x:x+self.patch_size]

                # Predict
                pred_mask = self.predict_patch(patch)

                # Add to full prediction
                h, w = pred_mask.shape
                full_pred[y:y+h, x:x+w] += pred_mask
                counts[y:y+h, x:x+w] += 1

        # Average overlapping predictions
        full_pred = np.divide(full_pred, counts, where=counts > 0)

        # Apply threshold
        binary_mask = (full_pred > threshold).astype(np.uint8)

        # Save prediction
        logger.info(f"Saving prediction to {output_path}")

        # Save probability map
        prob_path = output_path.replace('.tif', '_prob.tif')
        with rasterio.open(
            prob_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=full_pred.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(full_pred, 1)

        logger.info(f"Probability map saved to {prob_path}")

        # Save binary mask
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=binary_mask.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(binary_mask, 1)

        logger.info(f"Binary mask saved to {output_path}")

        # Calculate statistics
        crop_pixels = np.sum(binary_mask)
        total_pixels = binary_mask.size
        crop_percentage = crop_pixels / total_pixels * 100

        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Crop pixels: {crop_pixels:,}")
        logger.info(f"  Total pixels: {total_pixels:,}")
        logger.info(f"  Crop percentage: {crop_percentage:.2f}%")

        return binary_mask


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference on satellite image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input GeoTIFF image')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save prediction')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size')
    parser.add_argument('--stride', type=int, default=128,
                       help='Stride for sliding window')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    parser.add_argument('--in_channels', type=int, default=4,
                       help='Number of input channels')
    parser.add_argument('--num_actions', type=int, default=16,
                       help='Number of actions')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference engine
    engine = SegmentationInference(
        model_path=args.model,
        in_channels=args.in_channels,
        num_actions=args.num_actions,
        patch_size=args.patch_size,
        stride=args.stride
    )

    # Run prediction
    prediction = engine.predict_image(
        image_path=args.image,
        output_path=args.output,
        threshold=args.threshold
    )

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
