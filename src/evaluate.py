"""
Evaluation and visualization for trained RL segmentation model.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import rasterio
from rasterio.transform import from_bounds

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import GeoDataLoader, DatasetSplitter
from src.environment import CropSegmentationEnv
from src.agent import PPOAgent
from src.rewards import MetricsCalculator


class Evaluator:
    """
    Evaluator for trained RL segmentation models.
    """

    def __init__(self, config: Dict, checkpoint_path: str):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.load_data()

        # Initialize agent and load checkpoint
        self.agent = PPOAgent(
            in_channels=config['in_channels'],
            num_actions=config['num_actions'],
            device=str(self.device)
        )
        self.agent.load(checkpoint_path)
        self.agent.network.eval()

        # Metrics calculator
        self.metrics_calc = MetricsCalculator()

    def load_data(self):
        """Load and prepare data."""
        config = self.config

        # Initialize data loader
        loader = GeoDataLoader(
            image_path=config['image_path'],
            shapefile_path=config['shapefile_path'],
            patch_size=config['patch_size'],
            stride=config['stride'],
            normalize=True
        )

        # Load data
        self.full_image, self.full_mask = loader.load_all()
        self.transform = loader.transform
        self.crs = loader.crs

        # Extract patches
        patches = loader.extract_patches()

        # Split into train/val/test
        _, _, self.test_patches = DatasetSplitter.spatial_split(
            patches,
            train_ratio=config.get('train_ratio', 0.7),
            val_ratio=config.get('val_ratio', 0.15),
            test_ratio=config.get('test_ratio', 0.15)
        )

        print(f"Test set size: {len(self.test_patches)} patches")

    def evaluate_patch(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Evaluate single patch.

        Args:
            image: Image patch (C, H, W)
            mask: Ground truth mask (H, W)

        Returns:
            Tuple of (predicted_mask, metrics)
        """
        # Create temporary environment
        env = CropSegmentationEnv(
            image_patches=[image],
            mask_patches=[mask],
            patch_size=self.config['patch_size'],
            action_mode=self.config['action_mode'],
            reward_type=self.config['reward_type'],
            num_actions=self.config['num_actions']
        )

        # Run episode
        obs, _ = env.reset()
        done = False

        while not done:
            action, _, _ = self.agent.select_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Get predicted mask
        predicted_mask = env.predicted_mask

        # Calculate metrics
        metrics = self.metrics_calc.calculate_all(predicted_mask, mask)

        return predicted_mask, metrics

    def evaluate_all(self) -> Dict:
        """
        Evaluate on all test patches.

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        for patch in tqdm(self.test_patches, desc="Evaluating"):
            _, metrics = self.evaluate_patch(patch['image'], patch['mask'])
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_median'] = np.median(values)

        return aggregated

    def visualize_predictions(self, n_samples: int = 5):
        """
        Visualize predictions on random samples.

        Args:
            n_samples: Number of samples to visualize
        """
        # Select random patches
        indices = np.random.choice(len(self.test_patches), min(n_samples, len(self.test_patches)), replace=False)

        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for idx, patch_idx in enumerate(indices):
            patch = self.test_patches[patch_idx]
            image = patch['image']
            gt_mask = patch['mask']

            # Get prediction
            pred_mask, metrics = self.evaluate_patch(image, gt_mask)

            # Visualize
            # Image (RGB or first 3 channels)
            if image.shape[0] >= 3:
                img_vis = np.transpose(image[:3], (1, 2, 0))
            else:
                img_vis = image[0]

            axes[idx, 0].imshow(img_vis)
            axes[idx, 0].set_title('Input Image')
            axes[idx, 0].axis('off')

            # Ground truth
            axes[idx, 1].imshow(gt_mask, cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')

            # Prediction
            axes[idx, 2].imshow(pred_mask, cmap='gray')
            axes[idx, 2].set_title(f'Prediction\nDice: {metrics["dice"]:.3f}')
            axes[idx, 2].axis('off')

            # Overlay
            overlay = img_vis.copy() if len(img_vis.shape) == 3 else np.stack([img_vis]*3, axis=-1)
            if overlay.max() > 1.0:
                overlay = overlay / overlay.max()

            # Color ground truth in green, prediction in red
            overlay_vis = overlay.copy()
            overlay_vis[gt_mask > 0.5, 1] = 1.0  # Green for GT
            overlay_vis[pred_mask > 0.5, 0] = 1.0  # Red for prediction
            # Yellow where they overlap

            axes[idx, 3].imshow(overlay_vis)
            axes[idx, 3].set_title(f'Overlay\nIoU: {metrics["iou"]:.3f}')
            axes[idx, 3].axis('off')

        plt.tight_layout()
        save_path = self.output_dir / 'predictions_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()

    def plot_metrics_distribution(self, metrics: Dict):
        """
        Plot metrics distribution.

        Args:
            metrics: Dictionary with aggregated metrics
        """
        # Extract mean values
        metric_names = ['dice', 'iou', 'pixel_accuracy', 'precision', 'recall', 'f1']
        mean_values = [metrics[f'{name}_mean'] for name in metric_names]
        std_values = [metrics[f'{name}_std'] for name in metric_names]

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metric_names))
        bars = ax.bar(x, mean_values, yerr=std_values, capsize=5, alpha=0.7)

        # Color bars
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Evaluation Metrics on Test Set', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, mean_values, std_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / 'metrics_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
        plt.close()

    def save_full_prediction(self):
        """
        Reconstruct and save full prediction on entire image.
        """
        print("Generating full image prediction...")

        # Initialize full prediction mask
        full_pred = np.zeros_like(self.full_mask, dtype=np.float32)
        counts = np.zeros_like(self.full_mask, dtype=np.float32)

        # Process each test patch
        for patch in tqdm(self.test_patches, desc="Processing patches"):
            image = patch['image']
            mask = patch['mask']
            pos_y, pos_x = patch['position']

            # Get prediction
            pred_mask, _ = self.evaluate_patch(image, mask)

            # Add to full prediction
            h, w = pred_mask.shape
            full_pred[pos_y:pos_y+h, pos_x:pos_x+w] += pred_mask
            counts[pos_y:pos_y+h, pos_x:pos_x+w] += 1

        # Average overlapping predictions
        full_pred = np.divide(full_pred, counts, where=counts > 0)

        # Save as GeoTIFF
        output_path = self.output_dir / 'full_prediction.tif'

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=full_pred.shape[0],
            width=full_pred.shape[1],
            count=1,
            dtype=full_pred.dtype,
            crs=self.crs,
            transform=self.transform
        ) as dst:
            dst.write(full_pred, 1)

        print(f"Full prediction saved to {output_path}")

        # Also save visualization
        self._visualize_full_prediction(full_pred)

    def _visualize_full_prediction(self, full_pred: np.ndarray):
        """Visualize full prediction."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image (first 3 bands)
        if self.full_image.shape[0] >= 3:
            img_vis = np.transpose(self.full_image[:3], (1, 2, 0))
            # Normalize for visualization
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        else:
            img_vis = self.full_image[0]

        axes[0].imshow(img_vis)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')

        # Ground truth
        axes[1].imshow(self.full_mask, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontsize=14)
        axes[1].axis('off')

        # Prediction
        im = axes[2].imshow(full_pred, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2].set_title('RL Agent Prediction', fontsize=14)
        axes[2].axis('off')

        # Add colorbar
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = self.output_dir / 'full_prediction_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Full prediction visualization saved to {save_path}")
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained RL segmentation model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--save_full', action='store_true',
                       help='Save full image prediction')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found. Using default config.")
        from src.train import get_default_config
        config = get_default_config()

    config['output_dir'] = args.output

    # Initialize evaluator
    evaluator = Evaluator(config, args.checkpoint)

    # Evaluate
    print("Evaluating model on test set...")
    metrics = evaluator.evaluate_all()

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50 + "\n")

    # Visualizations
    evaluator.visualize_predictions(n_samples=args.n_samples)
    evaluator.plot_metrics_distribution(metrics)

    # Save full prediction if requested
    if args.save_full:
        evaluator.save_full_prediction()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
