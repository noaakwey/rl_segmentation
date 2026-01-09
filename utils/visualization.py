"""
Visualization utilities for RL segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path


def plot_training_curves(metrics: Dict[str, List[float]],
                         save_path: Optional[str] = None,
                         figsize: tuple = (15, 10)):
    """
    Plot training curves for various metrics.

    Args:
        metrics: Dictionary with metric names and values over time
        save_path: Path to save figure
        figsize: Figure size
    """
    sns.set_style('whitegrid')

    # Create subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(values, linewidth=2)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode/Epoch')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        # Add smoothed curve if enough data points
        if len(values) > 10:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(values, size=min(10, len(values)//3))
            ax.plot(smoothed, linewidth=2, linestyle='--', alpha=0.7, label='Smoothed')
            ax.legend()

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_segmentation_comparison(image: np.ndarray,
                                ground_truth: np.ndarray,
                                prediction: np.ndarray,
                                metrics: Optional[Dict] = None,
                                save_path: Optional[str] = None):
    """
    Plot comparison of ground truth and prediction.

    Args:
        image: Input image (C, H, W) or (H, W)
        ground_truth: Ground truth mask (H, W)
        prediction: Predicted mask (H, W)
        metrics: Optional metrics dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Prepare image for visualization
    if len(image.shape) == 3:
        if image.shape[0] >= 3:
            img_vis = np.transpose(image[:3], (1, 2, 0))
        else:
            img_vis = image[0]
    else:
        img_vis = image

    # Normalize image
    if img_vis.max() > 1.0:
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

    # Plot input image
    axes[0].imshow(img_vis, cmap='gray' if len(img_vis.shape) == 2 else None)
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')

    # Plot ground truth
    axes[1].imshow(ground_truth, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')

    # Plot prediction
    im = axes[2].imshow(prediction, cmap='RdYlGn', vmin=0, vmax=1)
    title = 'Prediction'
    if metrics:
        title += f"\nDice: {metrics.get('dice', 0):.3f}"
    axes[2].set_title(title, fontsize=14)
    axes[2].axis('off')

    # Plot overlay
    overlay = np.zeros((*ground_truth.shape, 3))
    overlay[ground_truth > 0.5] = [0, 1, 0]  # Green for GT
    overlay[prediction > 0.5] = [1, 0, 0]    # Red for prediction
    overlay[(ground_truth > 0.5) & (prediction > 0.5)] = [1, 1, 0]  # Yellow for overlap

    axes[3].imshow(overlay)
    title = 'Overlay (GT=Green, Pred=Red)'
    if metrics:
        title += f"\nIoU: {metrics.get('iou', 0):.3f}"
    axes[3].set_title(title, fontsize=14)
    axes[3].axis('off')

    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")

    plt.show()


def plot_metrics_heatmap(metrics_matrix: np.ndarray,
                        metric_names: List[str],
                        sample_names: List[str],
                        save_path: Optional[str] = None):
    """
    Plot heatmap of metrics across samples.

    Args:
        metrics_matrix: Matrix of metrics (samples x metrics)
        metric_names: List of metric names
        sample_names: List of sample names
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(metrics_matrix,
               annot=True,
               fmt='.3f',
               cmap='YlGnBu',
               xticklabels=metric_names,
               yticklabels=sample_names,
               cbar_kws={'label': 'Metric Value'},
               ax=ax)

    ax.set_title('Metrics Across Samples', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Samples', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics heatmap saved to {save_path}")

    plt.show()


def plot_reward_distribution(rewards: List[float],
                            save_path: Optional[str] = None):
    """
    Plot distribution of rewards.

    Args:
        rewards: List of reward values
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    axes[0].axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.3f}')
    axes[0].set_xlabel('Reward', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(rewards, vert=True)
    axes[1].set_ylabel('Reward', fontsize=12)
    axes[1].set_title('Reward Statistics', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reward distribution saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test visualizations
    metrics = {
        'reward': np.cumsum(np.random.randn(100) * 0.1),
        'dice': np.random.rand(100) * 0.5 + 0.5,
        'iou': np.random.rand(100) * 0.5 + 0.4,
    }
    plot_training_curves(metrics)

    # Test comparison
    image = np.random.rand(256, 256)
    gt = (np.random.rand(256, 256) > 0.5).astype(float)
    pred = (np.random.rand(256, 256) > 0.5).astype(float)
    plot_segmentation_comparison(image, gt, pred, {'dice': 0.85, 'iou': 0.75})
