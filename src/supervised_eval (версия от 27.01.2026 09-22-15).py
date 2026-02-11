"""
Evaluation for supervised U-Net segmentation model.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import GeoDataLoader, DatasetSplitter
from src.models import UNet
from src.rewards import MetricsCalculator


def _apply_tta(image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """Return TTA variants with tags describing the transform."""
    return [
        (image, 'none'),
        (np.flip(image, axis=2).copy(), 'hflip'),
        (np.flip(image, axis=1).copy(), 'vflip'),
        (np.flip(np.flip(image, axis=1), axis=2).copy(), 'hvflip'),
    ]


def _invert_tta(mask: np.ndarray, tag: str) -> np.ndarray:
    """Invert TTA transform back to the original orientation."""
    if tag == 'hflip':
        return np.flip(mask, axis=1)
    if tag == 'vflip':
        return np.flip(mask, axis=0)
    if tag == 'hvflip':
        return np.flip(np.flip(mask, axis=0), axis=1)
    return mask


def _predict_probs(model: UNet,
                   patches: List[Dict],
                   device: torch.device,
                   use_tta: bool = False) -> List[np.ndarray]:
    """Predict probability masks for a list of patches (optionally with TTA)."""
    probs = []
    with torch.no_grad():
        for patch in tqdm(patches, desc="Predicting", leave=False):
            image = patch['image']
            if not use_tta:
                img = torch.from_numpy(image).float().unsqueeze(0).to(device)
                logits = model(img)
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            else:
                tta_probs = []
                for aug_img, tag in _apply_tta(image):
                    img = torch.from_numpy(aug_img).float().unsqueeze(0).to(device)
                    logits = model(img)
                    aug_prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
                    tta_probs.append(_invert_tta(aug_prob, tag))
                prob = np.mean(tta_probs, axis=0)

            prob = prob.astype(np.float16)
            probs.append(prob)
    return probs


def _aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
        aggregated[f'{key}_median'] = float(np.median(values))
    return aggregated


def _evaluate_at_threshold(probs: List[np.ndarray],
                           patches: List[Dict],
                           threshold: float,
                           metrics_calc: MetricsCalculator) -> Dict[str, float]:
    """Evaluate metrics at a fixed probability threshold."""
    all_metrics: List[Dict[str, float]] = []
    for prob, patch in zip(probs, patches):
        pred = (prob > threshold).astype(np.float32)
        gt = patch['mask']
        all_metrics.append(metrics_calc.calculate_all(pred, gt))
    return _aggregate_metrics(all_metrics)


def _find_best_threshold(val_probs: List[np.ndarray],
                         val_patches: List[Dict],
                         metrics_calc: MetricsCalculator,
                         thresholds: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Select threshold by maximizing val Dice (reduces overfitting to test)."""
    best_threshold = float(thresholds[0])
    best_metrics: Dict[str, float] = {}
    best_dice = -1.0

    for thr in thresholds:
        metrics = _evaluate_at_threshold(val_probs, val_patches, float(thr), metrics_calc)
        dice = metrics['dice_mean']
        if dice > best_dice:
            best_dice = dice
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def _visualize_predictions(patches: List[Dict],
                           probs: List[np.ndarray],
                           threshold: float,
                           output_dir: Path,
                           n_samples: int = 6):
    """Save qualitative visualization for supervised predictions."""
    if not patches:
        return

    rng = np.random.default_rng(42)
    indices = rng.choice(len(patches), size=min(n_samples, len(patches)), replace=False)

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        patch = patches[int(idx)]
        prob = probs[int(idx)]
        pred = (prob > threshold).astype(np.float32)
        gt = patch['mask']
        image = patch['image']

        if image.shape[0] >= 3:
            img_vis = np.transpose(image[:3], (1, 2, 0))
        else:
            img_vis = image[0]

        axes[row, 0].imshow(img_vis)
        axes[row, 0].set_title('Input Image')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gt, cmap='gray')
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred, cmap='gray')
        axes[row, 2].set_title(f'Prediction (thr={threshold:.2f})')
        axes[row, 2].axis('off')

        overlay = img_vis.copy() if img_vis.ndim == 3 else np.stack([img_vis] * 3, axis=-1)
        if overlay.max() > 1.0:
            overlay = overlay / (overlay.max() + 1e-8)
        overlay_vis = overlay.copy()
        overlay_vis[gt > 0.5, 1] = 1.0
        overlay_vis[pred > 0.5, 0] = 1.0

        axes[row, 3].imshow(overlay_vis)
        axes[row, 3].set_title('Overlay (GT=green, Pred=red)')
        axes[row, 3].axis('off')

    plt.tight_layout()
    save_path = output_dir / 'supervised_predictions_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {save_path}")


def _plot_metrics_distribution(metrics: Dict[str, float], output_dir: Path):
    """Plot aggregated metrics with error bars."""
    metric_names = ['dice', 'iou', 'pixel_accuracy', 'precision', 'recall', 'f1', 'boundary_iou']
    mean_values = [metrics[f'{name}_mean'] for name in metric_names]
    std_values = [metrics[f'{name}_std'] for name in metric_names]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    bars = ax.bar(x, mean_values, yerr=std_values, capsize=5, alpha=0.8)

    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(metric_names)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Supervised Evaluation Metrics on Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=35, ha='right')
    ax.set_ylim(0, 1.0)

    for bar, mean, std in zip(bars, mean_values, std_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = output_dir / 'supervised_metrics_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Metrics plot saved to {save_path}")


def evaluate_supervised(config: Dict, checkpoint_path: str) -> Tuple[Dict, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = GeoDataLoader(
        image_path=config['image_path'],
        shapefile_path=config['shapefile_path'],
        patch_size=config['patch_size'],
        stride=config['stride'],
        normalize=True
    )

    image, _ = loader.load_all()
    patches = loader.extract_patches()
    _, val_patches, test_patches = DatasetSplitter.spatial_split(
        patches,
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15)
    )

    print(f"Test set size: {len(test_patches)} patches")

    model = UNet(in_channels=image.shape[0]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    metrics_calc = MetricsCalculator()
    eval_cfg = config.get('supervised_eval', {})
    thresholds = np.linspace(
        eval_cfg.get('threshold_min', 0.3),
        eval_cfg.get('threshold_max', 0.7),
        int(eval_cfg.get('threshold_steps', 9))
    )
    use_tta = bool(eval_cfg.get('use_tta', True))

    print("Predicting on validation set for threshold selection...")
    val_probs = _predict_probs(model, val_patches, device, use_tta=use_tta)
    best_threshold, val_metrics = _find_best_threshold(val_probs, val_patches, metrics_calc, thresholds)
    print(f"Best threshold on val: {best_threshold:.3f} (val dice={val_metrics['dice_mean']:.4f})")

    print("Predicting on test set...")
    test_probs = _predict_probs(model, test_patches, device, use_tta=use_tta)

    print("Evaluating on test set...")
    aggregated = _evaluate_at_threshold(test_probs, test_patches, best_threshold, metrics_calc)

    # Save plots and visualizations to a dedicated folder
    output_dir = Path(config['output_dir']) / 'supervised_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    _visualize_predictions(
        test_patches,
        test_probs,
        threshold=best_threshold,
        output_dir=output_dir,
        n_samples=int(eval_cfg.get('n_visualize', 6))
    )
    _plot_metrics_distribution(aggregated, output_dir)

    return aggregated, best_threshold


def main():
    parser = argparse.ArgumentParser(description='Evaluate supervised U-Net model')
    parser.add_argument('--config', type=str, default='configs/custom_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file {args.config} not found.")

    metrics, threshold = evaluate_supervised(config, args.checkpoint)
    print("\n==================================================")
    print("EVALUATION RESULTS (SUPERVISED)")
    print("==================================================")
    print(f"threshold_selected_on_val: {threshold:.4f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
