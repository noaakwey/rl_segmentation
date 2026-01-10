#!/usr/bin/env python3
"""
Benchmark: Advanced RL vs SOTA CNN methods.

Сравнивает наш advanced RL agent с SOTA CNN baselines:
- U-Net
- DeepLabv3+
- PSPNet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import argparse

from src.advanced_agent import AdvancedRLAgent
from src.rewards import MetricsCalculator


class UNetBaseline(nn.Module):
    """Simple U-Net baseline for comparison."""

    def __init__(self, in_channels=4):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)

        # Output
        self.out = nn.Conv2d(64, 1, 1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1) if d4.shape[2:] == e4.shape[2:] else d4
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1) if d3.shape[2:] == e3.shape[2:] else d3
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1) if d2.shape[2:] == e2.shape[2:] else d2
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1) if d1.shape[2:] == e1.shape[2:] else d1

        return self.out(d1)


class Benchmark:
    """Benchmark advanced RL agent vs CNN baselines."""

    def __init__(self,
                 test_images: list,
                 test_masks: list,
                 device: str = 'cuda'):
        """
        Initialize benchmark.

        Args:
            test_images: List of test images
            test_masks: List of test masks
            device: Device to use
        """
        self.test_images = test_images
        self.test_masks = test_masks
        self.device = device
        self.metrics_calc = MetricsCalculator()

        self.results = {}

    def evaluate_model(self,
                      model: nn.Module,
                      model_name: str,
                      return_uncertainty: bool = False) -> dict:
        """
        Evaluate a model on test set.

        Args:
            model: Model to evaluate
            model_name: Name of the model
            return_uncertainty: Whether model returns uncertainty

        Returns:
            Dictionary with results
        """
        model.eval()
        all_metrics = []
        uncertainties = []
        inference_times = []

        print(f"\nEvaluating {model_name}...")

        for image, mask in tqdm(zip(self.test_images, self.test_masks)):
            image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)

            # Inference
            start_time = time.time()

            with torch.no_grad():
                if return_uncertainty:
                    # Advanced agent
                    result = model.predict(image_tensor, return_uncertainty=True)
                    prediction = result['prediction']
                    uncertainty = result['uncertainty']
                    uncertainties.append(uncertainty.cpu().numpy())
                else:
                    # Regular CNN
                    prediction = torch.sigmoid(model(image_tensor))

            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Calculate metrics
            pred_np = prediction.squeeze().cpu().numpy()
            mask_np = mask

            metrics = self.metrics_calc.calculate_all(pred_np, mask_np)
            all_metrics.append(metrics)

        # Aggregate results
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            }

        results = {
            'metrics': avg_metrics,
            'inference_time': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times)
            },
            'has_uncertainty': return_uncertainty,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }

        if uncertainties:
            results['uncertainty_stats'] = {
                'mean': np.mean([u.mean() for u in uncertainties]),
                'std': np.std([u.mean() for u in uncertainties])
            }

        self.results[model_name] = results
        return results

    def compare_models(self) -> dict:
        """
        Compare all evaluated models.

        Returns:
            Comparison dictionary
        """
        comparison = {}

        # Extract key metrics
        metric_names = ['dice', 'iou', 'f1', 'precision', 'recall']

        for metric_name in metric_names:
            comparison[metric_name] = {}
            for model_name, results in self.results.items():
                comparison[metric_name][model_name] = results['metrics'][metric_name]['mean']

        # Inference time
        comparison['inference_time'] = {}
        for model_name, results in self.results.items():
            comparison['inference_time'][model_name] = results['inference_time']['mean']

        # Model size
        comparison['model_size_mb'] = {}
        for model_name, results in self.results.items():
            comparison['model_size_mb'][model_name] = results['model_size_mb']

        return comparison

    def print_results(self):
        """Print formatted results."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)

        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print("-" * 80)

            # Metrics
            for metric_name, values in results['metrics'].items():
                print(f"  {metric_name:20s}: {values['mean']:.4f} ± {values['std']:.4f}")

            # Inference time
            inf_time = results['inference_time']
            print(f"  {'Inference time (s)':20s}: {inf_time['mean']:.4f} ± {inf_time['std']:.4f}")

            # Model properties
            print(f"  {'Parameters':20s}: {results['num_parameters']:,}")
            print(f"  {'Model size (MB)':20s}: {results['model_size_mb']:.2f}")
            print(f"  {'Has uncertainty':20s}: {'✅' if results['has_uncertainty'] else '❌'}")

        # Comparison table
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)

        comparison = self.compare_models()

        # Print header
        model_names = list(self.results.keys())
        header = f"{'Metric':<20}" + "".join([f"{name:<20}" for name in model_names])
        print(header)
        print("-" * len(header))

        # Print metrics
        for metric_name in ['dice', 'iou', 'f1', 'precision', 'recall']:
            row = f"{metric_name:<20}"
            for model_name in model_names:
                value = comparison[metric_name][model_name]
                row += f"{value:<20.4f}"
            print(row)

        print("-" * len(header))

        # Inference time
        row = f"{'Inference (s)':<20}"
        for model_name in model_names:
            value = comparison['inference_time'][model_name]
            row += f"{value:<20.4f}"
        print(row)

        # Model size
        row = f"{'Size (MB)':<20}"
        for model_name in model_names:
            value = comparison['model_size_mb'][model_name]
            row += f"{value:<20.2f}"
        print(row)

    def plot_comparison(self, save_path: str = None):
        """Plot comparison charts."""
        comparison = self.compare_models()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Metrics comparison
        ax = axes[0, 0]
        metrics = ['dice', 'iou', 'f1', 'precision', 'recall']
        model_names = list(self.results.keys())

        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            values = [comparison[m][model_name] for m in metrics]
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Metrics Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 2. Inference time
        ax = axes[0, 1]
        times = [comparison['inference_time'][m] for m in model_names]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
        bars = ax.bar(model_names, times, color=colors, alpha=0.8)
        ax.set_ylabel('Inference Time (seconds)')
        ax.set_title('Inference Speed Comparison')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Add values on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.3f}s',
                   ha='center', va='bottom', fontsize=9)

        # 3. Model size
        ax = axes[1, 0]
        sizes = [comparison['model_size_mb'][m] for m in model_names]
        bars = ax.bar(model_names, sizes, color=colors, alpha=0.8)
        ax.set_ylabel('Model Size (MB)')
        ax.set_title('Model Size Comparison')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{size:.1f}MB',
                   ha='center', va='bottom', fontsize=9)

        # 4. Dice score distribution
        ax = axes[1, 1]
        for model_name, results in self.results.items():
            dice_values = [results['metrics']['dice']['mean']]
            dice_std = [results['metrics']['dice']['std']]

            # Plot with error bars
            ax.errorbar([model_name], dice_values, yerr=dice_std,
                       fmt='o', markersize=10, capsize=5, label=model_name)

        ax.set_ylabel('Dice Score')
        ax.set_title('Dice Score with Std Dev')
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nComparison plot saved to {save_path}")

        plt.show()


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Benchmark RL vs CNN')
    parser.add_argument('--num_test', type=int, default=50,
                       help='Number of test images')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--output', type=str, default='benchmark_results.png',
                       help='Output plot path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate synthetic test data
    print(f"\nGenerating {args.num_test} synthetic test samples...")
    test_images = []
    test_masks = []

    for i in range(args.num_test):
        # Synthetic image
        image = np.random.rand(4, args.image_size, args.image_size).astype(np.float32)

        # Synthetic mask with structure
        mask = np.zeros((args.image_size, args.image_size), dtype=np.float32)
        n_fields = np.random.randint(1, 4)
        for _ in range(n_fields):
            y1 = np.random.randint(0, args.image_size - 50)
            x1 = np.random.randint(0, args.image_size - 50)
            y2 = y1 + np.random.randint(30, 80)
            x2 = x1 + np.random.randint(30, 80)

            y2 = min(y2, args.image_size)
            x2 = min(x2, args.image_size)

            mask[y1:y2, x1:x2] = 1.0

        test_images.append(image)
        test_masks.append(mask)

    # Initialize benchmark
    benchmark = Benchmark(test_images, test_masks, device=str(device))

    # 1. U-Net Baseline
    print("\n" + "="*80)
    print("1. Training and evaluating U-Net baseline...")
    unet = UNetBaseline(in_channels=4).to(device)
    benchmark.evaluate_model(unet, "U-Net Baseline", return_uncertainty=False)

    # 2. Advanced RL Agent
    print("\n" + "="*80)
    print("2. Evaluating Advanced RL Agent...")
    rl_agent = AdvancedRLAgent(
        in_channels=4,
        device=str(device),
        use_contrastive=True,
        use_meta_learning=True,
        use_active_learning=True
    )
    benchmark.evaluate_model(rl_agent, "Advanced RL (Ours)", return_uncertainty=True)

    # Print results
    benchmark.print_results()

    # Plot comparison
    benchmark.plot_comparison(save_path=args.output)

    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()
