"""
Helper functions for RL segmentation.
"""

import os
import random
import numpy as np
import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.

    Args:
        device: Device string ('cuda', 'cpu', or None for auto)

    Returns:
        torch.device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        save_path: Path to save file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension
    if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif save_path.suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {save_path.suffix}")

    logger.info(f"Configuration saved to {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load based on extension
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    logger.info(f"Configuration loaded from {config_path}")
    return config


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """
    Print model summary.

    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model)
    logger.info(f"\nModel Summary:")
    logger.info(f"Total trainable parameters: {total_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1e6:.2f} MB (float32)")

    # Print layer-wise info
    logger.info("\nLayers:")
    for name, module in model.named_children():
        params = count_parameters(module)
        logger.info(f"  {name}: {params:,} parameters")


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """
    Create experiment directory with timestamp.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name

    Returns:
        Path to experiment directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if experiment_name:
        exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    else:
        exp_dir = Path(base_dir) / timestamp

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'visualizations').mkdir(exist_ok=True)

    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current validation score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True

        return False


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name: str = ''):
        """
        Initialize meter.

        Args:
            name: Name of the meter
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update meter.

        Args:
            val: Value to add
            n: Number of items
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation."""
        return f"{self.name}: {self.avg:.4f}"


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    # Test helpers
    set_seed(42)
    device = get_device()

    # Test config save/load
    test_config = {
        'param1': 1.0,
        'param2': 'value',
        'nested': {'a': 1, 'b': 2}
    }
    save_config(test_config, 'test_config.yaml')
    loaded_config = load_config('test_config.yaml')
    print(loaded_config)

    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        should_stop = early_stop(score)
        print(f"Epoch {i}, Score: {score}, Stop: {should_stop}")

    # Test average meter
    meter = AverageMeter('loss')
    for val in [1.0, 0.8, 0.6, 0.7]:
        meter.update(val)
        print(meter)
