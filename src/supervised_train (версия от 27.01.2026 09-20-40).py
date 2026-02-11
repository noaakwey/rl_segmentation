"""
Supervised training pipeline for U-Net segmentation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import GeoDataLoader, DatasetSplitter
from src.models import UNet
from src.rewards import MetricsCalculator


class PatchDataset(Dataset):
    def __init__(self, patches: List[Dict]):
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        image = torch.from_numpy(patch['image']).float()
        mask = torch.from_numpy(patch['mask']).float().unsqueeze(0)
        return image, mask


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def build_sampler(patches: List[Dict], min_crop_fraction: float = 0.01) -> WeightedRandomSampler:
    weights = []
    for p in patches:
        is_pos = p.get('crop_fraction', 0.0) >= min_crop_fraction
        weights.append(1.0 if is_pos else 0.25)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_supervised(config: Dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    loader = GeoDataLoader(
        image_path=config['image_path'],
        shapefile_path=config['shapefile_path'],
        patch_size=config['patch_size'],
        stride=config['stride'],
        normalize=True
    )

    image, _ = loader.load_all()
    patches = loader.extract_patches()

    train_patches, val_patches, _ = DatasetSplitter.spatial_split(
        patches,
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15)
    )

    train_ds = PatchDataset(train_patches)
    val_ds = PatchDataset(val_patches)

    sup_cfg = config.get('supervised', {})
    batch_size = sup_cfg.get('batch_size', 8)
    num_epochs = sup_cfg.get('num_epochs', 50)
    lr = sup_cfg.get('learning_rate', 3e-4)
    weight_decay = sup_cfg.get('weight_decay', 1e-4)
    bce_weight = sup_cfg.get('bce_weight', 0.5)
    dice_weight = sup_cfg.get('dice_weight', 0.5)
    early_stopping_patience = int(sup_cfg.get('early_stopping_patience', 10))
    early_stopping_min_delta = float(sup_cfg.get('early_stopping_min_delta', 1e-3))

    if sup_cfg.get('balance', True):
        sampler = build_sampler(train_patches, min_crop_fraction=sup_cfg.get('min_crop_fraction', 0.01))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    in_channels = image.shape[0]
    model = UNet(in_channels=in_channels, base_channels=sup_cfg.get('base_channels', 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=float(sup_cfg.get('lr_reduce_factor', 0.5)),
        patience=int(sup_cfg.get('lr_reduce_patience', 5)),
        min_lr=float(sup_cfg.get('min_learning_rate', 1e-6))
    )
    bce = nn.BCEWithLogitsLoss()
    metrics = MetricsCalculator()

    # Output directories
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    best_dice = -1.0
    best_path = ckpt_dir / 'best_unet.pt'
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = bce_weight * bce(logits, masks) + dice_weight * dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_metrics = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                gts = masks.cpu().numpy()
                for p, g in zip(preds, gts):
                    val_metrics.append(metrics.calculate_all(p[0], g[0]))

        val_dice = float(np.mean([m['dice'] for m in val_metrics])) if val_metrics else 0.0
        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_dice={val_dice:.4f}")
        scheduler.step(val_dice)

        if val_dice > best_dice + early_stopping_min_delta:
            best_dice = val_dice
            torch.save({'model_state_dict': model.state_dict()}, best_path)
            print(f"Saved best model: {best_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch+1} (best_val_dice={best_dice:.4f}, "
                f"patience={early_stopping_patience})"
            )
            break


def main():
    parser = argparse.ArgumentParser(description='Train supervised U-Net for segmentation')
    parser.add_argument('--config', type=str, default='configs/custom_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file {args.config} not found.")

    train_supervised(config)


if __name__ == "__main__":
    main()
