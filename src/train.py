"""
Training pipeline for RL-based crop segmentation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
import yaml
import wandb
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import GeoDataLoader, DatasetSplitter
from src.environment import CropSegmentationEnv, SequentialSegmentationEnv
from src.agent import PPOAgent
from src.rewards import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for RL-based crop segmentation.
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'rl_segmentation'),
                config=config,
                name=config.get('experiment_name', 'rl_crop_seg')
            )

        # Load data
        logger.info("Loading data...")
        self.load_data()

        # Create environments
        logger.info("Creating environments...")
        self.create_environments()

        # Initialize agent
        logger.info("Initializing agent...")
        self.agent = PPOAgent(
            in_channels=config['in_channels'],
            num_actions=config['num_actions'],
            lr=config['learning_rate'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_epsilon=config['clip_epsilon'],
            value_coef=config['value_coef'],
            entropy_coef=config['entropy_coef'],
            device=str(self.device)
        )

        # Metrics calculator
        self.metrics_calc = MetricsCalculator()

        # Training statistics
        self.episode_rewards = []
        self.episode_metrics = []

        logger.info(f"Trainer initialized. Device: {self.device}")

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
        image, mask = loader.load_all()

        # Extract patches
        patches = loader.extract_patches()
        logger.info(f"Extracted {len(patches)} patches")

        # Split into train/val/test
        self.train_patches, self.val_patches, self.test_patches = DatasetSplitter.spatial_split(
            patches,
            train_ratio=config.get('train_ratio', 0.7),
            val_ratio=config.get('val_ratio', 0.15),
            test_ratio=config.get('test_ratio', 0.15)
        )

        logger.info(f"Train: {len(self.train_patches)}, "
                   f"Val: {len(self.val_patches)}, "
                   f"Test: {len(self.test_patches)}")

    def create_environments(self):
        """Create training and validation environments."""
        config = self.config

        # Extract images and masks
        train_images = [p['image'] for p in self.train_patches]
        train_masks = [p['mask'] for p in self.train_patches]
        val_images = [p['image'] for p in self.val_patches]
        val_masks = [p['mask'] for p in self.val_patches]

        # Create environments
        env_class = SequentialSegmentationEnv if config.get('sequential', False) else CropSegmentationEnv

        self.train_env = env_class(
            image_patches=train_images,
            mask_patches=train_masks,
            patch_size=config['patch_size'],
            action_mode=config['action_mode'],
            reward_type=config['reward_type'],
            num_actions=config['num_actions']
        )

        self.val_env = env_class(
            image_patches=val_images,
            mask_patches=val_masks,
            patch_size=config['patch_size'],
            action_mode=config['action_mode'],
            reward_type=config['reward_type'],
            num_actions=config['num_actions']
        )

    def run_episode(self, env, training: bool = True) -> Dict:
        """
        Run single episode.

        Args:
            env: Environment to run
            training: Whether in training mode

        Returns:
            Episode statistics
        """
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        done = False
        while not done:
            # Select action
            action, log_prob, value = self.agent.select_action(obs)

            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Store transition in buffer (only during training)
            if training:
                self.agent.buffer.add(
                    obs=obs['image'],
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done
                )

            episode_reward += reward
            episode_length += 1
            obs = next_obs

        # Calculate final metrics
        metrics = self.metrics_calc.calculate_all(
            env.predicted_mask,
            env.current_gt_mask
        )

        stats = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            **metrics
        }

        return stats

    def train_epoch(self, epoch: int) -> Dict:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Training statistics
        """
        config = self.config
        n_episodes = config.get('episodes_per_epoch', 10)

        epoch_stats = []

        # Collect rollouts
        for episode in tqdm(range(n_episodes), desc=f"Epoch {epoch}"):
            stats = self.run_episode(self.train_env, training=True)
            epoch_stats.append(stats)

            # Update policy after each episode
            if len(self.agent.buffer) > 0:
                update_stats = self.agent.update(
                    n_epochs=config.get('ppo_epochs', 4),
                    batch_size=config.get('batch_size', 64)
                )

                # Add update stats
                for key, value in update_stats.items():
                    stats[f'agent/{key}'] = value

            self.episode_rewards.append(stats['episode_reward'])
            self.episode_metrics.append(stats)

        # Average epoch statistics
        avg_stats = {}
        for key in epoch_stats[0].keys():
            avg_stats[f'train/{key}'] = np.mean([s[key] for s in epoch_stats])

        return avg_stats

    def validate(self, epoch: int) -> Dict:
        """
        Validate agent.

        Args:
            epoch: Current epoch number

        Returns:
            Validation statistics
        """
        n_episodes = self.config.get('val_episodes', 5)

        val_stats = []
        for episode in range(n_episodes):
            stats = self.run_episode(self.val_env, training=False)
            val_stats.append(stats)

        # Average validation statistics
        avg_stats = {}
        for key in val_stats[0].keys():
            avg_stats[f'val/{key}'] = np.mean([s[key] for s in val_stats])

        return avg_stats

    def train(self):
        """Run full training loop."""
        config = self.config
        n_epochs = config['num_epochs']

        best_val_dice = 0.0

        for epoch in range(n_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"{'='*50}")

            # Training
            train_stats = self.train_epoch(epoch)

            # Validation
            val_stats = self.validate(epoch)

            # Combine stats
            all_stats = {**train_stats, **val_stats}

            # Log statistics
            logger.info(f"\nTraining Stats:")
            logger.info(f"  Reward: {train_stats['train/episode_reward']:.4f}")
            logger.info(f"  Dice: {train_stats['train/dice']:.4f}")
            logger.info(f"  IoU: {train_stats['train/iou']:.4f}")

            logger.info(f"\nValidation Stats:")
            logger.info(f"  Reward: {val_stats['val/episode_reward']:.4f}")
            logger.info(f"  Dice: {val_stats['val/dice']:.4f}")
            logger.info(f"  IoU: {val_stats['val/iou']:.4f}")

            # Log to wandb
            if config.get('use_wandb', False):
                wandb.log(all_stats, step=epoch)

            # Save checkpoint
            if (epoch + 1) % config.get('save_interval', 10) == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.agent.save(str(checkpoint_path))
                logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Save best model
            if val_stats['val/dice'] > best_val_dice:
                best_val_dice = val_stats['val/dice']
                best_model_path = self.checkpoint_dir / 'best_model.pt'
                self.agent.save(str(best_model_path))
                logger.info(f"Best model saved: {best_model_path} (Dice: {best_val_dice:.4f})")

        # Save final model
        final_model_path = self.checkpoint_dir / 'final_model.pt'
        self.agent.save(str(final_model_path))
        logger.info(f"Training complete! Final model saved: {final_model_path}")

        # Close wandb
        if config.get('use_wandb', False):
            wandb.finish()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RL agent for crop segmentation')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image', type=str, help='Path to GeoTIFF image')
    parser.add_argument('--shapefile', type=str, help='Path to shapefile')
    parser.add_argument('--output', type=str, default='experiments/default',
                       help='Output directory')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {args.config} not found. Using default config.")
        config = get_default_config()

    # Override with command line arguments
    if args.image:
        config['image_path'] = args.image
    if args.shapefile:
        config['shapefile_path'] = args.shapefile
    if args.output:
        config['output_dir'] = args.output

    # Create timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['experiment_name'] = f"rl_crop_seg_{timestamp}"

    # Initialize trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()


def get_default_config() -> Dict:
    """Get default configuration."""
    return {
        # Data
        'image_path': 'data/raw/satellite_image.tif',
        'shapefile_path': 'data/raw/crop_boundaries.shp',
        'patch_size': 256,
        'stride': 128,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,

        # Environment
        'action_mode': 'patch',  # 'patch' or 'pixel'
        'reward_type': 'dice',  # 'dice', 'iou', 'pixel'
        'num_actions': 16,  # 4x4 grid for patch mode
        'sequential': False,

        # Model
        'in_channels': 4,  # RGB + predicted mask

        # Training
        'num_epochs': 100,
        'episodes_per_epoch': 10,
        'val_episodes': 5,

        # PPO hyperparameters
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 4,
        'batch_size': 64,

        # Logging
        'save_interval': 10,
        'use_wandb': False,
        'wandb_project': 'rl_crop_segmentation',

        # Output
        'output_dir': 'experiments/default'
    }


if __name__ == "__main__":
    main()
