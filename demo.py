#!/usr/bin/env python3
"""
Quick demo script to showcase RL-based crop segmentation pipeline.
This is a minimal example showing the core functionality.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("RL-Based Crop Segmentation - Quick Demo")
print("=" * 70)
print()

# Set seeds for reproducibility
from utils.helpers import set_seed, get_device
set_seed(42)
device = get_device()

print("\n[1/6] Generating synthetic data for demo...")
print("-" * 70)

# Generate synthetic data (since we may not have real data)
def generate_synthetic_data(size=256, num_patches=10):
    """Generate synthetic satellite images and crop masks."""
    images = []
    masks = []

    for i in range(num_patches):
        # Generate random multi-channel image
        image = np.random.rand(3, size, size).astype(np.float32)

        # Generate random crop mask with some structure
        mask = np.zeros((size, size), dtype=np.float32)

        # Add some rectangular "fields"
        n_fields = np.random.randint(1, 5)
        for _ in range(n_fields):
            x1 = np.random.randint(0, size - 50)
            y1 = np.random.randint(0, size - 50)
            x2 = x1 + np.random.randint(30, 100)
            y2 = y1 + np.random.randint(30, 100)

            x2 = min(x2, size)
            y2 = min(y2, size)

            mask[y1:y2, x1:x2] = 1.0

        images.append(image)
        masks.append(mask)

    return images, masks

# Generate data
train_images, train_masks = generate_synthetic_data(size=128, num_patches=20)
val_images, val_masks = generate_synthetic_data(size=128, num_patches=5)

print(f"✓ Generated {len(train_images)} training patches")
print(f"✓ Generated {len(val_images)} validation patches")
print(f"  Patch size: {train_images[0].shape}")

print("\n[2/6] Creating RL environment...")
print("-" * 70)

from src.environment import CropSegmentationEnv

env = CropSegmentationEnv(
    image_patches=train_images,
    mask_patches=train_masks,
    patch_size=128,
    action_mode='patch',
    reward_type='dice',
    num_actions=16  # 4x4 grid
)

print(f"✓ Environment created")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space} ({env.action_space.n} actions)")

print("\n[3/6] Initializing PPO agent...")
print("-" * 70)

from src.agent import PPOAgent

agent = PPOAgent(
    in_channels=4,  # 3 image channels + 1 predicted mask
    num_actions=16,
    lr=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    device=str(device)
)

total_params = sum(p.numel() for p in agent.network.parameters())
print(f"✓ PPO agent initialized")
print(f"  Total parameters: {total_params:,}")
print(f"  Device: {device}")

print("\n[4/6] Training agent (quick demo - 5 episodes)...")
print("-" * 70)

from src.rewards import MetricsCalculator

metrics_calc = MetricsCalculator()
episode_rewards = []
episode_dices = []

for episode in range(5):
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    # Run episode
    while not done:
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store in buffer
        agent.buffer.add(
            obs=obs['image'],
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done
        )

        episode_reward += reward
        obs = next_obs

    # Update policy
    if len(agent.buffer) > 0:
        update_stats = agent.update(n_epochs=2, batch_size=16)

    # Track metrics
    final_dice = info['current_dice']
    episode_rewards.append(episode_reward)
    episode_dices.append(final_dice)

    print(f"  Episode {episode+1}/5: "
          f"Reward={episode_reward:.3f}, "
          f"Dice={final_dice:.3f}, "
          f"Policy Loss={update_stats.get('policy_loss', 0):.3f}")

print("\n[5/6] Evaluating on validation set...")
print("-" * 70)

val_env = CropSegmentationEnv(
    image_patches=val_images,
    mask_patches=val_masks,
    patch_size=128,
    action_mode='patch',
    reward_type='dice',
    num_actions=16
)

val_metrics = []

for i in range(len(val_images)):
    obs, _ = val_env.reset()
    done = False

    while not done:
        action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated

    metrics = metrics_calc.calculate_all(
        val_env.predicted_mask,
        val_env.current_gt_mask
    )
    val_metrics.append(metrics)

# Calculate average metrics
avg_dice = np.mean([m['dice'] for m in val_metrics])
avg_iou = np.mean([m['iou'] for m in val_metrics])
avg_pixel_acc = np.mean([m['pixel_accuracy'] for m in val_metrics])

print(f"✓ Validation complete")
print(f"  Average Dice: {avg_dice:.3f}")
print(f"  Average IoU: {avg_iou:.3f}")
print(f"  Average Pixel Accuracy: {avg_pixel_acc:.3f}")

print("\n[6/6] Saving demo model...")
print("-" * 70)

demo_dir = Path("experiments/demo/checkpoints")
demo_dir.mkdir(parents=True, exist_ok=True)
model_path = demo_dir / "demo_model.pt"

agent.save(str(model_path))
print(f"✓ Model saved to {model_path}")

print("\n" + "=" * 70)
print("Demo Complete! 🎉")
print("=" * 70)
print()
print("Summary:")
print(f"  • Training episodes: 5")
print(f"  • Final training Dice: {episode_dices[-1]:.3f}")
print(f"  • Validation Dice: {avg_dice:.3f}")
print(f"  • Model saved: {model_path}")
print()
print("Next steps:")
print("  1. Train on real data: python src/train.py --config configs/default_config.yaml")
print("  2. Evaluate model: python src/evaluate.py --checkpoint <path> --config <config>")
print("  3. Run inference: python src/inference.py --model <path> --image <image>")
print()
print("For detailed usage, see README.md")
print("=" * 70)
