"""
Example notebook demonstrating RL-based crop segmentation pipeline.
Convert this to .ipynb using: jupytext --to notebook example_usage.py
"""

# %% [markdown]
# # RL-based Crop Segmentation - Example Usage
#
# This notebook demonstrates how to use the RL-based crop segmentation pipeline
# to segment agricultural fields from satellite imagery.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import GeoDataLoader, DatasetSplitter
from src.environment import CropSegmentationEnv
from src.agent import PPOAgent
from src.rewards import MetricsCalculator
from utils.visualization import plot_segmentation_comparison
from utils.helpers import set_seed, get_device

# Set random seed for reproducibility
set_seed(42)

# Get device
device = get_device()

# %% [markdown]
# ## 2. Load and Explore Data

# %%
# Initialize data loader
loader = GeoDataLoader(
    image_path="../data/raw/satellite_image.tif",
    shapefile_path="../data/raw/crop_boundaries.shp",
    patch_size=256,
    stride=128,
    normalize=True
)

# Load data
image, mask = loader.load_all()

print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Number of bands: {image.shape[0]}")

# %% [markdown]
# ### Visualize original data

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Show RGB composite (first 3 bands)
if image.shape[0] >= 3:
    rgb = np.transpose(image[:3], (1, 2, 0))
    # Normalize for visualization
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    axes[0].imshow(rgb)
else:
    axes[0].imshow(image[0], cmap='gray')

axes[0].set_title('Satellite Image')
axes[0].axis('off')

# Show ground truth mask
axes[1].imshow(mask, cmap='RdYlGn')
axes[1].set_title('Ground Truth Crop Boundaries')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Get dataset statistics

# %%
stats = loader.get_statistics()

print("\nDataset Statistics:")
print(f"  Image size: {stats['image_shape']}")
print(f"  Number of bands: {stats['num_bands']}")
print(f"  Crop pixels: {stats['crop_pixels']:,}")
print(f"  Total pixels: {stats['total_pixels']:,}")
print(f"  Crop percentage: {stats['crop_percentage']:.2f}%")
print(f"  CRS: {stats['crs']}")

# %% [markdown]
# ## 3. Extract and Split Patches

# %%
# Extract patches
patches = loader.extract_patches()
print(f"Total patches: {len(patches)}")
print(f"Patches with crops: {sum(p['has_crop'] for p in patches)}")

# Split into train/val/test
train_patches, val_patches, test_patches = DatasetSplitter.spatial_split(
    patches,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

print(f"\nDataset split:")
print(f"  Train: {len(train_patches)} patches")
print(f"  Validation: {len(val_patches)} patches")
print(f"  Test: {len(test_patches)} patches")

# %% [markdown]
# ### Visualize sample patches

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for idx in range(3):
    patch = train_patches[idx]

    # Image
    if patch['image'].shape[0] >= 3:
        img_vis = np.transpose(patch['image'][:3], (1, 2, 0))
    else:
        img_vis = patch['image'][0]

    axes[idx, 0].imshow(img_vis)
    axes[idx, 0].set_title(f'Patch {idx} - Image')
    axes[idx, 0].axis('off')

    # Mask
    axes[idx, 1].imshow(patch['mask'], cmap='gray')
    axes[idx, 1].set_title(f'Patch {idx} - Mask')
    axes[idx, 1].axis('off')

    # Statistics
    crop_pct = np.sum(patch['mask']) / patch['mask'].size * 100
    axes[idx, 2].text(0.5, 0.5,
                     f"Has crop: {patch['has_crop']}\n"
                     f"Crop %: {crop_pct:.1f}%\n"
                     f"Position: {patch['position']}",
                     ha='center', va='center',
                     fontsize=10)
    axes[idx, 2].axis('off')

    # Overlay
    overlay = np.zeros((*patch['mask'].shape, 3))
    if img_vis.max() > 1.0:
        img_gray = img_vis / img_vis.max()
    else:
        img_gray = img_vis

    if len(img_gray.shape) == 3:
        overlay = img_gray.copy()
    else:
        overlay = np.stack([img_gray] * 3, axis=-1)

    overlay[patch['mask'] > 0.5, 1] = 1.0  # Green for crops

    axes[idx, 3].imshow(overlay)
    axes[idx, 3].set_title(f'Patch {idx} - Overlay')
    axes[idx, 3].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Create RL Environment

# %%
# Extract images and masks
train_images = [p['image'] for p in train_patches[:50]]  # Use subset for demo
train_masks = [p['mask'] for p in train_patches[:50]]

# Create environment
env = CropSegmentationEnv(
    image_patches=train_images,
    mask_patches=train_masks,
    patch_size=256,
    action_mode='patch',
    reward_type='dice',
    num_actions=16  # 4x4 grid
)

print(f"Environment created!")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# %% [markdown]
# ### Test environment

# %%
obs, info = env.reset()
print(f"Initial observation shape: {obs['image'].shape}")
print(f"Initial info: {info}")

# Take a random action
action = env.action_space.sample()
next_obs, reward, terminated, truncated, next_info = env.step(action)

print(f"\nAfter random action {action}:")
print(f"  Reward: {reward:.4f}")
print(f"  Current Dice: {next_info['current_dice']:.4f}")
print(f"  Current IoU: {next_info['current_iou']:.4f}")

# %% [markdown]
# ## 5. Initialize RL Agent

# %%
agent = PPOAgent(
    in_channels=4,  # Image channels + predicted mask
    num_actions=16,
    lr=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    device=str(device)
)

print("PPO Agent initialized!")
print(f"Number of parameters: {sum(p.numel() for p in agent.network.parameters()):,}")

# %% [markdown]
# ## 6. Training Loop (Demo)

# %%
# Short training demo
n_episodes = 5
metrics_calc = MetricsCalculator()

episode_rewards = []
episode_dices = []

for episode in range(n_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Select action
        action, log_prob, value = agent.select_action(obs)

        # Take step
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

    # Update agent
    if len(agent.buffer) > 0:
        update_stats = agent.update(n_epochs=2, batch_size=32)

    # Get final metrics
    final_dice = info['current_dice']

    episode_rewards.append(episode_reward)
    episode_dices.append(final_dice)

    print(f"Episode {episode+1}/{n_episodes}: "
          f"Reward={episode_reward:.4f}, "
          f"Dice={final_dice:.4f}")

# %% [markdown]
# ### Plot training progress

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(episode_rewards, marker='o')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[0].set_title('Training Rewards')
axes[0].grid(True, alpha=0.3)

axes[1].plot(episode_dices, marker='o', color='green')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Dice Score')
axes[1].set_title('Segmentation Quality')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Evaluate Agent

# %%
# Evaluate on test set
val_images = [p['image'] for p in val_patches[:10]]
val_masks = [p['mask'] for p in val_patches[:10]]

val_env = CropSegmentationEnv(
    image_patches=val_images,
    mask_patches=val_masks,
    patch_size=256,
    action_mode='patch',
    reward_type='dice',
    num_actions=16
)

# Run evaluation
val_metrics = []

for i in range(5):
    obs, _ = val_env.reset()
    done = False

    while not done:
        action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated

    # Calculate metrics
    metrics = metrics_calc.calculate_all(
        val_env.predicted_mask,
        val_env.current_gt_mask
    )
    val_metrics.append(metrics)

    print(f"Validation {i+1}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")

# Average metrics
avg_metrics = {}
for key in val_metrics[0].keys():
    avg_metrics[key] = np.mean([m[key] for m in val_metrics])

print(f"\nAverage Validation Metrics:")
for key, value in avg_metrics.items():
    print(f"  {key}: {value:.4f}")

# %% [markdown]
# ### Visualize predictions

# %%
# Visualize a few predictions
n_viz = 3

fig, axes = plt.subplots(n_viz, 4, figsize=(16, 4*n_viz))

for idx in range(n_viz):
    # Reset environment
    obs, _ = val_env.reset()
    done = False

    # Run episode
    while not done:
        action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated

    # Get results
    image = val_env.current_image
    gt_mask = val_env.current_gt_mask
    pred_mask = val_env.predicted_mask

    # Calculate metrics
    metrics = metrics_calc.calculate_all(pred_mask, gt_mask)

    # Visualize
    if image.shape[0] >= 3:
        img_vis = np.transpose(image[:3], (1, 2, 0))
    else:
        img_vis = image[0]

    axes[idx, 0].imshow(img_vis)
    axes[idx, 0].set_title('Input')
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(gt_mask, cmap='gray')
    axes[idx, 1].set_title('Ground Truth')
    axes[idx, 1].axis('off')

    axes[idx, 2].imshow(pred_mask, cmap='gray')
    axes[idx, 2].set_title(f'Prediction\nDice: {metrics["dice"]:.3f}')
    axes[idx, 2].axis('off')

    # Overlay
    overlay = np.zeros((*gt_mask.shape, 3))
    overlay[gt_mask > 0.5] = [0, 1, 0]
    overlay[pred_mask > 0.5] = [1, 0, 0]
    overlay[(gt_mask > 0.5) & (pred_mask > 0.5)] = [1, 1, 0]

    axes[idx, 3].imshow(overlay)
    axes[idx, 3].set_title(f'Overlay\nIoU: {metrics["iou"]:.3f}')
    axes[idx, 3].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Save Model

# %%
# Save trained agent
save_path = Path('../experiments/demo/checkpoints')
save_path.mkdir(parents=True, exist_ok=True)

agent.save(str(save_path / 'demo_model.pt'))
print(f"Model saved to {save_path / 'demo_model.pt'}")

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated:
# 1. Loading geospatial data (GeoTIFF + Shapefile)
# 2. Creating patches and splitting dataset
# 3. Setting up RL environment for segmentation
# 4. Training PPO agent
# 5. Evaluating and visualizing results
#
# For full training, use the `train.py` script with appropriate configuration!
