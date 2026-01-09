"""
Reinforcement Learning Environment for Crop Segmentation.
The agent learns to segment crop fields from satellite imagery.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Tuple, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CropSegmentationEnv(gym.Env):
    """
    Gymnasium environment for crop field segmentation using RL.

    The agent receives an image patch and must predict a segmentation mask.
    Actions represent pixel-wise or patch-wise segmentation decisions.
    """

    metadata = {'render_modes': ['rgb_array', 'human']}

    def __init__(self,
                 image_patches: list,
                 mask_patches: list,
                 patch_size: int = 256,
                 action_mode: str = 'patch',  # 'pixel' or 'patch'
                 reward_type: str = 'dice',  # 'dice', 'iou', 'pixel'
                 num_actions: int = 16,  # For patch mode: sqrt of this = grid size
                 render_mode: Optional[str] = None):
        """
        Initialize environment.

        Args:
            image_patches: List of image patches (C, H, W)
            mask_patches: List of ground truth masks (H, W)
            patch_size: Size of image patches
            action_mode: 'pixel' for pixel-wise, 'patch' for sub-patch decisions
            reward_type: Type of reward calculation
            num_actions: Number of discrete actions (sub-patches)
            render_mode: Rendering mode
        """
        super().__init__()

        self.image_patches = image_patches
        self.mask_patches = mask_patches
        self.patch_size = patch_size
        self.action_mode = action_mode
        self.reward_type = reward_type
        self.num_actions = num_actions
        self.render_mode = render_mode

        # Validate data
        assert len(image_patches) == len(mask_patches), "Mismatch in patches"
        assert len(image_patches) > 0, "No patches provided"

        # Get image dimensions
        self.num_channels = image_patches[0].shape[0]
        self.height = image_patches[0].shape[1]
        self.width = image_patches[0].shape[2]

        # Current episode state
        self.current_patch_idx = 0
        self.current_image = None
        self.current_gt_mask = None
        self.predicted_mask = None
        self.step_count = 0

        # For patch mode: divide image into grid
        if action_mode == 'patch':
            self.grid_size = int(np.sqrt(num_actions))
            assert self.grid_size ** 2 == num_actions, "num_actions must be perfect square"
            self.sub_patch_h = self.height // self.grid_size
            self.sub_patch_w = self.width // self.grid_size
            self.max_steps = num_actions
        else:  # pixel mode
            self.max_steps = self.height * self.width

        # Define action and observation spaces
        self._define_spaces()

        logger.info(f"Environment initialized: {len(image_patches)} patches, "
                   f"mode={action_mode}, reward={reward_type}")

    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation: image + current predicted mask
        obs_channels = self.num_channels + 1  # Image + mask

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_channels, self.height, self.width),
                dtype=np.float32
            ),
            'step': spaces.Box(
                low=0,
                high=self.max_steps,
                shape=(1,),
                dtype=np.int32
            )
        })

        # Action: discrete actions for sub-patches or pixels
        if self.action_mode == 'patch':
            # Each action selects a sub-patch and classifies it (crop/no-crop)
            self.action_space = spaces.Discrete(self.num_actions * 2)
        else:
            # Each action selects a pixel and sets its value
            self.action_space = spaces.Discrete(self.height * self.width * 2)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Select random patch
        self.current_patch_idx = self.np_random.integers(0, len(self.image_patches))
        self.current_image = self.image_patches[self.current_patch_idx].copy()
        self.current_gt_mask = self.mask_patches[self.current_patch_idx].copy()

        # Initialize predicted mask (all zeros)
        self.predicted_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.step_count = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action to execute

        Returns:
            observation: New observation
            reward: Reward received
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Apply action to predicted mask
        self._apply_action(action)
        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        terminated = self.step_count >= self.max_steps
        truncated = False

        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: int):
        """
        Apply action to predicted mask.

        Args:
            action: Action index
        """
        if self.action_mode == 'patch':
            # Decode action: which sub-patch and what value
            patch_idx = action // 2
            value = action % 2

            # Calculate sub-patch position
            row = patch_idx // self.grid_size
            col = patch_idx % self.grid_size

            y_start = row * self.sub_patch_h
            y_end = min((row + 1) * self.sub_patch_h, self.height)
            x_start = col * self.sub_patch_w
            x_end = min((col + 1) * self.sub_patch_w, self.width)

            # Set sub-patch value
            self.predicted_mask[y_start:y_end, x_start:x_end] = float(value)

        else:  # pixel mode
            # Decode action: which pixel and what value
            pixel_idx = action // 2
            value = action % 2

            y = pixel_idx // self.width
            x = pixel_idx % self.width

            self.predicted_mask[y, x] = float(value)

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on segmentation quality.

        Returns:
            Reward value
        """
        if self.reward_type == 'dice':
            return self._dice_reward()
        elif self.reward_type == 'iou':
            return self._iou_reward()
        elif self.reward_type == 'pixel':
            return self._pixel_accuracy_reward()
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _dice_reward(self) -> float:
        """Calculate Dice coefficient as reward."""
        pred = self.predicted_mask > 0.5
        gt = self.current_gt_mask > 0.5

        intersection = np.sum(pred & gt)
        union = np.sum(pred) + np.sum(gt)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        dice = 2.0 * intersection / (union + 1e-8)
        return dice

    def _iou_reward(self) -> float:
        """Calculate IoU as reward."""
        pred = self.predicted_mask > 0.5
        gt = self.current_gt_mask > 0.5

        intersection = np.sum(pred & gt)
        union = np.sum(pred | gt)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        iou = intersection / (union + 1e-8)
        return iou

    def _pixel_accuracy_reward(self) -> float:
        """Calculate pixel accuracy as reward."""
        pred = self.predicted_mask > 0.5
        gt = self.current_gt_mask > 0.5

        correct = np.sum(pred == gt)
        total = pred.size

        accuracy = correct / total
        return accuracy

    def _get_observation(self) -> Dict:
        """
        Get current observation.

        Returns:
            Observation dictionary
        """
        # Concatenate image and predicted mask
        obs_image = np.concatenate([
            self.current_image,
            self.predicted_mask[np.newaxis, :, :]
        ], axis=0).astype(np.float32)

        return {
            'image': obs_image,
            'step': np.array([self.step_count], dtype=np.int32)
        }

    def _get_info(self) -> Dict:
        """
        Get additional information.

        Returns:
            Info dictionary
        """
        return {
            'patch_idx': self.current_patch_idx,
            'step_count': self.step_count,
            'current_dice': self._dice_reward(),
            'current_iou': self._iou_reward(),
            'current_accuracy': self._pixel_accuracy_reward()
        }

    def render(self):
        """Render environment state."""
        if self.render_mode == 'rgb_array':
            # Return RGB visualization
            return self._render_frame()
        elif self.render_mode == 'human':
            # Display visualization
            import matplotlib.pyplot as plt
            frame = self._render_frame()
            plt.imshow(frame)
            plt.axis('off')
            plt.show()

    def _render_frame(self) -> np.ndarray:
        """
        Create visualization frame.

        Returns:
            RGB image array
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Show RGB image (first 3 channels or grayscale)
        if self.num_channels >= 3:
            img_vis = np.transpose(self.current_image[:3], (1, 2, 0))
        else:
            img_vis = self.current_image[0]

        axes[0].imshow(img_vis)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Show ground truth
        axes[1].imshow(self.current_gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # Show prediction
        axes[2].imshow(self.predicted_mask, cmap='gray')
        axes[2].set_title(f'Prediction (Step {self.step_count})')
        axes[2].axis('off')

        # Convert to array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return frame


class SequentialSegmentationEnv(CropSegmentationEnv):
    """
    Sequential segmentation environment where agent processes image sequentially.
    Agent moves through image in raster scan order.
    """

    def __init__(self, *args, **kwargs):
        """Initialize sequential environment."""
        super().__init__(*args, **kwargs)
        self.current_position = (0, 0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed, options)
        self.current_position = (0, 0)
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action and move to next position.

        Args:
            action: Binary action (0 or 1) for current position
        """
        # Set current pixel/patch
        y, x = self.current_position

        if self.action_mode == 'patch':
            y_end = min(y + self.sub_patch_h, self.height)
            x_end = min(x + self.sub_patch_w, self.width)
            self.predicted_mask[y:y_end, x:x_end] = float(action)

            # Move to next patch
            x += self.sub_patch_w
            if x >= self.width:
                x = 0
                y += self.sub_patch_h
        else:
            self.predicted_mask[y, x] = float(action)

            # Move to next pixel
            x += 1
            if x >= self.width:
                x = 0
                y += 1

        self.current_position = (y, x)
        self.step_count += 1

        # Calculate reward (sparse - only at the end)
        if self.step_count >= self.max_steps:
            reward = self._calculate_reward()
        else:
            reward = 0.0

        terminated = self.step_count >= self.max_steps
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
