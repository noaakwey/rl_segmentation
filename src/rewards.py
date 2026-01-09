"""
Reward functions and metrics for crop segmentation task.
"""

import numpy as np
import torch
from typing import Union, Tuple
from scipy.ndimage import distance_transform_edt


class SegmentationRewards:
    """
    Collection of reward functions for segmentation tasks.
    """

    @staticmethod
    def dice_coefficient(pred: np.ndarray,
                        gt: np.ndarray,
                        smooth: float = 1e-8) -> float:
        """
        Calculate Dice coefficient (F1 score for segmentation).

        Args:
            pred: Predicted mask (H, W) or (B, H, W)
            gt: Ground truth mask (H, W) or (B, H, W)
            smooth: Smoothing factor

        Returns:
            Dice coefficient [0, 1]
        """
        pred = (pred > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)

    @staticmethod
    def iou_score(pred: np.ndarray,
                 gt: np.ndarray,
                 smooth: float = 1e-8) -> float:
        """
        Calculate Intersection over Union (IoU/Jaccard index).

        Args:
            pred: Predicted mask
            gt: Ground truth mask
            smooth: Smoothing factor

        Returns:
            IoU score [0, 1]
        """
        pred = (pred > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt) - intersection

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        iou = (intersection + smooth) / (union + smooth)
        return float(iou)

    @staticmethod
    def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate pixel-wise accuracy.

        Args:
            pred: Predicted mask
            gt: Ground truth mask

        Returns:
            Accuracy [0, 1]
        """
        pred = (pred > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        correct = np.sum(pred == gt)
        total = pred.size

        return float(correct / total)

    @staticmethod
    def precision_recall_f1(pred: np.ndarray,
                           gt: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            pred: Predicted mask
            gt: Ground truth mask

        Returns:
            Tuple of (precision, recall, f1)
        """
        pred = (pred > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return float(precision), float(recall), float(f1)

    @staticmethod
    def boundary_iou(pred: np.ndarray,
                    gt: np.ndarray,
                    dilation: int = 2) -> float:
        """
        Calculate IoU for boundary pixels (emphasizes edge quality).

        Args:
            pred: Predicted mask
            gt: Ground truth mask
            dilation: Dilation radius for boundary

        Returns:
            Boundary IoU score
        """
        from scipy.ndimage import binary_dilation

        pred = (pred > 0.5).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)

        # Get boundaries
        pred_boundary = binary_dilation(pred, iterations=dilation) ^ pred
        gt_boundary = binary_dilation(gt, iterations=dilation) ^ gt

        # Calculate IoU on boundaries
        intersection = np.sum(pred_boundary & gt_boundary)
        union = np.sum(pred_boundary | gt_boundary)

        if union == 0:
            return 1.0

        return float(intersection / union)

    @staticmethod
    def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Hausdorff distance between boundaries.

        Args:
            pred: Predicted mask
            gt: Ground truth mask

        Returns:
            Hausdorff distance (lower is better)
        """
        pred = (pred > 0.5).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)

        # Get distance transforms
        pred_dt = distance_transform_edt(1 - pred)
        gt_dt = distance_transform_edt(1 - gt)

        # Hausdorff distance
        hd1 = np.max(pred_dt[gt == 1]) if np.any(gt == 1) else 0
        hd2 = np.max(gt_dt[pred == 1]) if np.any(pred == 1) else 0

        return float(max(hd1, hd2))


class AdaptiveReward:
    """
    Adaptive reward that combines multiple metrics.
    """

    def __init__(self,
                 dice_weight: float = 0.5,
                 iou_weight: float = 0.3,
                 boundary_weight: float = 0.2,
                 prev_reward: float = 0.0):
        """
        Initialize adaptive reward.

        Args:
            dice_weight: Weight for Dice coefficient
            iou_weight: Weight for IoU
            boundary_weight: Weight for boundary IoU
            prev_reward: Previous reward (for delta calculation)
        """
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight
        self.prev_reward = prev_reward

        self.rewards = SegmentationRewards()

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate combined reward.

        Args:
            pred: Predicted mask
            gt: Ground truth mask

        Returns:
            Combined reward value
        """
        dice = self.rewards.dice_coefficient(pred, gt)
        iou = self.rewards.iou_score(pred, gt)
        boundary = self.rewards.boundary_iou(pred, gt)

        reward = (self.dice_weight * dice +
                 self.iou_weight * iou +
                 self.boundary_weight * boundary)

        return reward

    def delta_reward(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate reward as improvement over previous step.

        Args:
            pred: Predicted mask
            gt: Ground truth mask

        Returns:
            Delta reward (improvement)
        """
        current_reward = self.__call__(pred, gt)
        delta = current_reward - self.prev_reward
        self.prev_reward = current_reward

        return delta


class ShapedReward:
    """
    Shaped reward that provides intermediate feedback.
    """

    def __init__(self, base_reward_fn: callable):
        """
        Initialize shaped reward.

        Args:
            base_reward_fn: Base reward function
        """
        self.base_reward_fn = base_reward_fn
        self.step_penalties = []

    def __call__(self,
                pred: np.ndarray,
                gt: np.ndarray,
                step: int,
                max_steps: int) -> float:
        """
        Calculate shaped reward with step penalty.

        Args:
            pred: Predicted mask
            gt: Ground truth mask
            step: Current step
            max_steps: Maximum steps

        Returns:
            Shaped reward
        """
        # Base reward
        base_reward = self.base_reward_fn(pred, gt)

        # Time penalty (encourage efficiency)
        time_penalty = -0.01 * (step / max_steps)

        # Bonus for early completion
        if base_reward > 0.9 and step < max_steps * 0.8:
            early_bonus = 0.1 * (1 - step / max_steps)
        else:
            early_bonus = 0.0

        total_reward = base_reward + time_penalty + early_bonus

        return total_reward


class CurriculumReward:
    """
    Curriculum learning reward that adjusts difficulty.
    """

    def __init__(self, initial_threshold: float = 0.5):
        """
        Initialize curriculum reward.

        Args:
            initial_threshold: Initial threshold for success
        """
        self.threshold = initial_threshold
        self.success_count = 0
        self.total_count = 0

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate curriculum reward.

        Args:
            pred: Predicted mask
            gt: Ground truth mask

        Returns:
            Curriculum reward
        """
        dice = SegmentationRewards.dice_coefficient(pred, gt)

        # Binary reward based on threshold
        success = dice >= self.threshold
        reward = 1.0 if success else 0.0

        # Update statistics
        self.total_count += 1
        if success:
            self.success_count += 1

        # Adapt threshold based on success rate
        if self.total_count >= 100:
            success_rate = self.success_count / self.total_count
            if success_rate > 0.8:
                # Increase difficulty
                self.threshold = min(0.95, self.threshold + 0.05)
            elif success_rate < 0.5:
                # Decrease difficulty
                self.threshold = max(0.3, self.threshold - 0.05)

            # Reset counters
            self.success_count = 0
            self.total_count = 0

        return reward


class MetricsCalculator:
    """
    Calculate all segmentation metrics.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.rewards = SegmentationRewards()

    def calculate_all(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """
        Calculate all metrics.

        Args:
            pred: Predicted mask
            gt: Ground truth mask

        Returns:
            Dictionary with all metrics
        """
        precision, recall, f1 = self.rewards.precision_recall_f1(pred, gt)

        metrics = {
            'dice': self.rewards.dice_coefficient(pred, gt),
            'iou': self.rewards.iou_score(pred, gt),
            'pixel_accuracy': self.rewards.pixel_accuracy(pred, gt),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'boundary_iou': self.rewards.boundary_iou(pred, gt),
            # 'hausdorff': self.rewards.hausdorff_distance(pred, gt)
        }

        return metrics


if __name__ == "__main__":
    # Test rewards
    pred = np.random.rand(256, 256) > 0.5
    gt = np.random.rand(256, 256) > 0.5

    rewards = SegmentationRewards()
    print(f"Dice: {rewards.dice_coefficient(pred, gt):.4f}")
    print(f"IoU: {rewards.iou_score(pred, gt):.4f}")
    print(f"Pixel Accuracy: {rewards.pixel_accuracy(pred, gt):.4f}")

    calc = MetricsCalculator()
    metrics = calc.calculate_all(pred, gt)
    print("\nAll metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
