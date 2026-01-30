"""
Segmentation metrics for crop mapping.
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import distance_transform_edt


class SegmentationRewards:
    """
    Collection of segmentation metrics.
    """

    @staticmethod
    def dice_coefficient(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8) -> float:
        pred = (pred > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)

    @staticmethod
    def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8) -> float:
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
        pred = (pred > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        correct = np.sum(pred == gt)
        total = pred.size
        return float(correct / total)

    @staticmethod
    def precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
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
    def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation: int = 2) -> float:
        from scipy.ndimage import binary_dilation

        pred = (pred > 0.5).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)

        pred_boundary = binary_dilation(pred, iterations=dilation) ^ pred
        gt_boundary = binary_dilation(gt, iterations=dilation) ^ gt

        intersection = np.sum(pred_boundary & gt_boundary)
        union = np.sum(pred_boundary | gt_boundary)

        if union == 0:
            return 1.0

        return float(intersection / union)

    @staticmethod
    def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
        pred = (pred > 0.5).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)

        pred_dt = distance_transform_edt(1 - pred)
        gt_dt = distance_transform_edt(1 - gt)

        hd1 = np.max(pred_dt[gt == 1]) if np.any(gt == 1) else 0
        hd2 = np.max(gt_dt[pred == 1]) if np.any(pred == 1) else 0

        return float(max(hd1, hd2))


class MetricsCalculator:
    """Calculate all segmentation metrics."""

    def __init__(self):
        self.rewards = SegmentationRewards()

    def calculate_all(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        precision, recall, f1 = self.rewards.precision_recall_f1(pred, gt)

        return {
            "dice": self.rewards.dice_coefficient(pred, gt),
            "iou": self.rewards.iou_score(pred, gt),
            "pixel_accuracy": self.rewards.pixel_accuracy(pred, gt),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "boundary_iou": self.rewards.boundary_iou(pred, gt),
        }
