"""
Meta-Learning для быстрой адаптации к новым регионам.

Использует MAML (Model-Agnostic Meta-Learning) для того, чтобы агент
мог быстро адаптироваться к новым типам полей, регионам, сезонам
с минимальным количеством примеров.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from copy import deepcopy


class MAMLSegmentation(nn.Module):
    """
    MAML для быстрой адаптации сегментации к новым доменам.

    Meta-learning позволяет:
    1. Быстро адаптироваться к новым регионам (другая страна, климат)
    2. Адаптироваться к новым сезонам (весна, лето, осень)
    3. Работать с новыми типами полей (рис, кукуруза, пшеница)
    4. Адаптироваться к разным разрешениям спутников
    """

    def __init__(self,
                 base_model: nn.Module,
                 inner_lr: float = 0.01,
                 num_inner_steps: int = 5,
                 first_order: bool = False):
        """
        Initialize MAML.

        Args:
            base_model: Base segmentation model
            inner_lr: Learning rate for inner loop (adaptation)
            num_inner_steps: Number of gradient steps in inner loop
            first_order: Use first-order approximation (faster but less accurate)
        """
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

    def clone_model(self) -> nn.Module:
        """Clone the base model."""
        return deepcopy(self.base_model)

    def adapt(self,
              model: nn.Module,
              support_images: torch.Tensor,
              support_masks: torch.Tensor,
              compute_second_order: bool = True) -> nn.Module:
        """
        Adapt model to support set (inner loop).

        Args:
            model: Model to adapt
            support_images: Support images (N, C, H, W)
            support_masks: Support masks (N, 1, H, W)
            compute_second_order: Whether to compute second-order gradients

        Returns:
            Adapted model
        """
        adapted_model = self.clone_model()

        # Create optimizer for inner loop
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        # Inner loop: adapt to support set
        for step in range(self.num_inner_steps):
            # Forward pass
            predictions = adapted_model(support_images)

            # Compute loss (Dice loss)
            loss = self.dice_loss(predictions, support_masks)

            # Backward pass
            inner_optimizer.zero_grad()

            if compute_second_order and not self.first_order:
                loss.backward(create_graph=True)
            else:
                loss.backward()

            inner_optimizer.step()

        return adapted_model

    def forward(self,
                support_images: torch.Tensor,
                support_masks: torch.Tensor,
                query_images: torch.Tensor,
                query_masks: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Meta-learning forward pass.

        Args:
            support_images: Support set images (K, C, H, W)
            support_masks: Support set masks (K, 1, H, W)
            query_images: Query set images (Q, C, H, W)
            query_masks: Query set masks (Q, 1, H, W)

        Returns:
            Tuple of (meta_loss, info_dict)
        """
        # Adapt to support set
        adapted_model = self.adapt(
            self.base_model,
            support_images,
            support_masks,
            compute_second_order=not self.first_order
        )

        # Evaluate on query set
        query_predictions = adapted_model(query_images)

        # Compute meta loss
        meta_loss = self.dice_loss(query_predictions, query_masks)

        # Calculate metrics
        with torch.no_grad():
            dice_score = self.dice_coefficient(query_predictions, query_masks)
            iou_score = self.iou(query_predictions, query_masks)

        info = {
            'meta_loss': meta_loss.item(),
            'query_dice': dice_score.item(),
            'query_iou': iou_score.item(),
            'predictions': query_predictions.detach()
        }

        return meta_loss, info

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        pred = torch.sigmoid(pred)
        smooth = 1e-5

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice coefficient."""
        pred = (torch.sigmoid(pred) > 0.5).float()
        smooth = 1e-5

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        return (2. * intersection + smooth) / (union + smooth)

    def iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU."""
        pred = (torch.sigmoid(pred) > 0.5).float()
        smooth = 1e-5

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        return (intersection + smooth) / (union + smooth)


class TaskDistribution:
    """
    Распределение задач для meta-learning.

    Каждая задача = новый регион/сезон/тип поля
    """

    def __init__(self,
                 datasets: List[Dict],
                 k_shot: int = 5,
                 q_query: int = 15):
        """
        Initialize task distribution.

        Args:
            datasets: List of datasets (different domains)
            k_shot: Number of support examples per task
            q_query: Number of query examples per task
        """
        self.datasets = datasets
        self.k_shot = k_shot
        self.q_query = q_query

    def sample_task(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a task (support + query sets).

        Returns:
            Tuple of (support_images, support_masks, query_images, query_masks)
        """
        # Select random dataset (domain)
        dataset = np.random.choice(self.datasets)

        # Sample support examples
        support_indices = np.random.choice(len(dataset), self.k_shot, replace=False)
        support_images = torch.stack([dataset[i]['image'] for i in support_indices])
        support_masks = torch.stack([dataset[i]['mask'] for i in support_indices])

        # Sample query examples (different from support)
        remaining_indices = list(set(range(len(dataset))) - set(support_indices))
        query_indices = np.random.choice(remaining_indices, self.q_query, replace=False)
        query_images = torch.stack([dataset[i]['image'] for i in query_indices])
        query_masks = torch.stack([dataset[i]['mask'] for i in query_indices])

        return support_images, support_masks, query_images, query_masks


class ProtoNet(nn.Module):
    """
    Prototypical Networks для few-shot segmentation.

    Альтернатива MAML: учится создавать "прототипы" для разных типов полей
    и классифицирует пиксели на основе близости к прототипам.
    """

    def __init__(self, encoder: nn.Module, embedding_dim: int = 256):
        """
        Initialize Prototypical Network.

        Args:
            encoder: Feature encoder
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        self.encoder = encoder

        # Projection to embedding space
        self.projector = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # Assuming encoder outputs 512 channels
            nn.ReLU(),
            nn.Conv2d(256, embedding_dim, 1)
        )

    def compute_prototypes(self,
                          support_images: torch.Tensor,
                          support_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class prototypes from support set.

        Args:
            support_images: Support images (K, C, H, W)
            support_masks: Support masks (K, 1, H, W)

        Returns:
            Tuple of (crop_prototype, non_crop_prototype)
        """
        # Encode support images
        features = self.encoder(support_images)  # (K, 512, H', W')
        embeddings = self.projector(features)  # (K, embedding_dim, H', W')

        K, D, H, W = embeddings.shape

        # Downsample masks
        masks = F.interpolate(support_masks, size=(H, W), mode='nearest')

        # Compute prototypes
        # Crop prototype: average of embeddings where mask = 1
        crop_mask = masks == 1  # (K, 1, H, W)
        crop_embeddings = embeddings * crop_mask  # (K, D, H, W)
        crop_prototype = crop_embeddings.sum(dim=[0, 2, 3]) / (crop_mask.sum() + 1e-5)  # (D,)

        # Non-crop prototype: average of embeddings where mask = 0
        non_crop_mask = masks == 0
        non_crop_embeddings = embeddings * non_crop_mask
        non_crop_prototype = non_crop_embeddings.sum(dim=[0, 2, 3]) / (non_crop_mask.sum() + 1e-5)

        return crop_prototype, non_crop_prototype

    def predict_with_prototypes(self,
                                query_images: torch.Tensor,
                                crop_prototype: torch.Tensor,
                                non_crop_prototype: torch.Tensor) -> torch.Tensor:
        """
        Predict query images using prototypes.

        Args:
            query_images: Query images (Q, C, H, W)
            crop_prototype: Crop class prototype (D,)
            non_crop_prototype: Non-crop class prototype (D,)

        Returns:
            Predicted masks (Q, 1, H, W)
        """
        # Encode query images
        features = self.encoder(query_images)
        embeddings = self.projector(features)  # (Q, D, H', W')

        Q, D, H, W = embeddings.shape

        # Compute distances to prototypes
        embeddings_flat = embeddings.view(Q, D, -1).permute(0, 2, 1)  # (Q, H*W, D)

        # Distance to crop prototype
        crop_dist = torch.cdist(embeddings_flat, crop_prototype.unsqueeze(0).unsqueeze(0))  # (Q, H*W, 1)
        crop_dist = crop_dist.view(Q, H, W, 1).permute(0, 3, 1, 2)  # (Q, 1, H, W)

        # Distance to non-crop prototype
        non_crop_dist = torch.cdist(embeddings_flat, non_crop_prototype.unsqueeze(0).unsqueeze(0))
        non_crop_dist = non_crop_dist.view(Q, H, W, 1).permute(0, 3, 1, 2)

        # Classification: closer to crop prototype -> crop
        predictions = (crop_dist < non_crop_dist).float()

        # Upsample to original resolution
        predictions = F.interpolate(predictions, size=query_images.shape[2:], mode='bilinear', align_corners=False)

        return predictions

    def forward(self,
                support_images: torch.Tensor,
                support_masks: torch.Tensor,
                query_images: torch.Tensor,
                query_masks: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            support_images: Support set images (K, C, H, W)
            support_masks: Support set masks (K, 1, H, W)
            query_images: Query set images (Q, C, H, W)
            query_masks: Query set masks (Q, 1, H, W)

        Returns:
            Tuple of (loss, info_dict)
        """
        # Compute prototypes from support set
        crop_prototype, non_crop_prototype = self.compute_prototypes(
            support_images, support_masks
        )

        # Predict query set
        predictions = self.predict_with_prototypes(
            query_images, crop_prototype, non_crop_prototype
        )

        # Compute loss
        loss = F.binary_cross_entropy(predictions, query_masks)

        # Metrics
        with torch.no_grad():
            dice = self.dice_coefficient(predictions, query_masks)
            iou = self.iou(predictions, query_masks)

        info = {
            'loss': loss.item(),
            'dice': dice.item(),
            'iou': iou.item(),
            'predictions': predictions.detach()
        }

        return loss, info

    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice coefficient."""
        pred = (pred > 0.5).float()
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection + smooth) / (union + smooth)

    def iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU."""
        pred = (pred > 0.5).float()
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)


class DomainAdaptation(nn.Module):
    """
    Domain Adaptation для переноса между регионами.

    Использует adversarial training чтобы сделать features
    domain-invariant.
    """

    def __init__(self, feature_extractor: nn.Module, feature_dim: int = 512):
        """
        Initialize domain adaptation.

        Args:
            feature_extractor: Feature extractor
            feature_dim: Feature dimension
        """
        super().__init__()
        self.feature_extractor = feature_extractor

        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Gradient reversal layer coefficient
        self.alpha = 1.0

    def set_alpha(self, alpha: float):
        """Set gradient reversal coefficient."""
        self.alpha = alpha

    def forward(self,
                source_images: torch.Tensor,
                target_images: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Domain adaptation forward pass.

        Args:
            source_images: Images from source domain (B, C, H, W)
            target_images: Images from target domain (B, C, H, W)

        Returns:
            Tuple of (domain_loss, info_dict)
        """
        # Extract features
        source_features = self.feature_extractor(source_images)  # (B, feature_dim)
        target_features = self.feature_extractor(target_images)

        # Gradient reversal (negate gradient during backprop)
        source_features_reversed = GradientReversalFunction.apply(source_features, self.alpha)
        target_features_reversed = GradientReversalFunction.apply(target_features, self.alpha)

        # Domain prediction
        source_domain = self.domain_discriminator(source_features_reversed)
        target_domain = self.domain_discriminator(target_features_reversed)

        # Domain labels: source = 0, target = 1
        source_labels = torch.zeros_like(source_domain)
        target_labels = torch.ones_like(target_domain)

        # Domain classification loss
        domain_loss = (F.binary_cross_entropy(source_domain, source_labels) +
                      F.binary_cross_entropy(target_domain, target_labels))

        # Domain accuracy
        with torch.no_grad():
            source_acc = ((source_domain < 0.5).float() == source_labels).float().mean()
            target_acc = ((target_domain > 0.5).float() == target_labels).float().mean()

        info = {
            'domain_loss': domain_loss.item(),
            'source_domain_acc': source_acc.item(),
            'target_domain_acc': target_acc.item()
        }

        return domain_loss, info


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer.

    During forward pass, acts as identity.
    During backward pass, reverses and scales gradient.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


if __name__ == "__main__":
    # Test meta-learning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.models import CNNFeatureExtractor

    # Create simple model for testing
    class SimpleSegModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = CNNFeatureExtractor(4, 256)
            self.decoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 256 * 256)
            )

        def forward(self, x):
            features = self.encoder(x)
            output = self.decoder(features)
            return output.view(-1, 1, 256, 256)

    model = SimpleSegModel().to(device)

    # Test MAML
    maml = MAMLSegmentation(model, inner_lr=0.01, num_inner_steps=3)

    # Generate dummy data
    support_images = torch.randn(5, 4, 256, 256).to(device)
    support_masks = (torch.randn(5, 1, 256, 256) > 0).float().to(device)
    query_images = torch.randn(10, 4, 256, 256).to(device)
    query_masks = (torch.randn(10, 1, 256, 256) > 0).float().to(device)

    meta_loss, info = maml(support_images, support_masks, query_images, query_masks)
    print(f"MAML Meta Loss: {meta_loss.item():.4f}")
    print(f"Query Dice: {info['query_dice']:.4f}")
    print(f"Query IoU: {info['query_iou']:.4f}")

    print("\n✓ Meta-learning module working!")
