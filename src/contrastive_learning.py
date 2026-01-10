"""
Contrastive Learning для улучшения представлений.

Агент учится различать:
1. Поля разных типов (пашня vs не-пашня)
2. Границы vs внутренние регионы
3. Сложные vs простые случаи
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class ContrastiveEncoder(nn.Module):
    """
    Encoder для contrastive learning.
    Создает embeddings, которые близки для похожих регионов
    и далеки для разных.
    """

    def __init__(self, in_channels: int = 4, embedding_dim: int = 128):
        """
        Initialize contrastive encoder.

        Args:
            in_channels: Number of input channels
            embedding_dim: Dimension of embeddings
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Embeddings (B, embedding_dim)
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projector(features)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Притягивает embeddings похожих примеров (одного класса)
    и отталкивает embeddings разных классов.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Initialize SupCon loss.

        Args:
            temperature: Temperature parameter
            base_temperature: Base temperature
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute SupCon loss.

        Args:
            features: Normalized embeddings (B, embedding_dim)
            labels: Labels (B,)
            mask: Optional mask (B, B)

        Returns:
            Loss value
        """
        device = features.device
        batch_size = features.shape[0]

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Create label mask: 1 if same label, 0 otherwise
        labels = labels.contiguous().view(-1, 1)
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)

        # Remove diagonal (self-similarity)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        # Mask out self-contrast
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class BoundaryContrastiveLearning(nn.Module):
    """
    Contrastive learning специально для границ полей.

    Учится различать:
    - Четкие границы vs размытые границы
    - Границы поля vs внутренние регионы
    - Углы vs прямые участки границ
    """

    def __init__(self, in_channels: int = 4):
        """
        Initialize boundary contrastive learning.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        # Encoder для boundary regions
        self.boundary_encoder = ContrastiveEncoder(in_channels, embedding_dim=128)

        # Classifier для типов границ
        self.boundary_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 типа: четкая, размытая, угол, прямая
        )

        # Loss
        self.contrastive_loss = SupConLoss(temperature=0.07)

    def extract_boundary_patches(self,
                                 image: torch.Tensor,
                                 mask: torch.Tensor,
                                 patch_size: int = 32,
                                 num_patches: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patches around boundaries.

        Args:
            image: Input image (B, C, H, W)
            mask: Ground truth mask (B, 1, H, W)
            patch_size: Size of extracted patches
            num_patches: Number of patches to extract

        Returns:
            Tuple of (boundary_patches, labels)
        """
        B, C, H, W = image.shape

        # Detect boundaries using edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        if image.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()

        edges_x = F.conv2d(mask, sobel_x, padding=1)
        edges_y = F.conv2d(mask, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

        # Find boundary pixels
        boundary_pixels = (edges > 0.1).squeeze(1)  # (B, H, W)

        patches = []
        labels = []

        for b in range(B):
            boundary_coords = torch.where(boundary_pixels[b])

            if len(boundary_coords[0]) < num_patches:
                continue

            # Sample random boundary points
            indices = torch.randperm(len(boundary_coords[0]))[:num_patches]

            for idx in indices:
                y = boundary_coords[0][idx].item()
                x = boundary_coords[1][idx].item()

                # Extract patch centered at boundary point
                y1 = max(0, y - patch_size // 2)
                y2 = min(H, y + patch_size // 2)
                x1 = max(0, x - patch_size // 2)
                x2 = min(W, x + patch_size // 2)

                patch = image[b:b+1, :, y1:y2, x1:x2]

                # Pad if necessary
                if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                    patch = F.pad(patch, (0, patch_size - patch.shape[3],
                                         0, patch_size - patch.shape[2]))

                patches.append(patch)

                # Classify boundary type based on local edge patterns
                # 0: четкая граница, 1: размытая, 2: угол, 3: прямая
                local_edge = edges[b, 0, y1:y2, x1:x2]
                edge_strength = local_edge.mean().item()

                if edge_strength > 0.5:
                    label = 0  # четкая
                elif edge_strength > 0.2:
                    label = 3  # прямая
                elif edge_strength > 0.1:
                    label = 1  # размытая
                else:
                    label = 2  # угол (corner)

                labels.append(label)

        if len(patches) == 0:
            return None, None

        patches = torch.cat(patches, dim=0)
        labels = torch.tensor(labels, device=image.device)

        return patches, labels

    def forward(self,
                image: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with contrastive learning.

        Args:
            image: Input image (B, C, H, W)
            mask: Ground truth mask (B, 1, H, W)

        Returns:
            Tuple of (loss, info_dict)
        """
        # Extract boundary patches
        patches, labels = self.extract_boundary_patches(image, mask)

        if patches is None:
            return torch.tensor(0.0, device=image.device), {}

        # Encode patches
        embeddings = self.boundary_encoder(patches)

        # Contrastive loss
        contrastive_loss = self.contrastive_loss(embeddings, labels)

        # Classification loss (auxiliary)
        class_logits = self.boundary_classifier(embeddings)
        classification_loss = F.cross_entropy(class_logits, labels)

        # Total loss
        total_loss = contrastive_loss + 0.1 * classification_loss

        info = {
            'contrastive_loss': contrastive_loss.item(),
            'classification_loss': classification_loss.item(),
            'embeddings': embeddings.detach(),
            'labels': labels
        }

        return total_loss, info


class FieldTypeContrastive(nn.Module):
    """
    Contrastive learning для различных типов полей.

    Учится различать поля по:
    - Размеру
    - Форме (прямоугольные, неправильные)
    - Текстуре
    - Спектральной сигнатуре
    """

    def __init__(self, in_channels: int = 4):
        """
        Initialize field type contrastive learning.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        self.encoder = ContrastiveEncoder(in_channels, embedding_dim=256)
        self.contrastive_loss = SupConLoss(temperature=0.1)

        # Field attribute predictor
        self.attribute_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Size class, shape class, texture class
        )

    def compute_field_attributes(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attributes of field from mask.

        Args:
            mask: Binary mask (B, 1, H, W)

        Returns:
            Attribute tensor (B, 3) - [size_class, shape_class, texture_class]
        """
        B = mask.size(0)
        attributes = []

        for b in range(B):
            m = mask[b, 0]

            # Size: small, medium, large
            area = m.sum().item()
            total_area = m.numel()
            area_ratio = area / total_area

            if area_ratio < 0.1:
                size_class = 0  # small
            elif area_ratio < 0.3:
                size_class = 1  # medium
            else:
                size_class = 2  # large

            # Shape: compute compactness (perimeter^2 / area)
            # Simple approximation using edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            if m.is_cuda:
                sobel_x = sobel_x.cuda()

            edges_x = F.conv2d(m.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            edges_y = F.conv2d(m.unsqueeze(0).unsqueeze(0), sobel_x.transpose(-1, -2), padding=1)
            perimeter = torch.sqrt(edges_x ** 2 + edges_y ** 2).sum().item()

            compactness = (perimeter ** 2) / (area + 1e-6)

            if compactness < 20:
                shape_class = 0  # compact (круглое/квадратное)
            elif compactness < 50:
                shape_class = 1  # moderate
            else:
                shape_class = 2  # elongated/irregular

            # Texture: simplified - just use random for now
            texture_class = np.random.randint(0, 3)

            attributes.append([size_class, shape_class, texture_class])

        return torch.tensor(attributes, device=mask.device)

    def forward(self,
                image: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with field type contrastive learning.

        Args:
            image: Input image (B, C, H, W)
            mask: Ground truth mask (B, 1, H, W)

        Returns:
            Tuple of (loss, info_dict)
        """
        # Encode full images
        embeddings = self.encoder(image)

        # Compute field attributes
        attributes = self.compute_field_attributes(mask)

        # Create composite labels from attributes
        # Combine size, shape, texture into single label
        labels = attributes[:, 0] * 9 + attributes[:, 1] * 3 + attributes[:, 2]

        # Contrastive loss
        contrastive_loss = self.contrastive_loss(embeddings, labels)

        # Predict attributes
        pred_attributes = self.attribute_predictor(embeddings)

        # Attribute prediction loss
        attribute_loss = F.cross_entropy(pred_attributes, attributes.long())

        # Total loss
        total_loss = contrastive_loss + 0.2 * attribute_loss

        info = {
            'contrastive_loss': contrastive_loss.item(),
            'attribute_loss': attribute_loss.item(),
            'embeddings': embeddings.detach()
        }

        return total_loss, info


class HardNegativeMining(nn.Module):
    """
    Hard negative mining для улучшения обучения.

    Находит наиболее сложные negative examples и фокусируется на них.
    """

    def __init__(self, memory_size: int = 1000):
        """
        Initialize hard negative mining.

        Args:
            memory_size: Size of negative example memory
        """
        super().__init__()
        self.memory_size = memory_size
        self.negative_memory = []

    def add_negatives(self, embeddings: torch.Tensor, difficulties: torch.Tensor):
        """
        Add negative examples to memory.

        Args:
            embeddings: Negative embeddings
            difficulties: Difficulty scores
        """
        for emb, diff in zip(embeddings, difficulties):
            self.negative_memory.append((emb.detach().cpu(), diff.item()))

        # Keep only top-k hardest
        self.negative_memory.sort(key=lambda x: x[1], reverse=True)
        self.negative_memory = self.negative_memory[:self.memory_size]

    def get_hard_negatives(self, k: int = 32) -> torch.Tensor:
        """
        Get k hardest negative examples.

        Args:
            k: Number of negatives to retrieve

        Returns:
            Hard negative embeddings
        """
        if len(self.negative_memory) == 0:
            return None

        k = min(k, len(self.negative_memory))
        hard_negatives = [self.negative_memory[i][0] for i in range(k)]

        return torch.stack(hard_negatives)


if __name__ == "__main__":
    # Test contrastive learning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test boundary contrastive learning
    boundary_cl = BoundaryContrastiveLearning(in_channels=4).to(device)

    # Test input
    image = torch.randn(4, 4, 256, 256).to(device)
    mask = (torch.randn(4, 1, 256, 256) > 0).float().to(device)

    loss, info = boundary_cl(image, mask)
    print(f"Boundary Contrastive Loss: {loss.item():.4f}")
    print(f"Info: {list(info.keys())}")

    # Test field type contrastive learning
    field_cl = FieldTypeContrastive(in_channels=4).to(device)

    loss, info = field_cl(image, mask)
    print(f"\nField Type Contrastive Loss: {loss.item():.4f}")
    print(f"Info: {list(info.keys())}")
