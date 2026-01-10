"""
Hierarchical Multi-Scale RL Agent with Active Boundary Refinement.

Ключевые инновации:
1. Hierarchical policy: грубая сегментация -> уточнение границ
2. Multi-scale attention: агент работает на разных масштабах одновременно
3. Active boundary refinement: фокус на сложных границах
4. Uncertainty estimation: агент знает, где он не уверен
5. Graph Neural Network: моделирование пространственных отношений
6. Contrastive learning: различение похожих полей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from src.models import CNNFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiScaleAttention(nn.Module):
    """
    Multi-scale spatial attention mechanism.
    Агент учится фокусироваться на разных масштабах одновременно.
    """

    def __init__(self, in_channels: int, scales: List[int] = [1, 2, 4, 8]):
        """
        Initialize multi-scale attention.

        Args:
            in_channels: Number of input channels
            scales: List of scale factors
        """
        super().__init__()
        self.scales = scales

        # Attention for each scale
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in scales
        ])

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with multi-scale attention.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (attended features, attention maps)
        """
        B, C, H, W = x.shape
        multi_scale_features = []
        attention_maps = []

        for scale, attention_module in zip(self.scales, self.scale_attentions):
            # Downsample
            if scale > 1:
                x_scaled = F.avg_pool2d(x, scale, scale)
            else:
                x_scaled = x

            # Compute attention
            attention = attention_module(x_scaled)
            attention_maps.append(attention)

            # Apply attention
            attended = x_scaled * attention

            # Upsample back
            if scale > 1:
                attended = F.interpolate(attended, size=(H, W), mode='bilinear', align_corners=False)

            multi_scale_features.append(attended)

        # Fuse multi-scale features
        fused = torch.cat(multi_scale_features, dim=1)
        output = self.scale_fusion(fused)

        return output, attention_maps


class BoundaryRefinementModule(nn.Module):
    """
    Active boundary refinement module.
    Фокусируется на уточнении границ полей, а не на всем изображении.
    """

    def __init__(self, in_channels: int):
        """
        Initialize boundary refinement module.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        # Boundary detection
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 3, padding=1),  # +1 for coarse mask
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Boundary refinement
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, 3, padding=1),  # +2 for mask and boundary
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def detect_boundaries(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Detect uncertain boundary regions.

        Args:
            features: Feature tensor (B, C, H, W)
            mask: Current mask prediction (B, 1, H, W)

        Returns:
            Boundary probability map (B, 1, H, W)
        """
        # Edge detection on mask
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        if features.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()

        edges_x = F.conv2d(mask, sobel_x, padding=1)
        edges_y = F.conv2d(mask, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

        # Predict boundary uncertainty
        boundary_input = torch.cat([features, mask], dim=1)
        boundary_prob = self.boundary_detector(boundary_input)

        # Combine with edge detection
        boundary_prob = boundary_prob * (edges + 0.1)  # Bias towards detected edges

        return boundary_prob

    def forward(self, features: torch.Tensor, coarse_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine boundaries.

        Args:
            features: Feature tensor (B, C, H, W)
            coarse_mask: Coarse segmentation mask (B, 1, H, W)

        Returns:
            Tuple of (refined mask, boundary map)
        """
        # Detect boundaries
        boundary_map = self.detect_boundaries(features, coarse_mask)

        # Refine boundaries
        refiner_input = torch.cat([features, coarse_mask, boundary_map], dim=1)
        refinement = self.refiner(refiner_input)

        # Combine coarse mask with refinement
        # Apply strong refinement at boundaries, weak elsewhere
        refined_mask = coarse_mask * (1 - boundary_map) + refinement * boundary_map

        return refined_mask, boundary_map


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation module.
    Агент оценивает свою уверенность в предсказаниях.
    """

    def __init__(self, in_channels: int):
        """
        Initialize uncertainty estimator.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty.

        Args:
            features: Feature tensor (B, C, H, W)
            prediction: Prediction tensor (B, 1, H, W)

        Returns:
            Uncertainty map (B, 1, H, W)
        """
        x = torch.cat([features, prediction], dim=1)
        uncertainty = self.uncertainty_net(x)
        return uncertainty


class SpatialRelationGraph(nn.Module):
    """
    Graph Neural Network для моделирования пространственных отношений.
    Агент понимает, как разные регионы связаны друг с другом.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        """
        Initialize spatial relation graph.

        Args:
            feature_dim: Feature dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Distance + relative position
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Node update
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def build_graph(self, features: torch.Tensor, grid_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build spatial graph from feature map.

        Args:
            features: Feature tensor (B, C, H, W)
            grid_size: Number of nodes in grid

        Returns:
            Tuple of (node_features, edge_features)
        """
        B, C, H, W = features.shape

        # Sample grid of nodes
        h_step = H // grid_size
        w_step = W // grid_size

        node_features = []
        node_positions = []

        for i in range(grid_size):
            for j in range(grid_size):
                h = i * h_step + h_step // 2
                w = j * w_step + w_step // 2

                # Extract node feature
                node_feat = features[:, :, h, w]  # (B, C)
                node_features.append(node_feat)

                # Store position
                node_positions.append([h / H, w / W])

        node_features = torch.stack(node_features, dim=1)  # (B, N, C)
        node_positions = torch.tensor(node_positions, device=features.device)  # (N, 2)

        # Compute edge features (distance + relative position)
        N = grid_size * grid_size
        edge_features = torch.zeros(B, N, N, 4, device=features.device)

        for i in range(N):
            for j in range(N):
                if i != j:
                    pos_i = node_positions[i]
                    pos_j = node_positions[j]

                    # Euclidean distance
                    dist = torch.sqrt(torch.sum((pos_i - pos_j) ** 2))

                    # Relative position
                    rel_pos = pos_j - pos_i

                    edge_features[:, i, j] = torch.tensor([dist, dist, rel_pos[0], rel_pos[1]], device=features.device)

        return node_features, edge_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph network.

        Args:
            features: Feature tensor (B, C, H, W)

        Returns:
            Updated features (B, C, H, W)
        """
        B, C, H, W = features.shape

        # Build graph
        node_features, edge_features = self.build_graph(features)
        N = node_features.size(1)

        # Encode nodes
        node_embed = self.node_encoder(node_features)  # (B, N, hidden_dim)

        # Message passing
        messages = []
        for i in range(N):
            node_i = node_embed[:, i:i+1]  # (B, 1, hidden_dim)

            node_i_expand = node_i.expand(-1, N, -1)  # (B, N, hidden_dim)

            # Encode edges
            edge_embed = self.edge_encoder(edge_features[:, i])  # (B, N, hidden_dim)

            # Compute messages
            message_input = torch.cat([node_i_expand, node_embed, edge_embed], dim=-1)
            message = self.message_net(message_input)  # (B, N, hidden_dim)

            # Aggregate messages
            message = torch.sum(message, dim=1, keepdim=True)  # (B, 1, hidden_dim)
            messages.append(message)

        messages = torch.cat(messages, dim=1)  # (B, N, hidden_dim)

        # Update nodes
        update_input = torch.cat([node_embed, messages], dim=-1)
        updated_nodes = self.update_net(update_input)  # (B, N, hidden_dim)

        # Map back to spatial grid (simple interpolation)
        # For simplicity, use nearest neighbor mapping
        # In practice, you'd use more sophisticated interpolation

        return features  # Placeholder - in full implementation, update feature map


class HierarchicalSegmentationPolicy(nn.Module):
    """
    Hierarchical policy для сегментации.

    Уровень 1: Грубая сегментация (быстро)
    Уровень 2: Уточнение границ (точно)
    Уровень 3: Коррекция ошибок (метапредсказания)
    """

    def __init__(self,
                 in_channels: int = 4,
                 num_coarse_actions: int = 16,
                 num_refine_actions: int = 64):
        """
        Initialize hierarchical policy.

        Args:
            in_channels: Number of input channels
            num_coarse_actions: Actions for coarse segmentation
            num_refine_actions: Actions for boundary refinement
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_coarse_actions = num_coarse_actions
        self.num_refine_actions = num_refine_actions

        # Feature extraction
        self.feature_extractor = CNNFeatureExtractor(in_channels, features_dim=512)

        # Multi-scale attention
        self.multi_scale_attention = MultiScaleAttention(512)

        # Spatial relation graph
        self.spatial_graph = SpatialRelationGraph(512)

        # Level 1: Coarse segmentation policy
        self.coarse_policy = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_coarse_actions * 2)  # *2 for binary actions
        )

        # Level 2: Boundary refinement module
        self.boundary_refinement = BoundaryRefinementModule(512)

        # Level 3: Error correction policy
        self.error_correction = nn.Sequential(
            nn.Linear(512 + 1, 256),  # +1 for uncertainty
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_refine_actions)
        )

        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(512)

        # Value networks for each level
        self.coarse_value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.refine_value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-level features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary with features at different levels
        """
        # Base features
        base_features = self.feature_extractor(x)  # (B, 512)

        # Reshape for spatial operations
        B = x.size(0)
        H, W = x.size(2) // 16, x.size(3) // 16  # After pooling
        spatial_features = base_features.view(B, 512, 1, 1).expand(B, 512, H, W)

        # Multi-scale attention
        attended_features, attention_maps = self.multi_scale_attention(spatial_features)

        # Spatial relations
        graph_features = self.spatial_graph(attended_features)

        return {
            'base': base_features,
            'spatial': spatial_features,
            'attended': attended_features,
            'graph': graph_features,
            'attention_maps': attention_maps
        }

    def coarse_segmentation(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Level 1: Coarse segmentation.

        Args:
            features: Feature dictionary

        Returns:
            Tuple of (coarse_mask, value)
        """
        base_features = features['base']

        # Policy
        action_logits = self.coarse_policy(base_features)

        # Value
        value = self.coarse_value(base_features)

        # Sample actions and create mask
        # Simplified version - in full implementation, use proper action space
        probs = torch.sigmoid(action_logits)
        actions = (probs > 0.5).float()

        # Create coarse mask (reshape actions to spatial grid)
        B = base_features.size(0)
        grid_size = int(np.sqrt(self.num_coarse_actions))
        coarse_mask = actions[:, ::2].view(B, 1, grid_size, grid_size)

        # Upsample to full resolution
        H, W = features['spatial'].size(2), features['spatial'].size(3)
        coarse_mask = F.interpolate(coarse_mask, size=(H * 16, W * 16), mode='bilinear')

        return coarse_mask, value

    def refine_boundaries(self,
                         features: Dict[str, torch.Tensor],
                         coarse_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Level 2: Boundary refinement.

        Args:
            features: Feature dictionary
            coarse_mask: Coarse segmentation mask

        Returns:
            Tuple of (refined_mask, boundary_map, value)
        """
        attended_features = features['attended']

        # Downsample mask to match feature resolution
        H, W = attended_features.size(2), attended_features.size(3)
        coarse_mask_down = F.interpolate(coarse_mask, size=(H, W), mode='bilinear')

        # Refine boundaries
        refined_mask, boundary_map = self.boundary_refinement(attended_features, coarse_mask_down)

        # Upsample back
        refined_mask = F.interpolate(refined_mask, size=(H * 16, W * 16), mode='bilinear')
        boundary_map = F.interpolate(boundary_map, size=(H * 16, W * 16), mode='bilinear')

        # Value
        value = self.refine_value(features['base'])

        return refined_mask, boundary_map, value

    def estimate_uncertainty(self,
                           features: Dict[str, torch.Tensor],
                           prediction: torch.Tensor) -> torch.Tensor:
        """
        Estimate prediction uncertainty.

        Args:
            features: Feature dictionary
            prediction: Current prediction

        Returns:
            Uncertainty map
        """
        attended_features = features['attended']

        # Downsample prediction
        H, W = attended_features.size(2), attended_features.size(3)
        prediction_down = F.interpolate(prediction, size=(H, W), mode='bilinear')

        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(attended_features, prediction_down)

        # Upsample
        uncertainty = F.interpolate(uncertainty, size=(H * 16, W * 16), mode='bilinear')

        return uncertainty

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full hierarchical forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary with predictions at each level
        """
        # Extract features
        features = self.extract_features(x)

        # Level 1: Coarse segmentation
        coarse_mask, coarse_value = self.coarse_segmentation(features)

        # Level 2: Boundary refinement
        refined_mask, boundary_map, refine_value = self.refine_boundaries(features, coarse_mask)

        # Estimate uncertainty
        uncertainty = self.estimate_uncertainty(features, refined_mask)

        return {
            'coarse_mask': coarse_mask,
            'refined_mask': refined_mask,
            'boundary_map': boundary_map,
            'uncertainty': uncertainty,
            'coarse_value': coarse_value,
            'refine_value': refine_value,
            'attention_maps': features['attention_maps']
        }


if __name__ == "__main__":
    # Test hierarchical policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HierarchicalSegmentationPolicy(in_channels=4).to(device)

    # Test input
    x = torch.randn(2, 4, 256, 256).to(device)

    # Forward pass
    output = model(x)

    print("Hierarchical Policy Output:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: {len(value)} attention maps")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
