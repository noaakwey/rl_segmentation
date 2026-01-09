"""
Neural network models for RL-based segmentation.
Includes feature extractors and policy/value networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Initialize conv block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, channels: int):
        """
        Initialize residual block.

        Args:
            channels: Number of channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class UNetEncoder(nn.Module):
    """
    U-Net style encoder for feature extraction.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        """
        Initialize encoder.

        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
        """
        super().__init__()

        # Encoder path
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (bottleneck features, skip connections)
        """
        # Encoder with skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)

        return e4, [e1, e2, e3]


class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for RL agent.
    """

    def __init__(self,
                 in_channels: int = 4,
                 features_dim: int = 512):
        """
        Initialize feature extractor.

        Args:
            in_channels: Number of input channels (image + mask)
            features_dim: Dimension of output features
        """
        super().__init__()

        self.features_dim = features_dim

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),

            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),

            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),

            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )

        # Output projection
        self.projection = nn.Linear(512, features_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Feature vector (B, features_dim)
        """
        # Extract features
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)

        # Project to desired dimension
        features = self.projection(features)

        return features


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-style feature extractor with residual connections.
    """

    def __init__(self,
                 in_channels: int = 4,
                 features_dim: int = 512):
        """
        Initialize ResNet feature extractor.

        Args:
            in_channels: Number of input channels
            features_dim: Output feature dimension
        """
        super().__init__()

        self.features_dim = features_dim

        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(512, features_dim)

    def _make_layer(self, in_channels: int, out_channels: int,
                   num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Create residual layer."""
        layers = []

        # First block with potential stride
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        else:
            layers.append(ResidualBlock(in_channels))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x


class PolicyNetwork(nn.Module):
    """
    Policy network for action selection.
    """

    def __init__(self,
                 features_dim: int = 512,
                 num_actions: int = 32,
                 hidden_dim: int = 256):
        """
        Initialize policy network.

        Args:
            features_dim: Input feature dimension
            num_actions: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input features

        Returns:
            Action logits
        """
        return self.network(features)


class ValueNetwork(nn.Module):
    """
    Value network for state value estimation.
    """

    def __init__(self,
                 features_dim: int = 512,
                 hidden_dim: int = 256):
        """
        Initialize value network.

        Args:
            features_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input features

        Returns:
            State value
        """
        return self.network(features)


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    """

    def __init__(self,
                 in_channels: int = 4,
                 num_actions: int = 32,
                 features_dim: int = 512,
                 hidden_dim: int = 256,
                 feature_extractor: str = 'cnn'):
        """
        Initialize Actor-Critic network.

        Args:
            in_channels: Number of input channels
            num_actions: Number of discrete actions
            features_dim: Feature dimension
            hidden_dim: Hidden layer dimension
            feature_extractor: Type of feature extractor ('cnn' or 'resnet')
        """
        super().__init__()

        # Feature extractor
        if feature_extractor == 'cnn':
            self.feature_extractor = CNNFeatureExtractor(in_channels, features_dim)
        elif feature_extractor == 'resnet':
            self.feature_extractor = ResNetFeatureExtractor(in_channels, features_dim)
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")

        # Policy head
        self.policy_net = PolicyNetwork(features_dim, num_actions, hidden_dim)

        # Value head
        self.value_net = ValueNetwork(features_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input observation (B, C, H, W)

        Returns:
            Tuple of (action_logits, state_value)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Get policy and value
        action_logits = self.policy_net(features)
        state_value = self.value_net(features)

        return action_logits, state_value

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            x: Input observation

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(x)

        # Sample action
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions.

        Args:
            x: Input observation
            actions: Actions to evaluate

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(x)

        # Get probabilities
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test feature extractor
    x = torch.randn(4, 4, 256, 256).to(device)

    extractor = CNNFeatureExtractor(in_channels=4, features_dim=512).to(device)
    features = extractor(x)
    print(f"Features shape: {features.shape}")

    # Test Actor-Critic
    ac_net = ActorCriticNetwork(in_channels=4, num_actions=32).to(device)
    logits, values = ac_net(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Values shape: {values.shape}")

    # Test action sampling
    action, log_prob, value = ac_net.get_action(x)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
