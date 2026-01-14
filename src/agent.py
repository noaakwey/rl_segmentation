"""
RL Agent implementation for crop segmentation.
Includes PPO (Proximal Policy Optimization) agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

from src.models import ActorCriticNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    """

    def __init__(self):
        """Initialize buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self,
            obs: np.ndarray,
            action: int,
            reward: float,
            value: float,
            log_prob: float,
            done: bool):
        """
        Add transition to buffer.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self) -> Dict:
        """
        Get all data from buffer.

        Returns:
            Dictionary with all rollout data
        """
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }

    def clear(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self):
        """Get buffer size."""
        return len(self.observations)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(self,
                 in_channels: int = 4,
                 num_actions: int = 32,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cuda'):
        """
        Initialize PPO agent.

        Args:
            in_channels: Number of input channels
            num_actions: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Initialize network
        self.network = ActorCriticNetwork(
            in_channels=in_channels,
            num_actions=num_actions
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }

        logger.info(f"PPO Agent initialized on {self.device}")

    def select_action(self, observation: Dict) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            observation: Environment observation

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation['image']).unsqueeze(0).to(self.device)

        # Get action
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(obs_tensor)

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    def compute_gae(self,
                   rewards: np.ndarray,
                   values: np.ndarray,
                   dones: np.ndarray,
                   next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards array
            values: Value estimates
            dones: Done flags
            next_value: Value of next state

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        # Add next value
        values = np.append(values, next_value)

        # Compute GAE
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_gae = 0

            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        # Compute returns
        returns = advantages + values[:-1]

        return advantages, returns

    def update(self,
              n_epochs: int = 10,
              batch_size: int = 64) -> Dict:
        """
        Update policy using PPO.

        Args:
            n_epochs: Number of optimization epochs
            batch_size: Mini-batch size

        Returns:
            Dictionary with training statistics
        """
        # Get rollout data
        data = self.buffer.get()

        observations = data['observations']
        actions = data['actions']
        old_log_probs = data['log_probs']
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Training loop
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }

        for epoch in range(n_epochs):
            # Generate random indices for mini-batches
            indices = np.arange(len(observations))
            np.random.shuffle(indices)

            for start_idx in range(0, len(observations), batch_size):
                end_idx = min(start_idx + batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]

                # Get mini-batch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                       self.value_coef * value_loss +
                       self.entropy_coef * entropy_loss)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                epoch_stats['policy_loss'].append(policy_loss.item())
                epoch_stats['value_loss'].append(value_loss.item())
                epoch_stats['entropy'].append(-entropy_loss.item())
                epoch_stats['total_loss'].append(loss.item())

        # Average statistics
        stats = {
            'policy_loss': np.mean(epoch_stats['policy_loss']),
            'value_loss': np.mean(epoch_stats['value_loss']),
            'entropy': np.mean(epoch_stats['entropy']),
            'total_loss': np.mean(epoch_stats['total_loss'])
        }

        # Clear buffer
        self.buffer.clear()

        return stats

    def save(self, path: str):
        """
        Save agent to file.

        Args:
            path: Save path
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """
        Load agent from file.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        logger.info(f"Agent loaded from {path}")


class DQNAgent:
    """
    Deep Q-Network (DQN) agent as alternative to PPO.
    [EXPERIMENTAL] This agent is currently a placeholder for future research.
    """

    def __init__(self,
                 in_channels: int = 4,
                 num_actions: int = 32,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 device: str = 'cuda'):
        """
        Initialize DQN agent.

        Args:
            in_channels: Number of input channels
            num_actions: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions

        # Q-network
        from src.models import CNNFeatureExtractor
        self.q_network = nn.Sequential(
            CNNFeatureExtractor(in_channels, 512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        ).to(self.device)

        # Target network
        self.target_network = nn.Sequential(
            CNNFeatureExtractor(in_channels, 512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        logger.info(f"DQN Agent initialized on {self.device}")

    def select_action(self, observation: Dict, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        obs_tensor = torch.FloatTensor(observation['image']).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.argmax().item()

    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    # Test PPO agent
    agent = PPOAgent(in_channels=4, num_actions=32)

    # Create dummy observation
    obs = {
        'image': np.random.rand(4, 256, 256).astype(np.float32),
        'step': np.array([0])
    }

    # Select action
    action, log_prob, value = agent.select_action(obs)
    print(f"Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
