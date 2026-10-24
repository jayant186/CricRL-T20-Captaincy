"""
Purpose
Implements a Deep Q-Network (DQN) agent with experience replay, target network stabilization, and Double DQN (DDQN) target calculation to make ball-by-ball tactical decisions during a T20 chase.
Architecture
- Online Q-Network: 6 -> 128 -> ReLU -> 128 -> ReLU -> 3
- Target Q-Network: Identical architecture, updated periodically or via Polyak averaging
- Action Selection: Epsilon-greedy exploration with linear or exponential decay
- Double DQN Target: y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))
  Mitigates positive maximization bias common in standard DQN.
- Loss: Smooth L1 (Huber) loss with gradient clipping for training stability.
- Experience Replay: Circular buffer backed by contiguous NumPy arrays.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. Neural Network Q-Function
class QNetwork(nn.Module):
    """ Multilayer Perceptron estimating action-values Q(s, a) for all actions """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)

# 2. Experience Replay Buffer
class ReplayBuffer:
    """ Fast ring buffer backed by contiguous NumPy arrays for storing transitions """
    def __init__(self, capacity: int = 100_000, state_dim: int = 6):
        self.capacity = int(capacity)
        self.state_dim = state_dim

        self.obs_buf = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.obs_buf[self._ptr] = obs
        self.action_buf[self._ptr] = action
        self.reward_buf[self._ptr] = reward
        self.next_obs_buf[self._ptr] = next_obs
        self.done_buf[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions converted to PyTorch tensors."""
        indices = np.random.randint(0, self._size, size=batch_size)

        obs = torch.as_tensor(self.obs_buf[indices], device=device, dtype=torch.float32)
        actions = torch.as_tensor(
            self.action_buf[indices], device=device, dtype=torch.int64
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            self.reward_buf[indices], device=device, dtype=torch.float32
        ).unsqueeze(1)
        next_obs = torch.as_tensor(
            self.next_obs_buf[indices], device=device, dtype=torch.float32
        )
        dones = torch.as_tensor(
            self.done_buf[indices], device=device, dtype=torch.float32
        ).unsqueeze(1)

        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return self._size

# 3. DQN Agent
class DQNAgent:
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: tuple[int, ...] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        tau: float = 1.0,  # 1.0 = hard update every target_update_freq, <1.0 = soft Polyak
        double_dqn: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.double_dqn = double_dqn

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self._total_steps = 0

        # Device selection: default to cpu for stable, fast small-MLP training
        if device is None or device == "auto":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer & Replay Buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """ Select an action using epsilon-greedy exploration """
        if not eval_mode:
            self._total_steps += 1
            # Linear epsilon decay
            fraction = min(1.0, self._total_steps / max(1, self.epsilon_decay_steps))
            self.epsilon = self.epsilon_start - fraction * (
                self.epsilon_start - self.epsilon_end
            )

        if not eval_mode and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """ Compute predicted Q-values for all discrete actions """
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return self.q_network(state_tensor).squeeze(0).cpu().numpy()

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> Optional[float]:
        """ Sample a batch from replay buffer and perform one gradient step """
        if len(self.replay_buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q-values: Q(s, a)
        current_q = self.q_network(obs).gather(1, actions)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action chosen by online network, value evaluated by target network
                next_actions = self.q_network(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_obs).gather(1, next_actions)
            else:
                # Standard DQN
                next_q = self.target_network(next_obs).max(dim=1, keepdim=True)[0]

            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # Smooth L1 (Huber) loss
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Target network update
        if self.tau < 1.0:
            # Soft Polyak update
            for target_param, online_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                )
        elif self._total_steps % self.target_update_freq == 0:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save(self, file_path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self._total_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(checkpoint, file_path)

    def load(self, file_path: Union[str, Path]) -> None:
        """Load model checkpoint"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {file_path}")
        checkpoint = torch.load(file_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self._total_steps = checkpoint.get("total_steps", 0)