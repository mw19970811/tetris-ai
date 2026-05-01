"""PPO (Proximal Policy Optimization) Agent for Tetris.

On-policy alternative to DQN. More stable but less sample-efficient.
Uses GAE for advantage estimation and PPO-Clip for policy updates.

Reference: Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional
from copy import deepcopy

from .model import ActorCritic
from .action_mask import create_action_mask, mask_logits, encode_action


class PPOBuffer:
    """On-policy rollout buffer for PPO."""

    def __init__(self, num_envs: int, rollout_steps: int,
                 board_shape: Tuple = (1, 22, 10), feature_dim: int = 53):
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.board_shape = board_shape
        self.feature_dim = feature_dim

        self.reset()

    def reset(self):
        self.boards = np.zeros((self.rollout_steps, self.num_envs) + self.board_shape, dtype=np.float32)
        self.features = np.zeros((self.rollout_steps, self.num_envs, self.feature_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_steps, self.num_envs), dtype=np.int64)
        self.log_probs = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_envs), dtype=bool)
        self.masks = np.zeros((self.rollout_steps, self.num_envs, 112), dtype=bool)
        self.step = 0

    def add(self, board: np.ndarray, features: np.ndarray,
            action: int, log_prob: float, value: float,
            reward: float, done: bool, mask: np.ndarray):
        idx = self.step
        self.boards[idx] = board
        self.features[idx] = features
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.masks[idx] = mask
        self.step += 1

    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "boards": torch.as_tensor(self.boards[:self.step].reshape(-1, *self.board_shape), device=device),
            "features": torch.as_tensor(self.features[:self.step].reshape(-1, self.feature_dim), device=device),
            "actions": torch.as_tensor(self.actions[:self.step].reshape(-1), device=device, dtype=torch.long),
            "log_probs": torch.as_tensor(self.log_probs[:self.step].reshape(-1), device=device),
            "values": torch.as_tensor(self.values[:self.step].reshape(-1), device=device),
            "rewards": torch.as_tensor(self.rewards[:self.step], device=device),
            "dones": torch.as_tensor(self.dones[:self.step], device=device),
            "masks": torch.as_tensor(self.masks[:self.step].reshape(-1, 112), device=device, dtype=torch.bool),
        }

    def __len__(self):
        return self.step


class PPO:
    """PPO agent with GAE and action masking."""

    def __init__(self,
                 num_actions: int = 112,
                 feature_dim: int = 53,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 lr: float = 2.5e-4,
                 batch_size: int = 256,
                 mini_batch_size: int = 64,
                 n_epochs: int = 4,
                 max_grad_norm: float = 0.5,
                 rollout_steps: int = 2048,
                 num_envs: int = 64,
                 device: str = "cuda"):
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.network = ActorCritic(
            num_actions=num_actions, feature_dim=feature_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = PPOBuffer(num_envs, rollout_steps, feature_dim=feature_dim)

        self.train_step = 0

    # ------------------------------------------------------------------ #
    #  Action Selection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, board: np.ndarray, features: np.ndarray,
                      legal_actions: List, deterministic: bool = False
                      ) -> Tuple[int, float, float]:
        """Sample action. Returns (action_idx, log_prob, value)."""
        mask = create_action_mask(legal_actions, self.num_actions, self.device)

        board_t = torch.as_tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        feat_t = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, value = self.network(board_t, feat_t)
        masked_logits = mask_logits(logits.squeeze(0), mask, -1e9)

        dist = torch.distributions.Categorical(logits=masked_logits)
        if deterministic:
            action = masked_logits.argmax()
        else:
            action = dist.sample()

        return int(action.item()), float(dist.log_prob(action).item()), float(value.squeeze().item())

    @torch.no_grad()
    def get_value(self, board: np.ndarray, features: np.ndarray) -> float:
        board_t = torch.as_tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        feat_t = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.network.get_value(board_t, feat_t).squeeze().item())

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        """Perform PPO update using collected rollout data."""
        batch = self.buffer.get_batch(self.device)
        rollout_size = len(batch["actions"])
        if rollout_size == 0:
            return {}

        # Compute GAE advantages and returns.
        rewards = batch["rewards"]  # (T, N)
        values = batch["values"]    # (T*N,)
        dones = batch["dones"]      # (T, N)

        # Reshape for per-env GAE computation.
        values_2d = values.view(self.buffer.step, self.num_envs)
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.buffer.step)):
            next_value = values_2d[t + 1] if t + 1 < self.buffer.step else torch.zeros(self.num_envs, device=self.device)
            delta = rewards[t] + self.gamma * next_value * (~dones[t]).float() - values_2d[t]
            last_gae = delta + self.gamma * self.gae_lambda * (~dones[t]).float() * last_gae
            advantages[t] = last_gae

        returns = advantages + values_2d
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        # Normalise advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs.
        flat_boards = batch["boards"]
        flat_features = batch["features"]
        flat_actions = batch["actions"]
        flat_log_probs = batch["log_probs"]
        flat_masks = batch["masks"]

        total_samples = rollout_size
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_boards = flat_boards[mb_idx]
                mb_features = flat_features[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_masks = flat_masks[mb_idx]

                new_log_probs, new_values, entropy = self.network.evaluate_actions(
                    mb_boards, mb_features, mb_actions, mb_masks
                )

                # Policy loss (PPO-Clip).
                ratio = (new_log_probs - mb_old_log_probs).exp()
                clipped_ratio = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()

                # Value loss.
                value_loss = F.mse_loss(new_values, mb_returns)

                # Total loss.
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics.
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += (mb_old_log_probs - new_log_probs).mean().item()
                n_updates += 1

        self.buffer.reset()
        self.train_step += 1

        for k in metrics:
            metrics[k] /= max(n_updates, 1)
        return metrics

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def state_dict(self):
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }

    def load_state_dict(self, state):
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.train_step = state["train_step"]
