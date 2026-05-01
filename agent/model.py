"""Neural network architectures for Tetris RL agent.

Provides:
  - DuelingDQN: CNN + MLP hybrid backbone with Dueling Head (primary).
  - ActorCritic: Shared backbone with separate policy + value heads (for PPO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .noisy_layers import NoisyLinear


# ------------------------------------------------------------------ #
#  CNN Backbone
# ------------------------------------------------------------------ #
class CNNBackbone(nn.Module):
    """Lightweight CNN for processing the 22x10 board bitmask."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 output_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 2, output_dim)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        # board: (B, 1, 22, 10)
        x = self.conv(board)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ------------------------------------------------------------------ #
#  MLP Backbone (handcrafted features)
# ------------------------------------------------------------------ #
class MLPBackbone(nn.Module):
    """MLP for 53-dim handcrafted feature vector."""

    def __init__(self, input_dim: int = 53, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ------------------------------------------------------------------ #
#  Dueling DQN (primary model for Rainbow DQN)
# ------------------------------------------------------------------ #
class DuelingDQN(nn.Module):
    """Dueling Network with Noisy Linear layers for exploration.

    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

    Architecture:
        board (B,1,22,10) ──▶ CNN ──┐
                                      ├── Concat ──▶ Dueling Head ──▶ Q(s,a)
        features (B,53)    ──▶ MLP ──┘
    """

    def __init__(self, num_actions: int = 112, feature_dim: int = 53,
                 cnn_channels: int = 32, hidden_dim: int = 128,
                 use_noisy: bool = True, sigma_init: float = 0.017):
        super().__init__()
        self.num_actions = num_actions
        self.use_noisy = use_noisy

        self.cnn = CNNBackbone(in_channels=1, base_channels=cnn_channels, output_dim=64)
        self.mlp = MLPBackbone(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=64)

        # Shared fusion layer.
        fusion_in = 64 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Dueling heads.
        linear_cls = NoisyLinear if use_noisy else nn.Linear
        linear_kwargs = {"sigma_init": sigma_init} if use_noisy else {}

        self.value_fc = nn.Sequential(
            linear_cls(hidden_dim, hidden_dim // 2, **linear_kwargs),
            nn.ReLU(inplace=True),
            linear_cls(hidden_dim // 2, 1, **linear_kwargs),
        )
        self.advantage_fc = nn.Sequential(
            linear_cls(hidden_dim, hidden_dim // 2, **linear_kwargs),
            nn.ReLU(inplace=True),
            linear_cls(hidden_dim // 2, num_actions, **linear_kwargs),
        )

    def forward(self, board: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions. (B, num_actions)"""
        cnn_out = self.cnn(board)
        mlp_out = self.mlp(features)
        fused = self.fusion(torch.cat([cnn_out, mlp_out], dim=-1))

        value = self.value_fc(fused)                       # (B, 1)
        advantage = self.advantage_fc(fused)               # (B, num_actions)
        # Mean-subtracted advantage for identifiability.
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        """Re-sample noise in all NoisyLinear layers (for exploration)."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m._sample_noise(next(m.parameters()).device)


# ------------------------------------------------------------------ #
#  Actor-Critic Network (for PPO)
# ------------------------------------------------------------------ #
class ActorCritic(nn.Module):
    """Shared backbone with separate policy (actor) and value (critic) heads.

    For PPO: outputs action logits and state value V(s).
    """

    def __init__(self, num_actions: int = 112, feature_dim: int = 53,
                 cnn_channels: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.num_actions = num_actions

        self.cnn = CNNBackbone(in_channels=1, base_channels=cnn_channels, output_dim=64)
        self.mlp = MLPBackbone(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=64)

        self.fusion = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Actor head: outputs action logits.
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Critic head: outputs scalar V(s).
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, board: torch.Tensor, features: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, value)."""
        cnn_out = self.cnn(board)
        mlp_out = self.mlp(features)
        fused = self.fusion(torch.cat([cnn_out, mlp_out], dim=-1))

        logits = self.actor(fused)
        value = self.critic(fused)
        return logits, value

    def get_policy(self, board: torch.Tensor, features: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None
                   ) -> torch.distributions.Categorical:
        """Return action distribution, optionally masked."""
        logits, _ = self.forward(board, features)
        if action_mask is not None:
            logits = torch.where(action_mask, logits, torch.full_like(logits, -1e9))
        return torch.distributions.Categorical(logits=logits)

    def get_value(self, board: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(board, features)
        return value

    def evaluate_actions(self, board: torch.Tensor, features: torch.Tensor,
                         actions: torch.Tensor, action_mask: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_probs, values, entropy) for given actions."""
        logits, values = self.forward(board, features)
        if action_mask is not None:
            logits = torch.where(action_mask, logits, torch.full_like(logits, -1e9))
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, values.squeeze(-1), entropy


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #
def create_model(model_type: str = "dueling_dqn", **kwargs) -> nn.Module:
    """Convenience factory to create models."""
    if model_type == "dueling_dqn":
        return DuelingDQN(**kwargs)
    elif model_type == "actor_critic":
        return ActorCritic(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
