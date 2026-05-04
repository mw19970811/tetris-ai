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
#  Architecture Presets
# ------------------------------------------------------------------ #
# Each preset defines the canonical dimensions for a DuelingDQN variant.
#
# CNN-based presets use:
#   cnn_channels  — CNN base channel count (output dim = cnn_channels × 2)
#   hidden_dim    — MLP hidden & fusion & dueling head width
#
# Transformer-based presets use:
#   d_model       — column embedding & transformer hidden dimension
#   num_layers    — TransformerEncoder layers (depth)
#   num_heads     — attention heads per layer
#   ff_dim        — feed-forward hidden dimension
#   hidden_dim    — MLP backbone & fusion & dueling head width
#
DUELING_PRESETS = {
    # ==================================================================
    #  CNN-based
    # ==================================================================
    "small": {
        "cnn_channels": 32,
        "hidden_dim": 128,
    },
    "medium": {
        "cnn_channels": 64,
        "hidden_dim": 256,
    },
    "large": {
        "cnn_channels": 128,
        "hidden_dim": 512,
    },

    # ==================================================================
    #  Transformer-based  (pre-LN, global residual, scaled init)
    # ==================================================================
    "transformer_small": {      # ~0.67M  —  baseline sanity check
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 512,
        "hidden_dim": 256,
    },
    "transformer_base": {       # ~5.7M   —  first serious transformer
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "ff_dim": 1024,
        "hidden_dim": 512,
    },
    "transformer_medium": {     # ~23M    —  deeper feature hierarchy
        "d_model": 384,
        "num_layers": 12,
        "num_heads": 8,
        "ff_dim": 1536,
        "hidden_dim": 768,
    },
    "transformer_large": {      # ~104M   —  ~100M-class
        "d_model": 640,
        "num_layers": 20,
        "num_heads": 10,
        "ff_dim": 2560,
        "hidden_dim": 1280,
    },
    "transformer_huge": {       # ~260M   —  deep column-reasoning tower
        "d_model": 896,
        "num_layers": 28,
        "num_heads": 14,
        "ff_dim": 3584,
        "hidden_dim": 1792,
    },
    "transformer_giant": {      # ~418M   —  ~500M-class
        "d_model": 1024,
        "num_layers": 32,
        "num_heads": 16,
        "ff_dim": 4096,
        "hidden_dim": 2048,
    },
    "transformer_mega": {       # ~967M   —  ~1B-class
        "d_model": 1280,
        "num_layers": 48,
        "num_heads": 16,
        "ff_dim": 5120,
        "hidden_dim": 2560,
    },
}

# Backwards-compat alias.
DUELING_PRESETS["transformer"] = DUELING_PRESETS["transformer_small"]


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
#  Board Column Transformer
# ------------------------------------------------------------------ #
class BoardColumnTransformer(nn.Module):
    """Process the Tetris board as a sequence of columns via self-attention.

    Each of the 10 columns (height 22) is projected to a d_model token;
    a TransformerEncoder learns cross-column interactions — e.g. which
    column pairs form line-clear opportunities.

    Stability features for deep RL:
      - Pre-LN  (norm_first=True) — stabilises gradient flow in deep stacks
      - Global residual — projected input bypasses all layers, then added
        back and LayerNorm'd before output
      - Scaled init — linear projections are initialised with std scaled by
        depth to prevent exploding activations in deep (>12L) configs
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4,
                 num_layers: int = 2, ff_dim: int = 512,
                 dropout: float = 0.0, global_residual: bool = True):
        super().__init__()
        self.global_residual = global_residual

        self.col_proj = nn.Linear(22, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 10, d_model))

        # Scale initialisation by depth for stable training of deep nets.
        init_scale = (2 * num_layers) ** -0.25 if num_layers > 8 else 1.0
        nn.init.normal_(self.col_proj.weight, std=0.02 * init_scale)
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-LN: critical for deep (>6L) stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,  # disabled when norm_first=True
        )

        # DeepNorm-style: scale residual branches by depth.
        if num_layers > 12:
            for layer in self.encoder.layers:
                layer.dropout.p = 0.0   # Dropout is RL-stability enemy
                # Scale down the FFN output before residual add.
                layer.norm2.weight.data.mul_(init_scale)
                layer.norm2.bias.data.mul_(init_scale)

        if global_residual:
            self.norm_out = nn.LayerNorm(d_model)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        # board: (B, 1, 22, 10) → (B, 10, 22)
        x = board.squeeze(1).permute(0, 2, 1)
        x = self.col_proj(x)                          # (B, 10, d_model)
        x = x + self.pos_embed

        if self.global_residual:
            x = self.norm_out(x + self.encoder(x))    # global residual
        else:
            x = self.encoder(x)

        return x.mean(dim=1)                          # (B, d_model)


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

    Architecture (CNN/MLP):
        board (B,1,22,10) ──▶ CNN ──┐
                                      ├── Concat ──▶ Dueling Head ──▶ Q(s,a)
        features (B,53)    ──▶ MLP ──┘

    Architecture (transformer):
        board (B,1,22,10) ──▶ ColumnTransformer ──┐
                                                     ├── Concat ──▶ Dueling Head ──▶ Q(s,a)
        features (B,53)    ──▶ MLP ─────────────────┘

    model_size presets (see DUELING_PRESETS for full details):

        CNN-based:
          small    — CNN  32ch, hidden 128    (~0.14M)
          medium   — CNN  64ch, hidden 256    (~0.51M)
          large    — CNN 128ch, hidden 512    (~1.96M)

        Transformer-based:
          transformer_small   —  d=128, L=2   (~0.67M)   baseline
          transformer_base    —  d=256, L=6   (~5.7M)    first serious
          transformer_medium  —  d=384, L=12  (~23M)
          transformer_large   —  d=640, L=20  (~104M)    ~100M-class
          transformer_huge    —  d=896, L=28  (~260M)    deep tower
          transformer_giant   — d=1024, L=32  (~418M)    ~500M-class
          transformer_mega    — d=1280, L=48  (~967M)    ~1B-class
    """

    def __init__(self, num_actions: int = 112, feature_dim: int = 53,
                 model_size: str = "small",
                 use_noisy: bool = True, sigma_init: float = 0.017):
        super().__init__()
        self.num_actions = num_actions
        self.use_noisy = use_noisy
        self.model_size = model_size

        cfg = DUELING_PRESETS[model_size]
        hidden_dim = cfg["hidden_dim"]
        is_transformer = "d_model" in cfg

        # Board encoder: CNN or ColumnTransformer.
        if is_transformer:
            self.board_encoder = BoardColumnTransformer(
                d_model=cfg["d_model"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                ff_dim=cfg["ff_dim"],
            )
            embed_dim = cfg["d_model"]
        else:
            cnn_channels = cfg["cnn_channels"]
            embed_dim = cnn_channels * 2
            self.board_encoder = CNNBackbone(
                in_channels=1, base_channels=cnn_channels, output_dim=embed_dim,
            )

        # MLP for handcrafted features (output dim matches board encoder).
        self.mlp = MLPBackbone(
            input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=embed_dim,
        )

        # Shared fusion layer.
        fusion_in = embed_dim * 2
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
        board_enc = self.board_encoder(board)
        mlp_out = self.mlp(features)
        fused = self.fusion(torch.cat([board_enc, mlp_out], dim=-1))

        value = self.value_fc(fused)                       # (B, 1)
        advantage = self.advantage_fc(fused)               # (B, num_actions)
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
def create_model(model_type: str = "dueling_dqn", model_size: str = "small",
                 **kwargs) -> nn.Module:
    """Convenience factory to create models."""
    if model_type == "dueling_dqn":
        return DuelingDQN(model_size=model_size, **kwargs)
    elif model_type == "actor_critic":
        return ActorCritic(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
