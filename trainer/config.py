"""Configuration dataclasses for training, environment, and model."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EnvConfig:
    cols: int = 10
    rows: int = 20
    hidden_rows: int = 2
    next_queue_size: int = 4
    bag_type: str = "7bag"
    max_steps: int = 10000
    use_cpp_env: bool = True  # Use C++ pybind11 backend for env simulation
    reward_weights: dict = field(default_factory=lambda: {
        "w_height": 0.0, "w_holes": 0.0, "w_bumpiness": 0.0,
        "w_well": 0.0, "w_survival": 0.01, "w_death": -100.0,
        # Amplified line-clear rewards to incentivise scoring over mere survival.
        # (single, double, triple, tetris) × level
        "line_scores": (0, 150, 500, 1000, 2000),
    })


@dataclass
class NetworkConfig:
    model_size: str = "small"  # Preset key from agent.model.DUELING_PRESETS
    # CNN: "small" | "medium" | "large"
    # Transformer: "transformer_small" | "transformer_base" | "transformer_medium"
    #            | "transformer_large" | "transformer_huge" | "transformer_giant"
    #            | "transformer_mega"
    cnn_channels: int = 32
    hidden_dim: int = 128
    feature_dim: int = 53
    num_actions: int = 112
    use_noisy: bool = True
    sigma_init: float = 0.01  # Lower initial noise for conservative exploration
    sigma_decay: float = 0.9999997  # Per-step σ decay; → ~41% at 3M, ~9% at 8M, ~1% at 15.6M


@dataclass
class DQNConfig:
    gamma: float = 0.99
    n_step: int = 5
    lr: float = 2.5e-4  # 10× lower than typical — prevents online drift between hard syncs
    batch_size: int = 256
    train_every: int = 4
    target_update_freq: int = 4000  # Periodic anchor hard-sync interval (for soft-sync mode)
    target_update_tau: float = 0.001  # Polyak averaging coefficient per training step
    use_hard_update: bool = True  # Soft sync (Polyak) every step; periodic hard anchor sync
    replay_capacity: int = 2_000_000  # Larger buffer for batch_size=256 (~4.4 GB RAM)
    per_alpha: float = 0.3  # Low α → near-uniform sampling, prevents death domination
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_frames: int = 3_000_000  # β anneals to 1.0 at ~77 % of total training updates
    per_reward_weight: float = 0.5  # Reward multiplier in blended priority
    per_reward_blend: float = 0.9  # Blend: (1-b)·|td|^α + b·|reward|·w
    loss_type: str = "huber"  # "huber" (SmoothL1Loss) or "mse" (MSELoss)
    huber_beta: float = 1.0  # SmoothL1Loss beta — L1←|td|≤β→L2 transition threshold
    grad_clip_norm: float = 10.0


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 2.5e-4
    batch_size: int = 256
    mini_batch_size: int = 64
    n_epochs: int = 4
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    num_envs: int = 64


@dataclass
class TrainingConfig:
    algorithm: str = "dqn"  # "dqn" or "ppo"
    total_samples: int = 1_000_000_000  # Total training samples consumed (invariant to batch_size)
    eval_every: int = 10_000
    eval_episodes: int = 100
    save_every: int = 10_000
    log_every: int = 100
    num_envs: int = 64
    num_pretrain_episodes: int = 1000
    num_pretrain_envs: int = 16
    pretrain_sample_tag: str = "latest"  # Tag for loading / saving pretrain samples
    pretrain_epochs: int = 50
    use_pretrain: bool = True
    curriculum_stages: List[dict] = field(default_factory=list)
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_keep_best: int = 5
    checkpoint_keep_latest: int = 1
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "tetris-ai"
    wandb_entity: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # --- Derived values ---
    _total_steps_override: Optional[int] = field(default=None, repr=False)

    @property
    def total_steps(self) -> int:
        """Env steps derived from total_samples + batch_size + train_every.

        Formula (DQN): total_steps = total_samples × train_every / batch_size
        Formula (PPO): total_steps = total_samples / num_envs
        """
        if self._total_steps_override is not None:
            return self._total_steps_override
        if self.algorithm == "dqn":
            return max(1, int(self.total_samples * self.dqn.train_every / self.dqn.batch_size))
        else:
            return max(1, int(self.total_samples / self.num_envs))

    @total_steps.setter
    def total_steps(self, value: int):
        """Allow CLI --steps to override the derived value."""
        self._total_steps_override = value
