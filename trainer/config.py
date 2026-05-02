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
        "w_height": 0.3, "w_holes": 1.5, "w_bumpiness": 0.2,
        "w_well": 0.5, "w_survival": 0.01, "w_death": -100.0,
    })


@dataclass
class NetworkConfig:
    cnn_channels: int = 32
    hidden_dim: int = 128
    feature_dim: int = 53
    num_actions: int = 112
    use_noisy: bool = True
    sigma_init: float = 0.017


@dataclass
class DQNConfig:
    gamma: float = 0.99
    n_step: int = 5
    lr: float = 6.25e-5
    batch_size: int = 32
    train_every: int = 4
    target_update_freq: int = 8000
    target_update_tau: float = 0.005
    use_hard_update: bool = True
    replay_capacity: int = 1_000_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_frames: int = 10_000_000
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
    total_steps: int = 50_000_000
    eval_every: int = 10_000
    eval_episodes: int = 100
    save_every: int = 50_000
    log_every: int = 100
    num_envs: int = 64
    num_pretrain_episodes: int = 1000
    pretrain_epochs: int = 50
    use_pretrain: bool = True
    curriculum_stages: List[dict] = field(default_factory=list)
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "tetris-ai"
    wandb_entity: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
