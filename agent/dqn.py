"""Rainbow DQN Agent for Tetris.

Combines:
  - Double DQN (decoupled action selection & evaluation)
  - Dueling Network (V + A decomposition)
  - Prioritized Experience Replay (TD-error based sampling)
  - Noisy Networks (learned exploration)
  - N-step TD (accelerated credit propagation)
  - Action Masking (only legal placements)

Training: off-policy, updates every `train_every` environment steps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
from copy import deepcopy

from .model import DuelingDQN
from .memory import PrioritizedReplayBuffer, UniformReplayBuffer
from .nstep_buffer import NStepBuffer
from .action_mask import create_action_mask, mask_logits, encode_action, decode_action


class RainbowDQN:
    """Rainbow DQN agent for placement-based Tetris."""

    def __init__(self,
                 num_actions: int = 112,
                 feature_dim: int = 53,
                 model_size: str = "small",
                 gamma: float = 0.99,
                 n_step: int = 5,
                 lr: float = 6.25e-5,
                 batch_size: int = 32,
                 train_every: int = 4,
                 target_update_freq: int = 8000,
                 target_update_tau: float = 0.001,
                 use_hard_update: bool = False,
                 replay_capacity: int = 1_000_000,
                 per_alpha: float = 0.3,
                 per_beta_start: float = 0.4,
                 per_beta_end: float = 1.0,
                 per_beta_frames: int = 10_000_000,
                 per_reward_weight: float = 0.5,
                 per_reward_blend: float = 0.3,
                 per_reward_clip: float = 10.0,
                 per_uniform_ratio: float = 0.2,
                 loss_type: str = "huber",
                 huber_beta: float = 1.0,
                 grad_clip_norm: float = 10.0,
                 use_noisy: bool = True,
                 sigma_init: float = 0.017,
                 sigma_decay: float = 1.0,
                 device: str = "cuda"):
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.train_every = train_every
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        self.use_hard_update = use_hard_update
        self.grad_clip_norm = grad_clip_norm
        self.loss_type = loss_type
        self.huber_beta = huber_beta
        self.sigma_decay = sigma_decay
        self._last_hard_sync_step = 0
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Networks.
        self.online_net = DuelingDQN(
            num_actions=num_actions, feature_dim=feature_dim,
            model_size=model_size,
            use_noisy=use_noisy, sigma_init=sigma_init,
        ).to(self.device)
        self.target_net = deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        # Optimizer.
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay buffer.
        self.memory = PrioritizedReplayBuffer(
            capacity=replay_capacity, alpha=per_alpha,
            beta_start=per_beta_start, beta_end=per_beta_end,
            beta_frames=per_beta_frames,
            reward_weight=per_reward_weight,
            reward_blend=per_reward_blend,
            reward_clip=per_reward_clip,
            uniform_ratio=per_uniform_ratio,
        )

        # N-step buffers (one per environment in parallel setup).
        self.nstep_buffers: Dict[int, NStepBuffer] = {}

        # Counters.
        self.train_step = 0
        self.env_step = 0

    # ------------------------------------------------------------------ #
    #  Action Selection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, board: np.ndarray, features: np.ndarray,
                      legal_actions: List, env_id: int = 0,
                      deterministic: bool = False) -> Tuple[int, int, int, bool]:
        """Select action using current online network (single env, B=1).

        Returns (rotation, column, hold, action_idx).
        Prefer ``select_actions_batch`` for multi-env setups.
        """
        mask = create_action_mask(legal_actions, self.num_actions, self.device)
        if not mask.any():
            return 0, 0, False, 0

        board_t = torch.as_tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        feat_t = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        q_values = self.online_net(board_t, feat_t).squeeze(0)
        masked_q = mask_logits(q_values, mask, -1e9)

        action_idx = int(masked_q.argmax().item())
        rotation, col, hold = decode_action(action_idx)
        return rotation, col, hold, action_idx

    @torch.no_grad()
    def select_actions_batch(self,
                              boards: np.ndarray,      # (N, 1, 22, 10)
                              features: np.ndarray,    # (N, 53)
                              legal_actions_list: List[List]
                              ) -> List[Tuple[int, int, int, int]]:
        """Select actions for N envs in a single GPU forward pass.

        Returns a list of (rotation, column, hold, action_idx) — one per env.

        This is the performance-critical path for training: a single B=N
        matmul replaces N serial B=1 kernel launches, eliminating
        CPU↔GPU synchronisation overhead.
        """
        # Build mask tensor: (N, num_actions).
        mask_list = [
            create_action_mask(la, self.num_actions, self.device)
            if la else torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)
            for la in legal_actions_list
        ]
        masks = torch.stack(mask_list)  # (N, 112)

        board_t = torch.as_tensor(boards, dtype=torch.float32, device=self.device)
        feat_t = torch.as_tensor(features, dtype=torch.float32, device=self.device)

        q_values = self.online_net(board_t, feat_t)   # (N, 112)
        masked_q = torch.where(masks, q_values, torch.full_like(q_values, -1e9))

        action_indices = masked_q.argmax(dim=-1).tolist()

        results = []
        for idx in action_indices:
            rot, col, hold = decode_action(int(idx))
            results.append((rot, col, hold, int(idx)))
        return results

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def update(self) -> Optional[Dict[str, float]]:
        """Perform one training step. Returns loss metrics dict."""
        if len(self.memory) < self.batch_size:
            return None

        batch, indices, weights = self.memory.sample(self.batch_size, self.train_step)

        # Move to device.
        board = torch.as_tensor(batch["board"], dtype=torch.float32, device=self.device)
        features = torch.as_tensor(batch["features"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_board = torch.as_tensor(batch["next_board"], dtype=torch.float32, device=self.device)
        next_features = torch.as_tensor(batch["next_features"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Double DQN: online selects, target evaluates.
            # Use gamma^n for the n-step bootstrap. When the episode ended early
            # (done=1), the reward sum already covers everything — no bootstrap.
            next_q_online = self.online_net(next_board, next_features)
            best_actions = next_q_online.argmax(dim=-1)
            next_q_target = self.target_net(next_board, next_features)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            gamma_n = self.gamma ** self.n_step
            target = rewards + gamma_n * next_q * (1.0 - dones)

        # Current Q values.
        current_q = self.online_net(board, features).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss with importance sampling weights.
        td_errors = target - current_q
        if self.loss_type == "mse":
            loss = (weights * F.mse_loss(current_q, target, reduction='none')).mean()
        else:
            loss = (weights * F.smooth_l1_loss(current_q, target, reduction='none',
                                                beta=self.huber_beta)).mean()

        # Update.
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Update priorities (hybrid: TD-error + reward).
        self.memory.update_priorities(
            indices,
            td_errors.abs().detach().cpu().numpy(),
            rewards.abs().detach().cpu().numpy(),
        )

        # Target network update.
        self.train_step += 1
        sync_event = None  # Set to step number when a hard sync occurs.

        if self.use_hard_update and self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            sync_event = self.train_step
        elif not self.use_hard_update:
            # Polyak soft update every training step.
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.data.copy_(self.target_update_tau * op.data + (1 - self.target_update_tau) * tp.data)
            # Periodic anchor hard sync (logs when it fires).
            if self.train_step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                sync_event = self.train_step

        # Scheduled sigma decay for NoisyLinear layers.
        sigma_mean = 0.0
        if self.sigma_decay < 1.0:
            for module in self.online_net.modules():
                if hasattr(module, 'scale_sigma'):
                    module.scale_sigma(self.sigma_decay)
                    sigma_mean += module.get_sigma_mean()

        # ---- metrics ----
        new_priorities = td_errors.abs().detach() ** self.memory.alpha
        reward_priorities = rewards.abs() * self.memory.reward_weight
        metrics = {
            "q_loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
            "td_error_mean": td_errors.abs().mean().item(),
            "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "w_mean": weights.mean().item(),
            "w_max": weights.max().item(),
            "prio_mean": new_priorities.mean().item(),
            "prio_max": new_priorities.max().item(),
            "rw_prio_mean": reward_priorities.mean().item(),
            "rw_prio_max": reward_priorities.max().item(),
            "reward_mean": rewards.mean().item(),
            "init_prio": float(self.memory._init_priority_for(0.0)),
        }
        if sync_event is not None:
            metrics["target_sync"] = sync_event
        if sigma_mean > 0:
            metrics["sigma_mean"] = sigma_mean
        return metrics

    # ------------------------------------------------------------------ #
    #  Experience Collection
    # ------------------------------------------------------------------ #
    def observe(self, env_id: int,
                state: Tuple[np.ndarray, np.ndarray], action_idx: int,
                reward: float, next_state: Tuple[np.ndarray, np.ndarray],
                done: bool):
        """Store transition via n-step buffer, push completed n-step returns to replay.

        state, next_state are (board, features) tuples.
        """
        if env_id not in self.nstep_buffers:
            self.nstep_buffers[env_id] = NStepBuffer(self.n_step, self.gamma)

        result = self.nstep_buffers[env_id].add(state, action_idx, reward, next_state, done)

        if result is not None:
            s, a, r, ns, d = result
            self.memory.add(s, a, r, ns, d)

        if done:
            # Drain remaining transitions on episode end.
            remaining = self.nstep_buffers[env_id].flush()
            for s, a, r, ns, d in remaining:
                self.memory.add(s, a, r, ns, d)
            self.nstep_buffers[env_id].reset()

        self.env_step += 1

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict:
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "env_step": self.env_step,
        }

    def load_state_dict(self, state: Dict):
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.train_step = state["train_step"]
        self.env_step = state["env_step"]

    def eval_mode(self):
        self.online_net.eval()

    def train_mode(self):
        self.online_net.train()
