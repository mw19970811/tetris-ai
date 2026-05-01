"""Main training loop for Tetris RL agent.

Orchestrates: environment sampling → experience collection → model update → evaluation → checkpointing.

Supports both DQN (Rainbow) and PPO algorithms.
"""

import os
import time
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Dict, Tuple
from collections import deque


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}:{s:02d}"


def _timestamp() -> str:
    """Current time formatted for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

from .config import TrainingConfig
from .logger import Logger
from .checkpoint import CheckpointManager
from .evaluator import Evaluator


class Trainer:
    """Main trainer orchestrating the RL training pipeline."""

    def __init__(self, config: TrainingConfig, resume: bool = False,
                 resume_from: Optional[str] = None):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        # Set seeds.
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Logger.
        self.logger = Logger(
            log_dir=config.log_dir,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
        )

        # Checkpoint manager.
        self.checkpoint = CheckpointManager(config.checkpoint_dir)

        # Create environments.
        self._create_envs()

        # Create agent.
        self._create_agent()

        # Evaluator.
        self.evaluator = Evaluator(
            env_creator=self._make_env, agent=self.agent,
            num_episodes=config.eval_episodes, device=str(self.device),
        )

        # Metrics tracking.
        self.episode_rewards = deque(maxlen=100)
        self.episode_lines = deque(maxlen=100)
        self.best_avg_score = 0.0

        # Resume state.
        self.resume_step = 0
        if resume:
            loaded_step = self._resume(resume_from)
            self.resume_step = loaded_step
        elif resume_from:
            loaded_step = self._resume(resume_from)
            self.resume_step = loaded_step

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    def _make_env(self):
        """Create a fresh environment instance."""
        from env.tetris_env import TetrisEnv, EnvConfig, RewardConfig
        from env.reward_calculator import RewardConfig as RC
        rc = RC(**self.cfg.env.reward_weights) if self.cfg.env.reward_weights else RC()
        ec = EnvConfig(
            cols=self.cfg.env.cols, rows=self.cfg.env.rows,
            hidden_rows=self.cfg.env.hidden_rows,
            next_queue_size=self.cfg.env.next_queue_size,
            bag_type=self.cfg.env.bag_type,
            max_steps=self.cfg.env.max_steps,
            reward=rc,
        )
        if self.cfg.env.use_cpp_env:
            from env.bindings.cpp_env import CppTetrisEnv
            return CppTetrisEnv(ec)
        return TetrisEnv(ec)

    def _create_envs(self):
        """Create vectorised environments for parallel sampling."""
        self.num_envs = self.cfg.num_envs
        self.envs = [self._make_env() for _ in range(self.num_envs)]
        self.obs = [env.reset() for env in self.envs]

    def _create_agent(self):
        """Create the RL agent."""
        if self.cfg.algorithm == "dqn":
            from agent.dqn import RainbowDQN
            self.agent = RainbowDQN(
                num_actions=self.cfg.network.num_actions,
                feature_dim=self.cfg.network.feature_dim,
                gamma=self.cfg.dqn.gamma,
                n_step=self.cfg.dqn.n_step,
                lr=self.cfg.dqn.lr,
                batch_size=self.cfg.dqn.batch_size,
                train_every=self.cfg.dqn.train_every,
                target_update_freq=self.cfg.dqn.target_update_freq,
                target_update_tau=self.cfg.dqn.target_update_tau,
                use_hard_update=self.cfg.dqn.use_hard_update,
                replay_capacity=self.cfg.dqn.replay_capacity,
                per_alpha=self.cfg.dqn.per_alpha,
                per_beta_start=self.cfg.dqn.per_beta_start,
                per_beta_end=self.cfg.dqn.per_beta_end,
                per_beta_frames=self.cfg.dqn.per_beta_frames,
                grad_clip_norm=self.cfg.dqn.grad_clip_norm,
                use_noisy=self.cfg.network.use_noisy,
                sigma_init=self.cfg.network.sigma_init,
                device=str(self.device),
            )
        elif self.cfg.algorithm == "ppo":
            from agent.ppo import PPO
            self.agent = PPO(
                num_actions=self.cfg.network.num_actions,
                feature_dim=self.cfg.network.feature_dim,
                gamma=self.cfg.ppo.gamma,
                gae_lambda=self.cfg.ppo.gae_lambda,
                clip_epsilon=self.cfg.ppo.clip_epsilon,
                value_coef=self.cfg.ppo.value_coef,
                entropy_coef=self.cfg.ppo.entropy_coef,
                lr=self.cfg.ppo.lr,
                batch_size=self.cfg.ppo.batch_size,
                mini_batch_size=self.cfg.ppo.mini_batch_size,
                n_epochs=self.cfg.ppo.n_epochs,
                max_grad_norm=self.cfg.ppo.max_grad_norm,
                rollout_steps=self.cfg.ppo.rollout_steps,
                num_envs=self.cfg.num_envs,
                device=str(self.device),
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.cfg.algorithm}")

    def _resume(self, checkpoint_path: Optional[str] = None) -> int:
        """Resume training from a checkpoint. Returns the loaded step number."""
        if checkpoint_path and os.path.isfile(checkpoint_path):
            path = checkpoint_path
        else:
            if checkpoint_path:
                print(f"[Trainer] Checkpoint not found: {checkpoint_path}")
            path = self.checkpoint.find_latest()

        if path is None:
            print("[Trainer] No checkpoint found. Starting from scratch.")
            return 0

        print(f"[Trainer] Loading checkpoint: {path}")
        try:
            # Load main checkpoint.
            checkpoint = torch.load(path, map_location="cpu")
            step = checkpoint.get("step", 0)

            # Restore agent state.
            if "agent_state" in checkpoint:
                self.agent.load_state_dict(checkpoint["agent_state"])
                print(f"[Trainer] Restored agent state (step {step:,}).")
            elif "model_state_dict" in checkpoint:
                net = getattr(self.agent, 'online_net', None) or getattr(self.agent, 'network', None)
                if net is not None:
                    net.load_state_dict(checkpoint["model_state_dict"])
                if hasattr(self.agent, 'target_net'):
                    self.agent.target_net.load_state_dict(checkpoint["model_state_dict"])
                print(f"[Trainer] Restored model weights (step {step:,}).")
            else:
                print("[Trainer] Warning: unrecognised checkpoint format.")

            # Restore replay buffer for DQN.
            if self.cfg.algorithm == "dqn" and hasattr(self.agent, 'memory'):
                buffer_path = path.replace(".pt", "_buffer.pt")
                if os.path.exists(buffer_path):
                    try:
                        buf = torch.load(buffer_path, map_location="cpu")
                        self.agent.memory.load_state_dict(buf)
                        print(f"[Trainer] Restored replay buffer ({len(self.agent.memory):,} transitions).")
                    except Exception as e:
                        print(f"[Trainer] Warning: failed to restore replay buffer: {e}")
                else:
                    print("[Trainer] No replay buffer file found — buffer starts empty.")

            # Restore best score from evaluation history (if available).
            if "best_avg_score" in checkpoint:
                self.best_avg_score = checkpoint["best_avg_score"]

            print(f"[Trainer] Resumed at step {step:,}.")
            return step

        except Exception as e:
            print(f"[Trainer] Error loading checkpoint: {e}")
            print("[Trainer] Starting from scratch.")
            return 0

    # ------------------------------------------------------------------ #
    #  Main Training Loop
    # ------------------------------------------------------------------ #
    def train(self, total_steps: Optional[int] = None):
        """Run the main training loop."""
        total_steps = total_steps or self.cfg.total_steps
        start_step = self.resume_step

        if start_step > 0:
            print(f"[Trainer] Resuming training: step {start_step:,} → {total_steps:,}, "
                  f"algorithm={self.cfg.algorithm}, envs={self.num_envs}")
        else:
            print(f"[Trainer] Starting training: {total_steps:,} steps, "
                  f"algorithm={self.cfg.algorithm}, envs={self.num_envs}")

        # Pretraining: skip if resuming (weights already loaded).
        if start_step == 0 and self.cfg.use_pretrain and self.cfg.algorithm == "dqn":
            self._pretrain()

        # Initial evaluation: skip if resuming (saves time).
        if start_step == 0:
            print("[Trainer] Running initial evaluation...")
            eval_metrics = self.evaluator.evaluate()
            self.logger.log_eval(0, **eval_metrics)
            print(f"[Trainer] Initial score: {eval_metrics['avg_score']:,.0f}")

        # Training loop.
        t_start = time.time()
        episode_rewards = [0.0] * self.num_envs
        episode_steps = [0] * self.num_envs

        for step in range(start_step, total_steps):
            # Collect experience for each env.
            for env_id in range(self.num_envs):
                env = self.envs[env_id]
                board, features = self.obs[env_id][0], self.obs[env_id][1]
                legal_actions = env.get_legal_actions()

                if not legal_actions:
                    self.obs[env_id] = env.reset()
                    episode_rewards[env_id] = 0.0
                    episode_steps[env_id] = 0
                    continue

                # Select action.
                if self.cfg.algorithm == "dqn":
                    rot, col, hold, action_idx = self.agent.select_action(
                        board, features, legal_actions, env_id
                    )
                    from env.tetris_env import Action
                    action = Action(rot, col, hold)
                else:
                    action_idx, log_prob, value = self.agent.select_action(
                        board, features, legal_actions, env_id
                    )
                    from env.tetris_env import Action
                    from agent.action_mask import decode_action
                    rot, col, hold = decode_action(action_idx)
                    action = Action(rot, col, hold)

                # Step environment.
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_board, next_features = obs[0], obs[1]

                episode_rewards[env_id] += reward
                episode_steps[env_id] += 1

                # Store experience (tuples avoid dict allocation overhead).
                state_tuple = (board, features)
                next_state_tuple = (next_board, next_features)

                if self.cfg.algorithm == "dqn":
                    self.agent.observe(env_id, state_tuple, action_idx,
                                       reward, next_state_tuple, done)
                else:
                    # PPO: store in rollout buffer.
                    mask = env.get_legal_actions_mask(self.cfg.network.num_actions)
                    self.agent.buffer.add(
                        board, features, action_idx, log_prob, value,
                        reward, done, mask
                    )

                self.obs[env_id] = obs

                # On episode end.
                if done:
                    self.episode_rewards.append(episode_rewards[env_id])
                    self.episode_lines.append(info.get("lines", 0))
                    episode_rewards[env_id] = 0.0
                    episode_steps[env_id] = 0

                    # Reset environment.
                    self.obs[env_id] = env.reset()

            # Training update.
            if self.cfg.algorithm == "dqn":
                if self.agent.env_step % self.agent.train_every == 0:
                    metrics = self.agent.update()
                    if metrics and self.agent.train_step % self.cfg.log_every == 0:
                        self.logger.log_train(self.agent.env_step, **metrics)
            else:
                # PPO: update after collecting full rollout.
                if len(self.agent.buffer) >= self.cfg.ppo.rollout_steps:
                    metrics = self.agent.update()
                    if metrics:
                        self.logger.log_train(self.agent.env_step, **metrics)

            # Log progress.
            if step % self.cfg.log_every == 0 and step > 0:
                elapsed = time.time() - t_start
                progress = (step - start_step) / max(total_steps - start_step, 1)
                fps = (step - start_step) * self.num_envs / elapsed if elapsed > 0 else 0
                eta = (elapsed / progress) - elapsed if progress > 0 else 0
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                print(f"\r[{_timestamp()}]  "
                      f"Step {step:>9,}/{total_steps:,} ({progress:.1%})  "
                      f"|  Avg100R: {avg_reward:>10,.1f}  "
                      f"|  FPS: {fps:>8,.0f}  "
                      f"|  Elapsed: {_fmt_duration(elapsed)}  "
                      f"|  ETA: {_fmt_duration(eta)}", end="")

            # Evaluation.
            if step % self.cfg.eval_every == 0 and step > 0:
                print()  # newline
                eval_metrics = self.evaluator.evaluate()
                self.logger.log_eval(step, **eval_metrics)
                elapsed = time.time() - t_start
                print(f"[{_timestamp()}]  "
                      f"[Eval  @ step {step:>9,}]  "
                      f"Avg: {eval_metrics['avg_score']:>12,.1f}  "
                      f"Max: {eval_metrics['max_score']:>12,.1f}  "
                      f"Lines: {eval_metrics['avg_lines']:>8,.1f}  "
                      f"|  Elapsed: {_fmt_duration(elapsed)}")

                # Save best model.
                if eval_metrics["avg_score"] > self.best_avg_score:
                    self.best_avg_score = eval_metrics["avg_score"]
                    self.checkpoint.save_full(
                        step, self.agent,
                        replay_buffer=self.agent.memory if hasattr(self.agent, 'memory') else None
                    )

            # Periodic checkpoint.
            if step % self.cfg.save_every == 0 and step > 0:
                self.checkpoint.save_full(
                    step, self.agent,
                    replay_buffer=self.agent.memory if hasattr(self.agent, 'memory') else None
                )

        # Final save.
        print("\n[Trainer] Saving final model...")
        self.checkpoint.save_full(total_steps, self.agent,
                                  replay_buffer=self.agent.memory if hasattr(self.agent, 'memory') else None)
        self.logger.save_metrics()
        self.logger.close()

        total_elapsed = time.time() - t_start
        print(f"[{_timestamp()}]  "
              f"[Trainer] Done.  "
              f"Total steps: {total_steps:,}  "
              f"Best avg score: {self.best_avg_score:,.1f}  "
              f"Duration: {_fmt_duration(total_elapsed)}")

    # ------------------------------------------------------------------ #
    #  Pretraining
    # ------------------------------------------------------------------ #
    def _pretrain(self):
        """Pretrain using Dellacherie behavior cloning."""
        print(f"[Pretrain] Collecting {self.cfg.num_pretrain_episodes} episodes from Dellacherie expert...")

        from agent.pretrain import Pretrainer
        pretrainer = Pretrainer(model_type="dqn", num_actions=self.cfg.network.num_actions,
                                feature_dim=self.cfg.network.feature_dim, device=str(self.device))

        env = self._make_env()
        boards, features, actions = pretrainer.collect_dataset(env, self.cfg.num_pretrain_episodes)

        print(f"[Pretrain] Collected {len(actions):,} transitions. Training BC...")
        state_dict = pretrainer.train(boards, features, actions,
                                       epochs=self.cfg.pretrain_epochs)

        # Load pretrained weights into agent.
        if self.cfg.algorithm == "dqn" and hasattr(self.agent, 'online_net'):
            # The pretrainer uses standard nn.Linear (use_noisy=False).
            # The DQN agent uses NoisyLinear (use_noisy=True) for value_fc
            # and advantage_fc. Only those layers need key conversion;
            # CNN / MLP / fusion layers use standard Linear — pass through.
            noisy_prefixes = ('value_fc.', 'advantage_fc.')
            converted = {}
            sigma_init = self.cfg.network.sigma_init
            for key, tensor in state_dict.items():
                is_noisy = any(key.startswith(p) for p in noisy_prefixes)
                if is_noisy and key.endswith('.weight'):
                    converted[key.replace('.weight', '.weight_mu')] = tensor
                    converted[key.replace('.weight', '.weight_sigma')] = torch.full_like(tensor, sigma_init)
                elif is_noisy and key.endswith('.bias'):
                    converted[key.replace('.bias', '.bias_mu')] = tensor
                    converted[key.replace('.bias', '.bias_sigma')] = torch.full_like(tensor, sigma_init)
                else:
                    converted[key] = tensor

            self.agent.online_net.load_state_dict(converted)
            self.agent.target_net.load_state_dict(converted)
            print("[Pretrain] Loaded pretrained weights into online + target networks.")


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
def create_trainer(config_dict: Optional[dict] = None) -> Trainer:
    """Factory to create Trainer from a config dict (for Hydra integration)."""
    if config_dict is None:
        config = TrainingConfig()
    else:
        config = TrainingConfig(**config_dict)
    return Trainer(config)


def main():
    """CLI entry point for training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Tetris RL Agent")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from latest checkpoint in --checkpoint-dir")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a specific checkpoint file")
    args = parser.parse_args()

    config = TrainingConfig(
        algorithm=args.algo,
        total_steps=args.steps,
        num_envs=args.envs,
        device=args.device,
        seed=args.seed,
        use_wandb=args.wandb,
        use_pretrain=not args.no_pretrain,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = Trainer(config, resume=args.resume, resume_from=args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
