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

# Module-level imports to avoid per-iteration import overhead.
from env.tetris_env import Action
from agent.action_mask import decode_action

# ANSI colours for inline health indicators.
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


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
from .logger import Logger, TeeLogger
from .checkpoint import CheckpointManager
from .evaluator import Evaluator
from .profiler import TrainingProfiler
from agent.dellacherie import DellacherieAgent


class Trainer:
    """Main trainer orchestrating the RL training pipeline."""

    def __init__(self, config: TrainingConfig, resume: bool = False,
                 resume_from: Optional[str] = None,
                 profile: bool = False):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.profiler = TrainingProfiler(enabled=profile)
        print(f"[Trainer] Using device: {self.device}")
        if profile:
            print("[Trainer] Profiling enabled — cumulative phase timings will be reported.")

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
        self.checkpoint = CheckpointManager(
            config.checkpoint_dir,
            keep_best=config.checkpoint_keep_best,
            keep_latest=config.checkpoint_keep_latest,
        )

        # Create environments.
        self._create_envs()

        # Create agent.
        self._create_agent()

        # Print model summary.
        self._print_model_summary()

        # Evaluator.
        self.evaluator = Evaluator(
            env_creator=self._make_env, agent=self.agent,
            num_episodes=config.eval_episodes, device=str(self.device),
        )

        # Metrics tracking.
        self.episode_rewards = deque(maxlen=100)
        self.episode_lines = deque(maxlen=100)
        self.episode_pieces = deque(maxlen=100)
        self.episode_scores = deque(maxlen=100)
        self.episode_steps = deque(maxlen=100)
        self.best_avg_score = 0.0
        self._last_ckpt_metrics: Dict = {}  # Previous checkpoint's eval for delta comparison
        self._last_train_metrics: Dict = {}  # Latest training step metrics for progress line

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
                model_size=self.cfg.network.model_size,
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
                per_reward_weight=self.cfg.dqn.per_reward_weight,
                per_reward_blend=self.cfg.dqn.per_reward_blend,
                loss_type=self.cfg.dqn.loss_type,
                huber_beta=self.cfg.dqn.huber_beta,
                grad_clip_norm=self.cfg.dqn.grad_clip_norm,
                use_noisy=self.cfg.network.use_noisy,
                sigma_init=self.cfg.network.sigma_init,
                sigma_decay=self.cfg.network.sigma_decay,
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

    # ------------------------------------------------------------------ #
    #  Model summary
    # ------------------------------------------------------------------ #
    def _print_model_summary(self):
        """Print DQN / PPO network structure and parameter breakdown."""
        net = getattr(self.agent, 'online_net', None) or getattr(self.agent, 'network', None)
        if net is None:
            return

        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        size_tag = f"{total/1e6:.1f}M" if total >= 1e6 else f"{total/1e3:.0f}K"

        print("=" * 60)
        print(f"  Model: {type(net).__name__}  ({size_tag} params, {trainable/1e6:.1f}M trainable)")
        print("=" * 60)

        # Board encoder info.
        encoder = getattr(net, 'board_encoder', None)
        if encoder is not None:
            etype = type(encoder).__name__
            eparams = sum(p.numel() for p in encoder.parameters())
            print(f"  Board encoder: {etype}  ({eparams/1e6:.2f}M params)")
            if hasattr(encoder, 'encoder'):
                n_layers = len(encoder.encoder.layers)
                d_model = encoder.encoder.layers[0].self_attn.embed_dim
                n_heads = encoder.encoder.layers[0].self_attn.num_heads
                ff_dim = encoder.encoder.layers[0].linear1.out_features
                pre_ln = encoder.encoder.layers[0].norm_first
                gr = getattr(encoder, 'global_residual', False)
                print(f"    d_model={d_model}  layers={n_layers}  heads={n_heads}"
                      f"  ff_dim={ff_dim}")
                print(f"    pre-LN={pre_ln}  global_residual={gr}")
            elif hasattr(encoder, 'conv'):
                channels = []
                for m in encoder.conv:
                    if isinstance(m, torch.nn.Conv2d):
                        channels.append(m.out_channels)
                print(f"    CNN channels: {channels}")

        # MLP backbone.
        mlp = getattr(net, 'mlp', None)
        if mlp is not None:
            mparams = sum(p.numel() for p in mlp.parameters())
            print(f"  MLP backbone: {mparams/1e3:.1f}K params")

        # Fusion layer.
        fusion = getattr(net, 'fusion', None)
        if fusion is not None:
            fparams = sum(p.numel() for p in fusion.parameters())
            print(f"  Fusion: {fparams/1e3:.1f}K params")

        # Dueling heads.
        v_params = sum(p.numel() for p in net.value_fc.parameters())
        a_params = sum(p.numel() for p in net.advantage_fc.parameters())
        print(f"  Value head:      {v_params/1e3:.1f}K params")
        print(f"  Advantage head:  {a_params/1e3:.1f}K params")

        # Layer table.
        print(f"\n  {'Layer':<32s} {'Output':<22s} {'Params':>10s}")
        print(f"  {'─'*32} {'─'*22} {'─'*10}")
        for name, module in net.named_children():
            n_p = sum(p.numel() for p in module.parameters())
            # Derive output shape.
            shape_str = ""
            if hasattr(module, 'out_features'):
                shape_str = f"({module.out_features},)"
            elif hasattr(module, 'num_actions'):
                shape_str = f"({module.num_actions} actions)"
            elif isinstance(module, (torch.nn.Sequential,)):
                # Pick last child's output.
                last = list(module.children())[-1] if list(module.children()) else None
                if hasattr(last, 'out_features'):
                    shape_str = f"({last.out_features},)"
                elif hasattr(last, 'num_actions'):
                    shape_str = f"({last.num_actions} actions)"
            if n_p >= 1e6:
                ptag = f"{n_p/1e6:.2f}M"
            elif n_p >= 1e3:
                ptag = f"{n_p/1e3:.1f}K"
            else:
                ptag = str(n_p)
            print(f"  {name:<32s} {shape_str:<22s} {ptag:>10s}")

        # Total.
        buf_size = getattr(getattr(self.agent, 'memory', None), 'capacity', 0)
        print(f"  {'─'*32} {'─'*22} {'─'*10}")
        print(f"  {'Total':<32s} {'':<22s} {size_tag:>10s}")
        if buf_size:
            print(f"\n  Replay buffer capacity: {buf_size:,}")
        print("=" * 60)

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
                        buf = torch.load(buffer_path, map_location="cpu", weights_only=False)
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

        # Tee stdout to log file.
        tee = TeeLogger(self.cfg.log_dir)
        print(f"[{_timestamp()}] Log file: {tee.path}")

        if start_step > 0:
            print(f"[Trainer] Resuming: step {start_step:,} → {total_steps:,}  "
                  f"({self.cfg.total_samples:,} samples)  "
                  f"algorithm={self.cfg.algorithm}  envs={self.num_envs}")
        else:
            print(f"[Trainer] Starting: {total_steps:,} env steps "
                  f"({self.cfg.total_samples:,} training samples)  "
                  f"algorithm={self.cfg.algorithm}  envs={self.num_envs}")

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
        dead_count = 0               # Cumulative dead-position counter

        for step in range(start_step, total_steps):
            # === Phase 1: Collect states + legal actions for all envs ===
            boards_list = []
            features_list = []
            legal_list = []

            for env_id in range(self.num_envs):
                board, features = self.obs[env_id][0], self.obs[env_id][1]

                with self.profiler.phase("legal_actions"):
                    legal = self.envs[env_id].get_legal_actions()

                if not legal:
                    # Dead position: no legal moves remain.  Log the board
                    # state that caused death, then reset for batch continuity.
                    dead_count += 1
                    # Capture death context before reset.
                    dead_board = board  # (1, 22, 10) — raw board at death
                    dead_height = float(dead_board.sum())  # blocks on board
                    with self.profiler.phase("env_reset"):
                        self.obs[env_id] = self.envs[env_id].reset()
                    episode_rewards[env_id] = 0.0
                    episode_steps[env_id] = 0
                    board, features = self.obs[env_id]
                    with self.profiler.phase("legal_actions"):
                        legal = self.envs[env_id].get_legal_actions()

                boards_list.append(board)
                features_list.append(features)
                legal_list.append(legal)

            stacked_boards = np.array(boards_list)      # (N, 1, 22, 10)
            stacked_features = np.array(features_list)  # (N, 53)

            # === Phase 2: Single batched GPU forward ===
            with self.profiler.phase("action_select"):
                if self.cfg.algorithm == "dqn":
                    batched_actions = self.agent.select_actions_batch(
                        stacked_boards, stacked_features, legal_list
                    )
                else:
                    batched_actions = self.agent.select_actions_batch(
                        stacked_boards, stacked_features, legal_list
                    )

            # === Phase 3: Step + observe for all envs ===
            for env_id in range(self.num_envs):
                env = self.envs[env_id]
                board, features = self.obs[env_id][0], self.obs[env_id][1]

                if self.cfg.algorithm == "dqn":
                    rot, col, hold, action_idx = batched_actions[env_id]
                    action = Action(rot, col, hold)
                else:
                    action_idx, log_prob, value = batched_actions[env_id]
                    rot, col, hold = decode_action(action_idx)
                    action = Action(rot, col, hold)

                with self.profiler.phase("env_step"):
                    obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_rewards[env_id] += reward
                episode_steps[env_id] += 1

                with self.profiler.phase("agent_observe"):
                    if self.cfg.algorithm == "dqn":
                        self.agent.observe(
                            env_id, (board, features), action_idx,
                            reward, (obs[0], obs[1]), done
                        )
                    else:
                        mask = env.get_legal_actions_mask(self.cfg.network.num_actions)
                        self.agent.buffer.add(
                            board, features, action_idx, log_prob, value,
                            reward, done, mask
                        )

                self.obs[env_id] = obs

                if done:
                    self.episode_rewards.append(episode_rewards[env_id])
                    self.episode_lines.append(info.get("lines", 0))
                    self.episode_pieces.append(info.get("pieces", 0))
                    self.episode_scores.append(info.get("score", 0))
                    self.episode_steps.append(info.get("steps", episode_steps[env_id]))
                    episode_rewards[env_id] = 0.0
                    episode_steps[env_id] = 0
                    with self.profiler.phase("env_reset"):
                        self.obs[env_id] = env.reset()

            # Training update.
            with self.profiler.phase("model_update"):
                if self.cfg.algorithm == "dqn":
                    if self.agent.env_step % self.agent.train_every == 0:
                        metrics = self.agent.update()
                        if metrics:
                            self._last_train_metrics = metrics
                            if self.agent.train_step % self.cfg.log_every == 0:
                                self.logger.log_train(self.agent.env_step, **metrics)
                else:
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
                avg_lines = np.mean(self.episode_lines) if self.episode_lines else 0
                avg_pieces = np.mean(self.episode_pieces) if self.episode_pieces else 0
                avg_score = np.mean(self.episode_scores) if self.episode_scores else 0
                avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0
                buf_size = len(self.agent.memory) if hasattr(self.agent, 'memory') else 0
                dead_rate = (dead_count / max(step - start_step, 1)) * 100.0

                # Current NoisyLinear sigma mean (for monitoring decay).
                sigma_mean = 0.0
                if hasattr(self.agent, 'online_net'):
                    sigma_mean = sum(
                        m.get_sigma_mean() for m in self.agent.online_net.modules()
                        if hasattr(m, 'get_sigma_mean')
                    )

                # Collect latest training metrics for progress line + TensorBoard.
                last_w_mean = last_prio_mean = last_rw_prio = last_init_prio = 0.0
                last_newcomers = 0
                if hasattr(self, '_last_train_metrics') and self._last_train_metrics:
                    last_w_mean = self._last_train_metrics.get("w_mean", 0)
                    last_prio_mean = self._last_train_metrics.get("prio_mean", 0)
                    last_rw_prio = self._last_train_metrics.get("rw_prio_mean", 0)
                    last_init_prio = self._last_train_metrics.get("init_prio", 0)
                    last_newcomers = self._last_train_metrics.get("newcomers", 0)

                self.logger.log_train_step(
                    step, avg_reward=avg_reward, avg_lines=avg_lines,
                    fps=fps, buffer_size=buf_size, elapsed=elapsed,
                    dead_count=dead_count, dead_rate=dead_rate,
                    avg_pieces=avg_pieces, avg_score=avg_score,
                    avg_steps=avg_steps, sigma_mean=sigma_mean,
                    w_mean=last_w_mean, td_prio_mean=last_prio_mean,
                    rw_prio_mean=last_rw_prio,
                )

                print(f"\r[{_timestamp()}]  "
                      f"Step {step:>9,}/{total_steps:,} ({progress:.1%})  "
                      f"|  Avg100R: {avg_reward:>10,.1f}  "
                      f"|  Steps: {avg_steps:>5.0f}/ep  "
                      f"|  Pieces: {avg_pieces:>5.0f}/ep  "
                      f"|  Dead: {dead_count:>5} ({dead_rate:.0f}%)  "
                      f"|  Sigma: {sigma_mean:.5f}  "
                      f"|  FPS: {fps:>7,.0f}", end="")
                # PER diagnostics line.
                if last_prio_mean > 0:
                    ratio = last_prio_mean / max(last_rw_prio, 0.001)
                    if ratio > 50:
                        health = f"{_RED}TD>>RW ({ratio:.0f}x) - death-dominated!{_RESET}"
                    elif ratio > 10:
                        health = f"{_YELLOW}TD>RW ({ratio:.0f}x){_RESET}"
                    else:
                        health = f"{_GREEN}OK ({ratio:.1f}x){_RESET}"
                    print(f"\n  PER:  W={last_w_mean:.3f}"
                          f"  TD-prio={last_prio_mean:.1f}"
                          f"  R-prio={last_rw_prio:.1f}"
                          f"  init={last_init_prio:.1f}"
                          f"  new={last_newcomers}"
                          f"  |  {health}", end="")
                # Profiler breakdown line.
                report = self.profiler.report(step, elapsed, self.num_envs)
                if report:
                    print(f"\n{report}")

            # Evaluation.
            if step % self.cfg.eval_every == 0 and step > 0:
                print()  # newline
                eval_metrics = self.evaluator.evaluate()
                self.logger.log_eval(step, **eval_metrics)

                # Dellacherie comparison (lightweight: 10 episodes).
                if step % (self.cfg.eval_every * 5) == 0 or step == self.cfg.eval_every:
                    try:
                        dl_metrics = self._eval_dellacherie(num_episodes=10)
                        self.logger.log({f"dellacherie/{k}": v for k, v in dl_metrics.items()}, step)
                    except Exception as e:
                        print(f"  [Dellacherie comparison skipped: {e}]")

                elapsed = time.time() - t_start
                print(f"[{_timestamp()}]  "
                      f"[Eval  @ step {step:>9,}]  "
                      f"Avg: {eval_metrics['avg_score']:>12,.1f}  "
                      f"Max: {eval_metrics['max_score']:>12,.1f}  "
                      f"Lines: {eval_metrics['avg_lines']:>6.1f}  "
                      f"Steps: {eval_metrics['avg_steps']:>6.0f}/ep  "
                      f"|  Elapsed: {_fmt_duration(elapsed)}")

                # Save best model.
                if eval_metrics["avg_score"] > self.best_avg_score:
                    self.best_avg_score = eval_metrics["avg_score"]
                    path = self.checkpoint.save_full(
                        step, self.agent,
                        replay_buffer=self.agent.memory if hasattr(self.agent, 'memory') else None
                    )
                    self.checkpoint.record_score(path, eval_metrics["avg_score"])

            # Periodic checkpoint.
            if step % self.cfg.save_every == 0 and step > 0:
                path = self.checkpoint.save_full(
                    step, self.agent,
                    replay_buffer=self.agent.memory if hasattr(self.agent, 'memory') else None
                )
                # Log delta vs previous checkpoint.
                prev = self._last_ckpt_metrics
                cur_eval = self.evaluator.evaluate()
                self.checkpoint.record_score(path, cur_eval["avg_score"])
                self._last_ckpt_metrics = cur_eval
                if prev:
                    delta_score = cur_eval["avg_score"] - prev["avg_score"]
                    delta_lines = cur_eval["avg_lines"] - prev["avg_lines"]
                    print(f"[{_timestamp()}]  "
                          f"[Ckpt @ step {step:>9,}]  "
                          f"Δ score: {delta_score:+,.1f}  "
                          f"Δ lines: {delta_lines:+,.1f}  "
                          f"cur: {cur_eval['avg_score']:,.1f}  "
                          f"prev: {prev['avg_score']:,.1f}")
                    self.logger.log({"ckpt/delta_score": delta_score,
                                     "ckpt/delta_lines": delta_lines,
                                     "ckpt/score": cur_eval["avg_score"],
                                     "ckpt/prev_score": prev["avg_score"]}, step)
                else:
                    self._last_ckpt_metrics = cur_eval
                    print(f"[{_timestamp()}]  "
                          f"[Ckpt @ step {step:>9,}]  "
                          f"score: {cur_eval['avg_score']:,.1f}  "
                          f"lines: {cur_eval['avg_lines']:.1f}  "
                          f"max: {cur_eval['max_score']:,.0f}")
                    self.logger.log({"ckpt/score": cur_eval["avg_score"],
                                     "ckpt/max_score": cur_eval["max_score"],
                                     "ckpt/lines": cur_eval["avg_lines"]}, step)

        # ================================================================ #
        #  Final Evaluation — Agent vs Dellacherie (same seeds, no noise)
        # ================================================================ #
        print("\n" + "=" * 60)
        print(f"[{_timestamp()}]  Final Evaluation — Agent vs Dellacherie")
        print("  (200 episodes, same seeds, deterministic, no exploration noise)")
        print("=" * 60)

        try:
            h2h = self.evaluator.head_to_head(num_episodes=200)
        except Exception as e:
            print(f"  Head-to-head evaluation failed: {e}")
            # Fallback: standard evaluation only.
            final_eval = self.evaluator.evaluate()
            h2h = None

        if h2h is not None:
            print(f"  Episodes:       {h2h['num_episodes']}")
            print(f"  ───────────  Agent  ───────────")
            print(f"  Avg Score:      {h2h['agent_avg']:>15,.1f}")
            print(f"  Max Score:      {h2h['agent_max']:>15,}")
            print(f"  Min Score:      {h2h['agent_min']:>15,}")
            print(f"  Std Score:      {h2h['agent_std']:>15,.1f}")
            print(f"  Avg Lines:      {h2h['agent_avg_lines']:>15,.1f}")
            print(f"  ───────────  Dellacherie  ──────")
            print(f"  Avg Score:      {h2h['dl_avg']:>15,.1f}")
            print(f"  Max Score:      {h2h['dl_max']:>15,}")
            print(f"  Min Score:      {h2h['dl_min']:>15,}")
            print(f"  Std Score:      {h2h['dl_std']:>15,.1f}")
            print(f"  Avg Lines:      {h2h['dl_avg_lines']:>15,.1f}")
            print(f"  ───────────  Comparison  ────────")
            print(f"  Mean Gap:       {h2h['mean_gap']:>+15,.1f}")
            print(f"  Median Gap:     {h2h['median_gap']:>+15,.1f}")
            print(f"  Win  Rate:      {h2h['win_rate']:>14.1%}  ({h2h['wins']}W / {h2h['losses']}L / {h2h['ties']}T)")
            print(f"  t-statistic:    {h2h['t_statistic']:>15.2f}")
            verdict_icon = "✓ Agent beats Dellacherie" if h2h['verdict'] == 'agent' else (
                "✗ Dellacherie still ahead" if h2h['verdict'] == 'dellacherie' else "≈ Tie"
            )
            print(f"  Verdict:        {verdict_icon}")

            # Log to TensorBoard / metrics.
            for k in ["agent_avg", "agent_max", "agent_avg_lines",
                      "dl_avg", "dl_max", "dl_avg_lines",
                      "mean_gap", "median_gap", "win_rate", "t_statistic",
                      "wins", "losses", "ties"]:
                self.logger.log({f"final/{k}": h2h[k]}, total_steps)

            final_eval = {"avg_score": h2h["agent_avg"],
                          "max_score": h2h["agent_max"],
                          "min_score": h2h["agent_min"],
                          "std_score": h2h["agent_std"],
                          "avg_lines": h2h["agent_avg_lines"]}
        else:
            print(f"  (Falling back to agent-only evaluation)")
            print(f"  Avg Score:      {final_eval['avg_score']:>15,.1f}")
            print(f"  Max Score:      {final_eval['max_score']:>15,.1f}")

        print("=" * 60)

        # Final save.
        print("\n[Trainer] Saving final model...")
        final_path = self.checkpoint.save_full(
            total_steps, self.agent,
            replay_buffer=self.agent.memory if hasattr(self.agent, 'memory') else None
        )
        self.checkpoint.record_score(final_path, final_eval["avg_score"])
        self.logger.save_metrics()
        self.logger.close()

        total_elapsed = time.time() - t_start
        print(f"[{_timestamp()}]  "
              f"[Trainer] Done.  "
              f"Total steps: {total_steps:,}  "
              f"Best avg score: {self.best_avg_score:,.1f}  "
              f"Final avg score: {final_eval['avg_score']:,.1f}  "
              f"Duration: {_fmt_duration(total_elapsed)}")
        tee.close()

    # ------------------------------------------------------------------ #
    #  Pretraining
    # ------------------------------------------------------------------ #
    def _pretrain(self):
        """Pretrain using Dellacherie behavior cloning.

        Fast path: if a previously saved pretrained weights file exists
        AND its model_size matches the current config, load it directly.
        Otherwise, delete stale cache and retrain from scratch.
        """
        from agent.pretrain import Pretrainer
        import os as _os
        import json as _json

        weights_path = _os.path.join("pretrain_samples", "pretrained_weights.pt")
        meta_path = _os.path.join("pretrain_samples", "pretrained_meta.json")

        # --- Fast path: load cached pretrained weights if compatible ---
        if _os.path.exists(weights_path):
            current_size = self.cfg.network.model_size
            cached_size = None
            if _os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        cached_size = _json.load(f).get("model_size")
                except Exception:
                    pass

            if cached_size is not None:
                if cached_size == current_size:
                    print(f"[Pretrain] Loading cached pretrained weights from {weights_path} ...")
                    converted = torch.load(weights_path, map_location="cpu")
                    try:
                        if self.cfg.algorithm == "dqn" and hasattr(self.agent, 'online_net'):
                            self.agent.online_net.load_state_dict(converted)
                            self.agent.target_net.load_state_dict(converted)
                        print("[Pretrain] Skipped data collection + BC training "
                              f"(cached weights loaded, model_size={current_size}).")
                        return
                    except RuntimeError as e:
                        print(f"[Pretrain] Cached weights incompatible: {e}")
                        print("[Pretrain] Deleting stale cache and retraining...")
                else:
                    print(f"[Pretrain] model_size changed ({cached_size} -> {current_size})"
                          f" -- cache invalid, retraining...")
                # Delete stale cache.
                for p in [weights_path, meta_path]:
                    if _os.path.exists(p):
                        _os.remove(p)
            else:
                # Weights file exists but no metadata -- old-format cache, delete it.
                print("[Pretrain] Cached weights found but no metadata "
                      "(old format) -- retraining...")
                _os.remove(weights_path)

        # --- Full path: collect + train + cache ---
        pretrainer = Pretrainer(model_type="dqn", num_actions=self.cfg.network.num_actions,
                                feature_dim=self.cfg.network.feature_dim,
                                model_size=self.cfg.network.model_size,
                                device=str(self.device))

        tag = self.cfg.pretrain_sample_tag
        sample_path = _os.path.join(pretrainer.sample_dir, f"samples_{tag}.npz")

        if _os.path.exists(sample_path):
            print(f"[Pretrain] Loading existing samples from {sample_path} ...")
            boards, features, actions, scores, meta = pretrainer.load_samples(tag)
            print(f"[Pretrain] Loaded {meta['num_transitions']:,} transitions "
                  f"(ep_score_mean={meta.get('episode_score_mean', 0):.0f}).")
        else:
            print(f"[Pretrain] No cached samples found ({sample_path}). "
                  f"Collecting {self.cfg.num_pretrain_episodes} episodes from Dellacherie expert...")
            pretrain_envs = min(self.cfg.num_pretrain_envs, self.cfg.num_pretrain_episodes)
            boards, features, actions, scores, meta = pretrainer.collect_dataset(
                self._make_env, self.cfg.num_pretrain_episodes, num_envs=pretrain_envs
            )
            pretrainer.save_samples(boards, features, actions, scores, meta)
            print(f"[Pretrain] Dellacherie score stats — "
                  f"mean={meta['dellacherie_score_mean']:.2f} "
                  f"std={meta['dellacherie_score_std']:.2f}  "
                  f"ep_score_mean={meta['episode_score_mean']:.0f}  "
                  f"ep_score_max={meta['episode_score_max']}")

        print(f"[Pretrain] Training BC on {len(actions):,} transitions...")
        state_dict = pretrainer.train(boards, features, actions,
                                       epochs=self.cfg.pretrain_epochs, scores=scores)

        # Load pretrained weights into agent.
        if self.cfg.algorithm == "dqn" and hasattr(self.agent, 'online_net'):
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

            # Cache the converted weights + metadata so next run can skip.
            _os.makedirs("pretrain_samples", exist_ok=True)
            torch.save(converted, weights_path)
            with open(meta_path, "w") as f:
                _json.dump({"model_size": self.cfg.network.model_size}, f)
            print(f"[Pretrain] Cached pretrained weights to {weights_path} "
                  f"(model_size={self.cfg.network.model_size}).")
            print("[Pretrain] Loaded pretrained weights into online + target networks.")

    # ------------------------------------------------------------------ #
    #  Dellacherie Comparison
    # ------------------------------------------------------------------ #
    def _eval_dellacherie(self, num_episodes: int = 10) -> Dict:
        """Run Dellacherie heuristic as a baseline for comparison.

        Returns metrics that can be compared against the learned policy.
        """
        dl_agent = DellacherieAgent()
        scores, lines_list, steps_list = [], [], []

        for _ in range(num_episodes):
            env = self._make_env()
            obs = env.reset()
            board_np = obs[0]
            done = False

            while not done:
                legal = env.get_legal_actions()
                if not legal:
                    break
                piece_idx = env._current_piece_name_idx
                rot, col, hold, score, feat = dl_agent.select_action(
                    board_np[0].astype(bool), legal, piece_idx
                )
                from env.tetris_env import Action
                obs, _, terminated, truncated, info = env.step(Action(rot, col, hold))
                board_np = obs[0]
                done = terminated or truncated

            scores.append(info.get("score", 0))
            lines_list.append(info.get("lines", 0))
            steps_list.append(info.get("steps", 0))

        return {
            "avg_score": float(np.mean(scores)),
            "max_score": int(np.max(scores)),
            "avg_lines": float(np.mean(lines_list)),
            "avg_steps": float(np.mean(steps_list)),
            "episodes": num_episodes,
        }


# ------------------------------------------------------------------ #
#  Config printing
# ------------------------------------------------------------------ #
def _print_config(config: TrainingConfig, args):
    """Print all configuration parameters before training starts.

    All values originate from either the TrainingConfig dataclass defaults
    or CLI args — no other source.
    """
    derived = config.total_steps

    def _kv(k, v, indent=2):
        return f"{' ' * indent}{k:24s} = {v}"

    print("=" * 60)
    print("  Training Configuration")
    print("=" * 60)

    print("  ── Training ──")
    print(_kv("algorithm", config.algorithm))
    print(_kv("total_samples", f"{config.total_samples:,}"))
    print(_kv("total_steps", f"{derived:,}  (derived)"))
    print(_kv("num_envs", config.num_envs))
    print(_kv("eval_every", f"{config.eval_every:,}"))
    print(_kv("eval_episodes", config.eval_episodes))
    print(_kv("save_every", f"{config.save_every:,}"))
    print(_kv("log_every", config.log_every))
    print(_kv("device", config.device))
    print(_kv("seed", config.seed))
    print(_kv("use_wandb", config.use_wandb))
    print(_kv("use_pretrain", config.use_pretrain))
    print(_kv("checkpoint_dir", config.checkpoint_dir))
    print(_kv("checkpoint_keep_best", config.checkpoint_keep_best))
    print(_kv("checkpoint_keep_latest", config.checkpoint_keep_latest))
    print(_kv("num_pretrain_episodes", config.num_pretrain_episodes))
    print(_kv("num_pretrain_envs", config.num_pretrain_envs))
    print(_kv("pretrain_epochs", config.pretrain_epochs))

    print("  ── Environment ──")
    env = config.env
    print(_kv("cols", env.cols))
    print(_kv("rows", env.rows))
    print(_kv("hidden_rows", env.hidden_rows))
    print(_kv("next_queue_size", env.next_queue_size))
    print(_kv("bag_type", env.bag_type))
    print(_kv("max_steps", f"{env.max_steps:,}"))
    print(_kv("use_cpp_env", env.use_cpp_env))
    print("  reward_weights:")
    for k, v in env.reward_weights.items():
        print(f"    {k:22s} = {v}")

    print("  ── Network ──")
    net = config.network
    print(_kv("model_size", net.model_size))
    print(_kv("cnn_channels", net.cnn_channels))
    print(_kv("hidden_dim", net.hidden_dim))
    print(_kv("feature_dim", net.feature_dim))
    print(_kv("num_actions", net.num_actions))
    print(_kv("use_noisy", net.use_noisy))
    print(_kv("sigma_init", net.sigma_init))
    print(_kv("sigma_decay", net.sigma_decay))

    print(f"  ── {config.algorithm.upper()} ──")
    if config.algorithm == "dqn":
        dqn = config.dqn
        print(_kv("gamma", dqn.gamma))
        print(_kv("lr", dqn.lr))
        print(_kv("batch_size", dqn.batch_size))
        print(_kv("train_every", dqn.train_every))
        print(_kv("n_step", dqn.n_step))
        print(_kv("target_update_freq", f"{dqn.target_update_freq:,}"))
        print(_kv("target_update_tau", dqn.target_update_tau))
        print(_kv("use_hard_update", dqn.use_hard_update))
        print(_kv("replay_capacity", f"{dqn.replay_capacity:,}"))
        print(_kv("per_alpha", dqn.per_alpha))
        print(_kv("per_reward_weight", dqn.per_reward_weight))
        print(_kv("per_reward_blend", dqn.per_reward_blend))
        print(_kv("per_beta_start", dqn.per_beta_start))
        print(_kv("per_beta_end", dqn.per_beta_end))
        print(_kv("per_beta_frames", f"{dqn.per_beta_frames:,}"))
        print(_kv("loss_type", dqn.loss_type))
        if dqn.loss_type == "huber":
            print(_kv("huber_beta", dqn.huber_beta))
        print(_kv("grad_clip_norm", dqn.grad_clip_norm))
    else:
        ppo = config.ppo
        print(_kv("gamma", ppo.gamma))
        print(_kv("gae_lambda", ppo.gae_lambda))
        print(_kv("clip_epsilon", ppo.clip_epsilon))
        print(_kv("value_coef", ppo.value_coef))
        print(_kv("entropy_coef", ppo.entropy_coef))
        print(_kv("lr", ppo.lr))
        print(_kv("batch_size", ppo.batch_size))
        print(_kv("mini_batch_size", ppo.mini_batch_size))
        print(_kv("n_epochs", ppo.n_epochs))
        print(_kv("max_grad_norm", ppo.max_grad_norm))
        print(_kv("rollout_steps", f"{ppo.rollout_steps:,}"))
        print(_kv("num_envs", ppo.num_envs))

    print("  ── CLI args ──")
    for k, v in sorted(vars(args).items()):
        print(_kv(k, v))

    print("=" * 60)


# ------------------------------------------------------------------ #
#  YAML config loading
# ------------------------------------------------------------------ #
def _load_yaml_config(path: str) -> dict:
    """Load a YAML config file.  Requires ``pyyaml``.

    Returns an empty dict if yaml is unavailable or the file is missing.
    """
    try:
        import yaml
    except ImportError:
        print("[Config] pyyaml not installed — using dataclass defaults only.")
        print("[Config]   pip install pyyaml   to enable YAML config files.")
        return {}

    if not os.path.exists(path):
        print(f"[Config] Config file not found: {path}  — using defaults.")
        return {}

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _merge_into_dataclass(cfg: TrainingConfig, yaml_dict: dict):
    """Merge top-level YAML keys into a TrainingConfig instance in-place.

    Top-level keys map to TrainingConfig fields directly.
    Nested keys (dqn, network, env, ppo) are merged into sub-dataclasses.
    """
    for key, value in yaml_dict.items():
        if key == "training" and isinstance(value, dict):
            for k, v in value.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        elif key == "dqn" and isinstance(value, dict):
            for k, v in value.items():
                if hasattr(cfg.dqn, k):
                    setattr(cfg.dqn, k, v)
        elif key == "network" and isinstance(value, dict):
            for k, v in value.items():
                if hasattr(cfg.network, k):
                    setattr(cfg.network, k, v)
        elif key == "env" and isinstance(value, dict):
            for k, v in value.items():
                if k == "reward_weights" and isinstance(v, dict):
                    cfg.env.reward_weights.update(v)
                elif hasattr(cfg.env, k):
                    setattr(cfg.env, k, v)
        elif key == "ppo" and isinstance(value, dict):
            for k, v in value.items():
                if hasattr(cfg.ppo, k):
                    setattr(cfg.ppo, k, v)


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
    parser = argparse.ArgumentParser(
        description="Train Tetris RL Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/train.py                                              # DQN, 1B samples
  python scripts/train.py --config configs/training/dqn_rainbow.yaml   # explicit config
  python scripts/train.py --samples 500000000                          # 500M samples
  python scripts/train.py --steps 10000000                             # Override: 10M env steps
  python scripts/train.py --algo ppo --samples 2000000000              # PPO, 2B samples""")
    parser.add_argument("--config", type=str, default="configs/training/dqn_rainbow.yaml",
                        help="Path to YAML config file (requires pyyaml).")
    parser.add_argument("--algo", type=str, default=None, choices=["dqn", "ppo"])
    parser.add_argument("--samples", type=int, default=None,
                        help="Total training samples consumed (DQN: transitions fed to optimizer. "
                             "Derived total_steps = samples × train_every / batch_size)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override: set env steps directly (bypasses derived computation)")
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from latest checkpoint in --checkpoint-dir")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a specific checkpoint file")
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-phase timing breakdown")
    args = parser.parse_args()

    # 1. Start from dataclass defaults.
    config = TrainingConfig()

    # 2. Overlay YAML config file (if pyyaml is installed).
    yaml_dict = _load_yaml_config(args.config)
    if yaml_dict:
        _merge_into_dataclass(config, yaml_dict)
        print(f"[Config] Loaded YAML config: {args.config}")

    # 3. Overlay CLI args (highest priority).
    if args.algo is not None:
        config.algorithm = args.algo
    if args.envs is not None:
        config.num_envs = args.envs
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.wandb is not None:
        config.use_wandb = args.wandb
    if args.no_pretrain:
        config.use_pretrain = False
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.samples is not None:
        config.total_samples = args.samples
    if args.steps is not None:
        config.total_steps = args.steps

    _print_config(config, args)

    trainer = Trainer(config, resume=args.resume, resume_from=args.resume_from,
                     profile=args.profile)
    trainer.train()


if __name__ == "__main__":
    main()
