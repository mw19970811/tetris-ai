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
                        if metrics and self.agent.train_step % self.cfg.log_every == 0:
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

                # TensorBoard curves.
                self.logger.log_train_step(
                    step, avg_reward=avg_reward, avg_lines=avg_lines,
                    fps=fps, buffer_size=buf_size, elapsed=elapsed,
                    dead_count=dead_count, dead_rate=dead_rate,
                    avg_pieces=avg_pieces, avg_score=avg_score,
                    avg_steps=avg_steps, sigma_mean=sigma_mean,
                )

                print(f"\r[{_timestamp()}]  "
                      f"Step {step:>9,}/{total_steps:,} ({progress:.1%})  "
                      f"|  Avg100R: {avg_reward:>10,.1f}  "
                      f"|  Pieces: {avg_pieces:>5.0f}/ep  "
                      f"|  Steps: {avg_steps:>6.0f}/ep  "
                      f"|  Sigma: {sigma_mean:.5f}  "
                      f"|  Stale: {dead_count:>5}  "
                      f"|  FPS: {fps:>8,.0f}  "
                      f"|  Elapsed: {_fmt_duration(elapsed)}  "
                      f"|  ETA: {_fmt_duration(eta)}", end="")
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

        Fast path: if a previously saved pretrained weights file exists,
        load it directly — no data collection, no BC training.
        """
        from agent.pretrain import Pretrainer
        import os as _os

        weights_path = _os.path.join("pretrain_samples", "pretrained_weights.pt")

        # --- Fast path: load cached pretrained weights ---
        if _os.path.exists(weights_path):
            print(f"[Pretrain] Loading cached pretrained weights from {weights_path} ...")
            converted = torch.load(weights_path, map_location="cpu")
            if self.cfg.algorithm == "dqn" and hasattr(self.agent, 'online_net'):
                self.agent.online_net.load_state_dict(converted)
                self.agent.target_net.load_state_dict(converted)
            print("[Pretrain] Skipped data collection + BC training (cached weights loaded).")
            return

        # --- Full path: collect + train + cache ---
        pretrainer = Pretrainer(model_type="dqn", num_actions=self.cfg.network.num_actions,
                                feature_dim=self.cfg.network.feature_dim, device=str(self.device))

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

            # Cache the converted weights so next run can skip everything.
            _os.makedirs("pretrain_samples", exist_ok=True)
            torch.save(converted, weights_path)
            print(f"[Pretrain] Cached pretrained weights to {weights_path}.")
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
  python scripts/train.py                                  # DQN, 1B samples
  python scripts/train.py --samples 500000000              # 500M samples
  python scripts/train.py --steps 10000000                 # Override: 10M env steps
  python scripts/train.py --algo ppo --samples 2000000000  # PPO, 2B samples""")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--samples", type=int, default=None,
                        help="Total training samples consumed (DQN: transitions fed to optimizer. "
                             "Derived total_steps = samples × train_every / batch_size)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override: set env steps directly (bypasses derived computation)")
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
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-phase timing breakdown")
    args = parser.parse_args()

    config = TrainingConfig(
        algorithm=args.algo,
        num_envs=args.envs,
        device=args.device,
        seed=args.seed,
        use_wandb=args.wandb,
        use_pretrain=not args.no_pretrain,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.samples is not None:
        config.total_samples = args.samples
    if args.steps is not None:
        config.total_steps = args.steps

    derived = config.total_steps
    print(f"[Config] total_samples={config.total_samples:,}  "
          f"batch_size={config.dqn.batch_size}  train_every={config.dqn.train_every}  "
          f"→ total_steps={derived:,}")
    trainer = Trainer(config, resume=args.resume, resume_from=args.resume_from,
                     profile=args.profile)
    trainer.train()


if __name__ == "__main__":
    main()
