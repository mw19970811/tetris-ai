"""Actor worker process: independently samples the environment and sends transitions to the Learner.

Each Actor runs:
  1. A C++ (or Python) Tetris environment.
  2. A local copy of the latest model parameters.
  3. Selects actions using epsilon-greedy or noisy-net exploration.

Communication:
  - Pulls model params from shared memory when signalled.
  - Pushes (state, action, reward, next_state, done) to the sample queue.
"""

import torch
import numpy as np
from typing import Callable, Any, Dict
from multiprocessing import Queue, Event
from collections import deque


class ActorWorker:
    """Runs in an independent process. Collects experience by playing episodes."""

    @staticmethod
    def run(actor_id: int,
            env_creator: Callable[[], Any],
            sample_queue: Queue,
            param_event: Event,
            stop_event: Event,
            sync_interval: int = 100):
        """Entry point for actor process (called via multiprocessing.Process).

        Args:
            actor_id: Unique ID for this actor (0..N-1).
            env_creator: Callable that returns a fresh environment.
            sample_queue: Multiprocessing queue to push samples.
            param_event: Set by trainer when new params are available.
            stop_event: Set to signal graceful shutdown.
            sync_interval: How often (in steps) to check for new params.
        """
        import torch
        import numpy as np

        env = env_creator()

        # Local model copy (lazy init on first param sync).
        local_model = None
        model_version = -1
        device = torch.device("cpu")

        step = 0
        episode_reward = 0.0

        obs = env.reset()
        board_np, feat_np = obs[0], obs[1]

        while not stop_event.is_set():
            # Pull latest params if available.
            if param_event.is_set():
                model_version = ActorWorker._pull_params(local_model, model_version)

            # Select action (simple random for initialisation until model ready).
            legal = env.get_legal_actions()
            if not legal:
                obs = env.reset()
                board_np, feat_np = obs[0], obs[1]
                continue

            # Simple exploration: random action if no model yet.
            if local_model is None:
                action = legal[np.random.randint(len(legal))]
                action_idx = 0
            else:
                # Placeholder: would use model inference.
                action = legal[np.random.randint(len(legal))]
                action_idx = 0

            # Step environment.
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_board, next_feat = obs_next[0], obs_next[1]
            episode_reward += reward

            # Push sample.
            sample = {
                "state": {"board": board_np, "features": feat_np},
                "action": action_idx,
                "reward": reward,
                "next_state": {"board": next_board, "features": next_feat},
                "done": done,
            }
            try:
                sample_queue.put(sample, timeout=0.1)
            except:
                pass

            board_np, feat_np = next_board, next_feat
            step += 1

            if done:
                obs = env.reset()
                board_np, feat_np = obs[0], obs[1]
                episode_reward = 0.0

            # Periodic param check.
            if step % sync_interval == 0 and param_event.is_set():
                model_version = ActorWorker._pull_params(local_model, model_version)

    @staticmethod
    def _pull_params(local_model, current_version: int) -> int:
        """Pull latest parameters from shared memory (placeholder)."""
        # In full implementation: read from multiprocessing.shared_memory.
        return current_version + 1
