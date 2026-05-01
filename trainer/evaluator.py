"""Evaluation loop for measuring agent performance."""

import numpy as np
import torch
from typing import Dict
from collections import defaultdict
from agent.action_mask import decode_action


class Evaluator:
    """Evaluates a trained agent over multiple episodes with deterministic play."""

    def __init__(self, env_creator, agent, num_episodes: int = 100,
                 device: str = "cuda"):
        self.env_creator = env_creator
        self.agent = agent
        self.num_episodes = num_episodes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return aggregate metrics."""
        self.agent.eval_mode()

        scores = []
        lines_list = []
        levels = []
        steps_list = []

        # Detect agent type for action dispatch.
        is_dqn = hasattr(self.agent, 'online_net')

        for ep in range(self.num_episodes):
            env = self.env_creator()
            obs = env.reset()
            board, features = obs[0], obs[1]
            done = False

            while not done:
                legal_actions = env.get_legal_actions()
                if not legal_actions:
                    break

                if is_dqn:
                    # DQN: select_action returns (rotation, col, hold, action_idx).
                    rot, col, hold, _ = self.agent.select_action(
                        board, features, legal_actions, deterministic=True
                    )
                else:
                    # PPO: select_action returns (action_idx, log_prob, value).
                    action_idx, _, _ = self.agent.select_action(
                        board, features, legal_actions, deterministic=True
                    )
                    rot, col, hold = decode_action(action_idx)

                from env.tetris_env import Action
                action = Action(rot, col, hold)
                obs, reward, terminated, truncated, info = env.step(action)
                board, features = obs[0], obs[1]
                done = terminated or truncated

            scores.append(info.get("score", 0))
            lines_list.append(info.get("lines", 0))
            levels.append(info.get("level", 1))
            steps_list.append(info.get("steps", 0))

        self.agent.train_mode()

        return {
            "avg_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "std_score": float(np.std(scores)),
            "avg_lines": float(np.mean(lines_list)),
            "avg_level": float(np.mean(levels)),
            "avg_steps": float(np.mean(steps_list)),
            "tetris_rate": 0.0,
        }

    def evaluate_with_details(self) -> Dict:
        """Extended evaluation with per-episode breakdown."""
        metrics = self.evaluate()
        return metrics
