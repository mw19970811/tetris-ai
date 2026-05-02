"""Evaluation loop for measuring agent performance.

Includes:
  - ``evaluate()``: standard deterministic evaluation of the learned policy.
  - ``head_to_head()``: same-seed 200-episode match against Dellacherie baseline.
"""

import numpy as np
import torch
from typing import Dict, List
from agent.action_mask import decode_action, encode_action
from agent.dellacherie import DellacherieAgent
from env.tetris_env import Action


def _run_episode(env, agent, is_dqn: bool) -> Dict:
    """Run one episode of *agent* on *env*, returning {score, lines, level, steps}."""
    obs = env.reset()
    board, features = obs[0], obs[1]
    done = False

    while not done:
        legal = env.get_legal_actions()
        if not legal:
            break

        if is_dqn:
            rot, col, hold, _ = agent.select_action(
                board, features, legal, deterministic=True
            )
        else:
            action_idx, _, _ = agent.select_action(
                board, features, legal, deterministic=True
            )
            rot, col, hold = decode_action(action_idx)

        obs, _, terminated, truncated, info = env.step(Action(rot, col, hold))
        board, features = obs[0], obs[1]
        done = terminated or truncated

    return {
        "score": info.get("score", 0),
        "lines": info.get("lines", 0),
        "level": info.get("level", 1),
        "steps": info.get("steps", 0),
    }


def _run_episode_dellacherie(env) -> Dict:
    """Run one episode of Dellacherie heuristic on *env*."""
    dl = DellacherieAgent()
    obs = env.reset()
    board_np = obs[0]
    done = False

    while not done:
        legal = env.get_legal_actions()
        if not legal:
            break
        rot, col, hold, _, _ = dl.select_action(
            board_np[0].astype(bool), legal, env._current_piece_name_idx
        )
        obs, _, terminated, truncated, info = env.step(Action(rot, col, hold))
        board_np = obs[0]
        done = terminated or truncated

    return {
        "score": info.get("score", 0),
        "lines": info.get("lines", 0),
        "level": info.get("level", 1),
        "steps": info.get("steps", 0),
    }


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
        """Run standard deterministic evaluation. Noise disabled via eval_mode."""
        self.agent.eval_mode()
        is_dqn = hasattr(self.agent, 'online_net')

        scores, lines_list, levels, steps_list = [], [], [], []
        for _ in range(self.num_episodes):
            env = self.env_creator()
            res = _run_episode(env, self.agent, is_dqn)
            scores.append(res["score"])
            lines_list.append(res["lines"])
            levels.append(res["level"])
            steps_list.append(res["steps"])

        self.agent.train_mode()
        return self._summarise(scores, lines_list, levels, steps_list)

    @torch.no_grad()
    def head_to_head(self, num_episodes: int = 200) -> Dict:
        """Agent vs Dellacherie on the **same** 200 seeds.

        Both sides play identical piece sequences (seed = episode index).
        Noise is disabled (eval_mode) — agent uses pure argmax over Q-values.

        Returns a dict with per-side aggregate metrics + comparison stats.
        """
        self.agent.eval_mode()
        is_dqn = hasattr(self.agent, 'online_net')

        agent_scores, dl_scores = [], []
        wins, losses, ties = 0, 0, 0
        gaps = []
        details: List[Dict] = []

        for ep in range(num_episodes):
            env_a = self.env_creator()
            env_d = self.env_creator()
            env_a.reset(seed=ep)
            env_d.reset(seed=ep)

            res_a = _run_episode(env_a, self.agent, is_dqn)
            res_d = _run_episode_dellacherie(env_d)

            agent_scores.append(res_a["score"])
            dl_scores.append(res_d["score"])

            gap = res_a["score"] - res_d["score"]
            gaps.append(gap)
            if gap > 0:
                wins += 1
            elif gap < 0:
                losses += 1
            else:
                ties += 1

            details.append({
                "episode": ep,
                "agent_score": res_a["score"],
                "dl_score": res_d["score"],
                "gap": gap,
                "agent_lines": res_a["lines"],
                "dl_lines": res_d["lines"],
            })

        self.agent.train_mode()

        # Statistical tests (lightweight).
        import math
        gaps_arr = np.array(gaps, dtype=float)
        mean_gap = float(np.mean(gaps_arr))
        std_gap = float(np.std(gaps_arr))
        se_gap = std_gap / math.sqrt(num_episodes) if num_episodes > 0 else 0
        t_stat = mean_gap / se_gap if se_gap > 0 else 0.0

        return {
            "num_episodes": num_episodes,
            # Agent.
            "agent_avg": float(np.mean(agent_scores)),
            "agent_max": int(np.max(agent_scores)),
            "agent_min": int(np.min(agent_scores)),
            "agent_std": float(np.std(agent_scores)),
            "agent_avg_lines": float(np.mean([d["agent_lines"] for d in details])),
            # Dellacherie.
            "dl_avg": float(np.mean(dl_scores)),
            "dl_max": int(np.max(dl_scores)),
            "dl_min": int(np.min(dl_scores)),
            "dl_std": float(np.std(dl_scores)),
            "dl_avg_lines": float(np.mean([d["dl_lines"] for d in details])),
            # Comparison.
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / num_episodes,
            "mean_gap": mean_gap,
            "median_gap": float(np.median(gaps_arr)),
            "std_gap": std_gap,
            "t_statistic": t_stat,
            "verdict": "agent" if mean_gap > 0 and wins > losses else (
                "dellacherie" if mean_gap < 0 and losses > wins else "tie"
            ),
            # Raw data.
            "details": details,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _summarise(scores, lines_list, levels, steps_list) -> Dict:
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
