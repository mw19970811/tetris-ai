"""Imitation learning pre-training using Dellacherie heuristic.

Collects (state, action) pairs from Dellacherie expert, then trains
the policy network via behavior cloning to provide a warm start.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional

import os
import json
import hashlib

from .model import DuelingDQN, ActorCritic
from .action_mask import create_action_mask, encode_action
from .dellacherie import DellacherieAgent, DellacherieConfig, DEFAULT_WEIGHTS
from env.tetris_env import Action

# Dellacherie weights (from particle swarm optimisation).
DELLACHERIE_WEIGHTS = {
    "landing_height": -4.500,
    "cleared_lines": 3.418,
    "holes": -7.899,
    "bumpiness": -3.386,
    "max_well": -3.129,
    "row_transitions": -2.000,
}


class DellacherieExpert:
    """Dellacherie heuristic Tetris AI — provides expert actions.

    Evaluates each legal placement using a weighted sum of six features
    and selects the one with the highest score.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or DELLACHERIE_WEIGHTS

    def select_action(self, board: np.ndarray, legal_actions: List,
                      current_piece: int, piece_shapes: dict) -> Tuple[int, int, int]:
        """Select best action using Dellacherie heuristic.

        Args:
            board: (22, 10) boolean array.
            legal_actions: List of (rotation, column, hold) tuples.
            current_piece: int 0-6.

        Returns:
            (rotation, column, hold) of the best action.
        """
        best_score = float('-inf')
        best_action = (0, 0, False)

        for action in legal_actions:
            rotation, column, hold = action.rotation, action.column, action.hold
            # Simulate placement.
            sim_board = board.copy()
            piece_name = ["I","O","T","S","Z","J","L"][current_piece]
            cells = self._get_cells(piece_name, rotation)

            # Ghost drop.
            gy = self._ghost_y(sim_board, cells, column, 2)
            if gy < 0:
                continue

            # Place piece.
            for cx, cy in cells:
                r, c = gy + cy, column + cx
                if 0 <= r < 22 and 0 <= c < 10:
                    sim_board[r, c] = True

            # Clear lines.
            cleared = self._clear_lines(sim_board)

            # Compute features.
            heights = self._column_heights(sim_board)
            holes = self._count_holes(sim_board)
            bump = self._bumpiness(heights)
            well = self._max_well(heights)
            row_trans = self._row_transitions(sim_board)

            score = (self.weights["landing_height"] * float(np.sum(heights))
                     + self.weights["cleared_lines"] * float(cleared)
                     + self.weights["holes"] * float(holes)
                     + self.weights["bumpiness"] * float(bump)
                     + self.weights["max_well"] * float(well)
                     + self.weights["row_transitions"] * float(row_trans))

            if score > best_score:
                best_score = score
                best_action = (rotation, column, hold)

        return best_action

    @staticmethod
    def _get_cells(piece_name, rotation):
        shapes = {
            "I": [[(0,0),(1,0),(2,0),(3,0)],[(0,0),(0,1),(0,2),(0,3)],
                  [(0,0),(1,0),(2,0),(3,0)],[(0,0),(0,1),(0,2),(0,3)]],
            "O": [[(0,0),(1,0),(0,1),(1,1)]] * 4,
            "T": [[(0,0),(1,0),(2,0),(1,1)],[(0,0),(0,1),(0,2),(1,1)],
                  [(1,0),(0,1),(1,1),(2,1)],[(1,0),(1,1),(1,2),(0,1)]],
            "S": [[(1,0),(2,0),(0,1),(1,1)],[(0,0),(0,1),(1,1),(1,2)],
                  [(1,0),(2,0),(0,1),(1,1)],[(0,0),(0,1),(1,1),(1,2)]],
            "Z": [[(0,0),(1,0),(1,1),(2,1)],[(1,0),(0,1),(1,1),(0,2)],
                  [(0,0),(1,0),(1,1),(2,1)],[(1,0),(0,1),(1,1),(0,2)]],
            "J": [[(0,0),(0,1),(1,1),(2,1)],[(0,0),(1,0),(0,1),(0,2)],
                  [(0,0),(1,0),(2,0),(2,1)],[(1,0),(1,1),(0,2),(1,2)]],
            "L": [[(2,0),(0,1),(1,1),(2,1)],[(0,0),(0,1),(0,2),(1,2)],
                  [(0,0),(1,0),(2,0),(0,1)],[(0,0),(1,0),(1,1),(1,2)]],
        }
        return shapes[piece_name][rotation]

    @staticmethod
    def _ghost_y(board, cells, col, start_y):
        y = start_y
        while True:
            collision = False
            for cx, cy in cells:
                r, c = y + 1 + cy, col + cx
                if c < 0 or c >= 10 or r >= 22:
                    collision = True; break
                if r >= 0 and board[r, c]:
                    collision = True; break
            if collision: break
            y += 1
        return y

    @staticmethod
    def _clear_lines(board):
        cleared = 0
        for r in range(21, -1, -1):
            if np.all(board[r]):
                board[1:r+1] = board[:r]
                board[0] = False
                cleared += 1
        return cleared

    @staticmethod
    def _column_heights(board):
        h = np.zeros(10, dtype=int)
        for c in range(10):
            filled = np.where(board[:, c])[0]
            h[c] = 22 - filled[0] if len(filled) > 0 else 0
        return h

    @staticmethod
    def _count_holes(board):
        holes = 0
        for c in range(10):
            found = False
            for r in range(22):
                if board[r, c]: found = True
                elif found: holes += 1
        return holes

    @staticmethod
    def _bumpiness(h):
        return int(np.sum(np.abs(np.diff(h))))

    @staticmethod
    def _max_well(h):
        w = 0
        for c in range(10):
            ld = h[c-1] - h[c] if c > 0 else 0
            rd = h[c+1] - h[c] if c < 9 else 0
            w = max(w, max(0, min(ld, rd)))
        return w

    @staticmethod
    def _row_transitions(board):
        t = 0
        for r in range(22):
            for c in range(11):
                left = board[r, c-1] if c > 0 else False
                right = board[r, c] if c < 10 else False
                if left != right: t += 1
        return t


class Pretrainer:
    """Trains a network to mimic Dellacherie via behavior cloning.

    Supports:
      - Ablation experiments via DellacherieConfig
      - Saving/loading collected samples to/from disk
      - Dellacherie score/value per transition for later comparison
    """

    def __init__(self, model_type: str = "dqn", num_actions: int = 112,
                 feature_dim: int = 53, lr: float = 1e-3, device: str = "cuda",
                 dellacherie_config: DellacherieConfig = None,
                 sample_dir: str = "pretrain_samples"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.model_type = model_type
        self.sample_dir = sample_dir

        if model_type == "dqn":
            self.model = DuelingDQN(num_actions=num_actions, feature_dim=feature_dim,
                                     use_noisy=False).to(self.device)
        else:
            self.model = ActorCritic(num_actions=num_actions, feature_dim=feature_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.expert = DellacherieAgent(dellacherie_config)

    def collect_dataset(self, env_creator, num_episodes: int = 1000,
                         num_envs: int = 16
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, dict]:
        """Run Dellacherie expert across N parallel envs to collect (state, action) pairs.

        Returns:
            boards, features, actions, dellacherie_scores, metadata
        """
        boards_list, features_list, actions_list = [], [], []
        scores_list = []
        episode_scores = []
        episode_lines = []

        envs = [env_creator() for _ in range(num_envs)]
        obs_list = [env.reset() for env in envs]
        total_completed = 0
        last_report = 0

        while total_completed < num_episodes:
            for i in range(num_envs):
                if total_completed >= num_episodes:
                    break

                obs_tuple = obs_list[i]
                board_np, feat_np = obs_tuple[0], obs_tuple[1]
                legal = envs[i].get_legal_actions()

                if not legal:
                    obs_list[i] = envs[i].reset()
                    total_completed += 1
                    continue

                # Expert action with Dellacherie score.
                rot, col, hold, score, features = self.expert.select_action(
                    board_np[0].astype(bool), legal, envs[i]._current_piece_name_idx
                )
                action_idx = encode_action(rot, col, hold)

                boards_list.append(board_np.copy())
                features_list.append(feat_np.copy())
                actions_list.append(action_idx)
                scores_list.append(score)

                next_obs, reward, terminated, truncated, info = envs[i].step(Action(rot, col, hold))
                obs_list[i] = next_obs
                if terminated or truncated:
                    episode_scores.append(info.get("score", 0))
                    episode_lines.append(info.get("lines", 0))
                    total_completed += 1
                    obs_list[i] = envs[i].reset()

            if total_completed - last_report >= 100:
                last_report = total_completed
                print(f"  Collected {total_completed}/{num_episodes} episodes, "
                      f"{len(actions_list):,} transitions")

        print(f"  Collected {total_completed}/{num_episodes} episodes, "
              f"{len(actions_list):,} transitions")

        # Stack tensors.
        boards = np.concatenate(boards_list) if boards_list else np.array([])
        if len(boards) > 0 and boards.ndim == 3:
            boards = boards[:, np.newaxis, :, :]
        features = np.stack(features_list) if features_list else np.array([])
        actions = np.array(actions_list, dtype=np.int64)
        scores = np.array(scores_list, dtype=np.float32)

        # Build metadata.
        metadata = {
            "num_episodes": total_completed,
            "num_transitions": len(actions_list),
            "dellacherie_score_mean": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "dellacherie_score_std": float(np.std(scores)) if len(scores) > 0 else 0.0,
            "episode_score_mean": float(np.mean(episode_scores)) if episode_scores else 0.0,
            "episode_score_max": int(np.max(episode_scores)) if episode_scores else 0,
            "episode_lines_mean": float(np.mean(episode_lines)) if episode_lines else 0.0,
            "active_features": self.expert.active_features,
            "weights": dict(self.expert.weights),
        }
        return boards, features, actions, scores, metadata

    # ------------------------------------------------------------------ #
    #  Save / Load samples
    # ------------------------------------------------------------------ #
    def save_samples(self, boards: np.ndarray, features: np.ndarray,
                     actions: np.ndarray, scores: np.ndarray,
                     metadata: dict, tag: str = "latest") -> str:
        """Persist collected samples to disk. Returns the save path."""
        os.makedirs(self.sample_dir, exist_ok=True)
        base = os.path.join(self.sample_dir, f"samples_{tag}")
        np.savez_compressed(
            base + ".npz",
            boards=boards, features=features, actions=actions, scores=scores,
        )
        with open(base + ".json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Pretrain] Saved {metadata['num_transitions']:,} transitions to {base}.npz")
        return base + ".npz"

    def load_samples(self, tag: str = "latest"
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, dict]:
        """Load previously saved pretrain samples. Returns (boards, features, actions, scores, metadata)."""
        base = os.path.join(self.sample_dir, f"samples_{tag}")
        data = np.load(base + ".npz")
        with open(base + ".json") as f:
            metadata = json.load(f)
        print(f"[Pretrain] Loaded {metadata['num_transitions']:,} transitions from {base}.npz")
        return (data["boards"], data["features"], data["actions"],
                data["scores"], metadata)

    def load_partial_samples(self, tag: str = "latest", max_transitions: int = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, dict]:
        """Load a subset of saved samples (useful for ablation with varied data sizes)."""
        boards, features, actions, scores, metadata = self.load_samples(tag)
        if max_transitions and max_transitions < len(actions):
            idx = np.random.choice(len(actions), max_transitions, replace=False)
            boards = boards
            features = features[idx]
            actions = actions[idx]
            scores = scores[idx]
            metadata["num_transitions"] = max_transitions
            metadata["partial"] = True
        return boards, features, actions, scores, metadata

    def _hash_config(self) -> str:
        """Short hash of the Dellacherie config for sample provenance."""
        raw = json.dumps({"weights": dict(self.expert.weights),
                          "active": self.expert.active_features}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    def train(self, boards: np.ndarray, features: np.ndarray, actions: np.ndarray,
              epochs: int = 50, batch_size: int = 256,
              scores: np.ndarray = None):
        """Behavior cloning: supervised learning on expert data.

        When *scores* is provided, transitions with higher Dellacherie
        scores receive higher sampling weight (optional).
        """
        if len(boards) == 0:
            return

        dataset = TensorDataset(
            torch.as_tensor(boards, dtype=torch.float32),
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
        )

        if scores is not None and len(scores) > 0:
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                sample_weights = 0.1 + 0.9 * (scores - s_min) / (s_max - s_min)
            else:
                sample_weights = np.ones_like(scores)
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.float32),
                num_samples=len(sample_weights), replacement=True,
            )
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_boards, batch_feats, batch_actions in loader:
                batch_boards = batch_boards.to(self.device)
                batch_feats = batch_feats.to(self.device)
                batch_actions = batch_actions.to(self.device)

                if self.model_type == "dqn":
                    q_values = self.model(batch_boards, batch_feats)
                    loss = nn.CrossEntropyLoss()(q_values, batch_actions)
                    pred = q_values.argmax(dim=-1)
                else:
                    logits, _ = self.model(batch_boards, batch_feats)
                    loss = nn.CrossEntropyLoss()(logits, batch_actions)
                    pred = logits.argmax(dim=-1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct += (pred == batch_actions).sum().item()
                total += len(batch_actions)

            acc = correct / max(total, 1)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(loader):.4f}  Acc: {acc:.3f}")

        self.model.eval()
        return self.model.state_dict()
