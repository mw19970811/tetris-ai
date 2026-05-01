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

from .model import DuelingDQN, ActorCritic
from .action_mask import create_action_mask, encode_action

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
    """Trains a network to mimic Dellacherie via behavior cloning."""

    def __init__(self, model_type: str = "dqn", num_actions: int = 112,
                 feature_dim: int = 53, lr: float = 1e-3, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.model_type = model_type

        if model_type == "dqn":
            self.model = DuelingDQN(num_actions=num_actions, feature_dim=feature_dim,
                                     use_noisy=False).to(self.device)
        else:
            self.model = ActorCritic(num_actions=num_actions, feature_dim=feature_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.expert = DellacherieExpert()

    def collect_dataset(self, env, num_episodes: int = 1000
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run Dellacherie expert to collect (state, action) pairs."""
        boards_list, features_list, actions_list = [], [], []

        for ep in range(num_episodes):
            obs = env.reset()
            board_np = obs[0]
            feat_np = obs[1]
            done = False

            while not done:
                legal = env.get_legal_actions()
                if not legal:
                    break

                # Expert action.
                rot, col, hold = self.expert.select_action(
                    board_np[0].astype(bool), legal, env._current_piece_name_idx, None
                )
                action_idx = encode_action(rot, col, hold)

                boards_list.append(board_np.copy())
                features_list.append(feat_np.copy())
                actions_list.append(action_idx)

                # Step environment.
                from env.tetris_env import Action
                obs, _, terminated, truncated, _ = env.step(Action(rot, col, hold))
                board_np, feat_np = obs[0], obs[1]
                done = terminated or truncated

            if (ep + 1) % 100 == 0:
                print(f"  Collected {ep + 1}/{num_episodes} episodes, {len(actions_list)} transitions")

        # Each board_np is (1, 22, 10) from state encoder — stack to (N, 22, 10).
        boards = np.concatenate([b for b in boards_list]) if len(boards_list) > 0 else np.array([])
        # Add channel dimension: (N, 22, 10) → (N, 1, 22, 10) for CNN NCHW.
        if len(boards) > 0 and boards.ndim == 3:
            boards = boards[:, np.newaxis, :, :]
        features = np.stack(features_list) if features_list else np.array([])
        actions = np.array(actions_list, dtype=np.int64)
        return boards, features, actions

    def train(self, boards: np.ndarray, features: np.ndarray, actions: np.ndarray,
              epochs: int = 50, batch_size: int = 256):
        """Behavior cloning: supervised learning on expert data."""
        if len(boards) == 0:
            return

        dataset = TensorDataset(
            torch.as_tensor(boards, dtype=torch.float32),
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
        )
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
