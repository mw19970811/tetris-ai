"""Dellacherie heuristic Tetris AI — standalone inference backend.

Pure-Python implementation of the Dellacherie six-feature evaluation
heuristic.  No GPU required.  Supports:

- Feature-level ablation (enable/disable individual features)
- Placement evaluation with full feature breakdown
- Action selection (argmax over legal placements)
- Episode roll-out with score tracking
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from copy import deepcopy

# Default Dellacherie weights (from particle swarm optimisation).
DEFAULT_WEIGHTS = {
    "landing_height": -4.500,
    "cleared_lines": 3.418,
    "holes": -7.899,
    "bumpiness": -3.386,
    "max_well": -3.129,
    "row_transitions": -2.000,
}

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]

# Piece shapes: [rotation][cell_idx] = (col_offset, row_offset)
_SHAPES = {
    "I": [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)],
          [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]],
    "O": [[(0, 0), (1, 0), (0, 1), (1, 1)]] * 4,
    "T": [[(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 1)],
          [(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (0, 1)]],
    "S": [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)],
          [(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
    "Z": [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (0, 2)],
          [(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (0, 2)]],
    "J": [[(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1), (0, 2)],
          [(0, 0), (1, 0), (2, 0), (2, 1)], [(1, 0), (1, 1), (0, 2), (1, 2)]],
    "L": [[(2, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (0, 1), (0, 2), (1, 2)],
          [(0, 0), (1, 0), (2, 0), (0, 1)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
}


@dataclass
class DellacherieConfig:
    """Configuration with per-feature ablation support.

    Set a weight to 0.0 to disable that feature in the scoring function.
    """
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @property
    def active_features(self) -> List[str]:
        return [k for k in DEFAULT_WEIGHTS if self.weights.get(k, 0.0) != 0.0]

    @property
    def ablated_features(self) -> List[str]:
        return [k for k in DEFAULT_WEIGHTS if self.weights.get(k, 0.0) == 0.0]


# ------------------------------------------------------------------ #
#  Feature computer (stateless — re-used across placements)
# ------------------------------------------------------------------ #

class FeatureComputer:
    """Stateless feature extraction for a 22×10 boolean board."""

    @staticmethod
    def column_heights(board: np.ndarray) -> np.ndarray:
        h = np.zeros(10, dtype=int)
        for c in range(10):
            filled = np.where(board[:, c])[0]
            h[c] = 22 - filled[0] if len(filled) > 0 else 0
        return h

    @staticmethod
    def count_holes(board: np.ndarray) -> int:
        holes = 0
        for c in range(10):
            found = False
            for r in range(22):
                if board[r, c]:
                    found = True
                elif found:
                    holes += 1
        return holes

    @staticmethod
    def bumpiness(heights: np.ndarray) -> int:
        return int(np.sum(np.abs(np.diff(heights))))

    @staticmethod
    def max_well(heights: np.ndarray) -> int:
        w = 0
        for c in range(10):
            ld = heights[c - 1] - heights[c] if c > 0 else 0
            rd = heights[c + 1] - heights[c] if c < 9 else 0
            w = max(w, max(0, min(ld, rd)))
        return w

    @staticmethod
    def row_transitions(board: np.ndarray) -> int:
        t = 0
        for r in range(22):
            for c in range(11):
                left = board[r, c - 1] if c > 0 else False
                right = board[r, c] if c < 10 else False
                if left != right:
                    t += 1
        return t

    @staticmethod
    def clear_lines(board_copy: np.ndarray) -> int:
        cleared = 0
        for r in range(21, -1, -1):
            if np.all(board_copy[r]):
                board_copy[1:r + 1] = board_copy[:r]
                board_copy[0] = False
                cleared += 1
        return cleared

    @staticmethod
    def ghost_y(board: np.ndarray, cells: List[Tuple[int, int]],
                col: int, start_y: int) -> int:
        y = start_y
        while True:
            collision = False
            for cx, cy in cells:
                r, c = y + 1 + cy, col + cx
                if c < 0 or c >= 10 or r >= 22:
                    collision = True
                    break
                if r >= 0 and board[r, c]:
                    collision = True
                    break
            if collision:
                break
            y += 1
        return y

    @staticmethod
    def extract_all(board: np.ndarray) -> Dict[str, float]:
        """Extract all six Dellacherie features from a board."""
        heights = FeatureComputer.column_heights(board)
        return {
            "landing_height": float(np.sum(heights)),
            "cleared_lines": 0.0,  # caller must set after placement
            "holes": float(FeatureComputer.count_holes(board)),
            "bumpiness": float(FeatureComputer.bumpiness(heights)),
            "max_well": float(FeatureComputer.max_well(heights)),
            "row_transitions": float(FeatureComputer.row_transitions(board)),
        }


# ------------------------------------------------------------------ #
#  Dellacherie Agent
# ------------------------------------------------------------------ #

class DellacherieAgent:
    """Dellacherie heuristic agent — placement evaluation + action selection.

    Can be used as:
      - An inference backend (no GPU, no model file)
      - An expert for imitation learning data collection
      - An ablation study driver (disable features via DellacherieConfig)
    """

    def __init__(self, config: Optional[DellacherieConfig] = None):
        self.config = config or DellacherieConfig()
        self.weights = self.config.weights
        self._fc = FeatureComputer()

    # ---- Placement evaluation ----------------------------------------- #

    def evaluate_placement(self, board: np.ndarray, piece_name: str,
                           rotation: int, column: int
                           ) -> Optional[Dict[str, float]]:
        """Simulate placing *piece_name* at (rotation, column) via ghost-drop.

        Returns a dict with all 6 feature values, 'score', and 'valid'=True,
        or None when the placement is impossible.
        """
        cells = _SHAPES[piece_name][rotation]
        gy = self._fc.ghost_y(board, cells, column, 2)
        if gy < 0:
            return None

        sim = board.copy()
        for cx, cy in cells:
            r, c = gy + cy, column + cx
            if 0 <= r < 22 and 0 <= c < 10:
                sim[r, c] = True

        cleared = self._fc.clear_lines(sim)
        heights = self._fc.column_heights(sim)

        features = {
            "landing_height": float(np.sum(heights)),
            "cleared_lines": float(cleared),
            "holes": float(self._fc.count_holes(sim)),
            "bumpiness": float(self._fc.bumpiness(heights)),
            "max_well": float(self._fc.max_well(heights)),
            "row_transitions": float(self._fc.row_transitions(sim)),
        }
        score = sum(self.weights[k] * features[k] for k in features)
        features["score"] = score
        features["valid"] = True
        return features

    # ---- Action selection --------------------------------------------- #

    def select_action(self, board: np.ndarray, legal_actions: List,
                      current_piece_idx: int
                      ) -> Tuple[int, int, int, float, Optional[Dict]]:
        """Select best (rotation, column, hold) via argmax of Dellacherie score.

        Returns (rotation, column, hold, score, features_dict).
        *features_dict* is the detailed breakdown of the winning placement
        (or None when no legal action is viable).
        """
        piece_name = PIECE_NAMES[current_piece_idx]
        best_score = float('-inf')
        best_action = (0, 0, False)
        best_features = None

        for action in legal_actions:
            rot = action.rotation if hasattr(action, 'rotation') else action[0]
            col = action.column if hasattr(action, 'column') else action[1]
            hold = action.hold if hasattr(action, 'hold') else action[2]

            result = self.evaluate_placement(board, piece_name, rot, col)
            if result is None:
                continue
            if result["score"] > best_score:
                best_score = result["score"]
                best_action = (rot, col, hold)
                best_features = result

        return (*best_action, best_score, best_features)

    # ---- Batch evaluation (for comparison / analysis) ----------------- #

    def evaluate_all_placements(self, board: np.ndarray,
                                 legal_actions: List,
                                 current_piece_idx: int
                                 ) -> List[Dict]:
        """Evaluate ALL legal placements and return sorted by score.

        Useful for comparing the Dellacherie ranking against a learned
        policy's Q-value ranking.
        """
        piece_name = PIECE_NAMES[current_piece_idx]
        results = []
        for action in legal_actions:
            rot = action.rotation if hasattr(action, 'rotation') else action[0]
            col = action.column if hasattr(action, 'column') else action[1]
            hold = action.hold if hasattr(action, 'hold') else action[2]

            result = self.evaluate_placement(board, piece_name, rot, col)
            if result is not None:
                result["rotation"] = rot
                result["column"] = col
                result["hold"] = hold
                results.append(result)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ---- Episode roll-out --------------------------------------------- #

    def run_episode(self, env) -> Dict:
        """Play one episode using Dellacherie heuristic, return stats."""
        obs = env.reset()
        board_np, feat_np = obs[0], obs[1]
        total_reward = 0.0
        steps = 0
        transitions = []
        done = False

        while not done:
            legal = env.get_legal_actions()
            if not legal:
                break

            piece_idx = env._current_piece_name_idx
            rot, col, hold, score, features = self.select_action(
                board_np[0].astype(bool), legal, piece_idx
            )

            from env.tetris_env import Action
            from agent.action_mask import encode_action
            action_idx = encode_action(rot, col, hold)

            transitions.append({
                "board": board_np.copy(),
                "features": feat_np.copy(),
                "action_idx": action_idx,
                "dellacherie_score": score,
                "dellacherie_features": features,
            })

            obs, reward, terminated, truncated, info = env.step(Action(rot, col, hold))
            board_np, feat_np = obs[0], obs[1]
            total_reward += reward
            steps += 1
            done = terminated or truncated

        return {
            "score": info.get("score", 0),
            "lines": info.get("lines", 0),
            "level": info.get("level", 1),
            "steps": steps,
            "transitions": transitions,
            "total_reward": total_reward,
        }

    @property
    def active_features(self) -> List[str]:
        return self.config.active_features
