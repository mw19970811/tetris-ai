"""Pure-Python Tetris environment with Gymnasium-compatible interface.

Placement-based action space: each action = (rotation, column, hold_flag).
Optimised for RL training throughput via numpy vectorisation where possible.
"""

import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

from .state_encoder import StateEncoder
from .reward_calculator import RewardCalculator, RewardConfig


# ------------------------------------------------------------------ #
#  Data Classes
# ------------------------------------------------------------------ #
@dataclass
class Action:
    """A placement action: rotate to `rotation`, drop at `column`, optionally hold first."""
    rotation: int
    column: int
    hold: bool = False

    def __hash__(self):
        return hash((self.rotation, self.column, self.hold))

    def __eq__(self, other):
        return (self.rotation, self.column, self.hold) == (other.rotation, other.column, other.hold)


@dataclass
class State:
    """Full observable state for RL agent."""
    board: np.ndarray       # (22, 10) bool
    current_piece: int       # 0-6
    current_rotation: int    # 0-3
    hold_piece: int          # 0-6 or -1
    next_queue: List[int]    # length = next_queue_size
    can_hold: bool


@dataclass
class EnvConfig:
    cols: int = 10
    rows: int = 20
    hidden_rows: int = 2
    lock_delay_ms: int = 500
    lock_moves_max: int = 15
    next_queue_size: int = 4
    bag_type: str = "7bag"
    max_steps: int = 10000
    reward: RewardConfig = field(default_factory=RewardConfig)


# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]

# Each piece: 4 rotations × 4 cells × (col_offset, row_offset)
SHAPES = {
    "I": [[(0,0),(1,0),(2,0),(3,0)], [(0,0),(0,1),(0,2),(0,3)],
          [(0,0),(1,0),(2,0),(3,0)], [(0,0),(0,1),(0,2),(0,3)]],
    "O": [[(0,0),(1,0),(0,1),(1,1)]] * 4,
    "T": [[(0,0),(1,0),(2,0),(1,1)], [(0,0),(0,1),(0,2),(1,1)],
          [(1,0),(0,1),(1,1),(2,1)], [(1,0),(1,1),(1,2),(0,1)]],
    "S": [[(1,0),(2,0),(0,1),(1,1)], [(0,0),(0,1),(1,1),(1,2)],
          [(1,0),(2,0),(0,1),(1,1)], [(0,0),(0,1),(1,1),(1,2)]],
    "Z": [[(0,0),(1,0),(1,1),(2,1)], [(1,0),(0,1),(1,1),(0,2)],
          [(0,0),(1,0),(1,1),(2,1)], [(1,0),(0,1),(1,1),(0,2)]],
    "J": [[(0,0),(0,1),(1,1),(2,1)], [(0,0),(1,0),(0,1),(0,2)],
          [(0,0),(1,0),(2,0),(2,1)], [(1,0),(1,1),(0,2),(1,2)]],
    "L": [[(2,0),(0,1),(1,1),(2,1)], [(0,0),(0,1),(0,2),(1,2)],
          [(0,0),(1,0),(2,0),(0,1)], [(0,0),(1,0),(1,1),(1,2)]],
}

# SRS wall kick tables
WALL_KICKS_NORMAL = {
    (0,1): [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
    (1,0): [(0,0),(1,0),(1,1),(0,-2),(1,-2)],
    (1,2): [(0,0),(1,0),(1,1),(0,-2),(1,-2)],
    (2,1): [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
    (2,3): [(0,0),(1,0),(1,-1),(0,2),(1,2)],
    (3,2): [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
    (3,0): [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
    (0,3): [(0,0),(1,0),(1,-1),(0,2),(1,2)],
}
WALL_KICKS_I = {
    (0,1): [(0,0),(-2,0),(1,0),(-2,1),(1,-2)],
    (1,0): [(0,0),(2,0),(-1,0),(2,-1),(-1,2)],
    (1,2): [(0,0),(-1,0),(2,0),(-1,-2),(2,1)],
    (2,1): [(0,0),(1,0),(-2,0),(1,2),(-2,-1)],
    (2,3): [(0,0),(2,0),(-1,0),(2,-1),(-1,2)],
    (3,2): [(0,0),(-2,0),(1,0),(-2,1),(1,-2)],
    (3,0): [(0,0),(1,0),(-2,0),(1,2),(-2,-1)],
    (0,3): [(0,0),(-1,0),(2,0),(-1,-2),(2,1)],
}

SCORE_TABLE = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
DROP_SPEEDS = [1000,800,650,500,400,320,250,180,130,90,70,55,45,35,28,22,17,14,11,9]


# ------------------------------------------------------------------ #
#  TetrisEnv
# ------------------------------------------------------------------ #
class TetrisEnv:
    """Gymnasium-style Tetris environment with placement-based actions."""

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.cfg = config
        self.total_rows = config.rows + config.hidden_rows
        self.encoder = StateEncoder(config.cols, self.total_rows, config.hidden_rows,
                                     config.next_queue_size, scheme="hybrid")
        self.reward_calc = RewardCalculator(config.reward)
        self.reset()

    # ------------------------------------------------------------------ #
    #  Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Reset environment. Returns (board_tensor, features_vector, info)."""
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

        self.board = np.zeros((self.total_rows, self.cfg.cols), dtype=bool)
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.step_count = 0
        self._terminated = False
        self._bag = []
        self._next_queue = deque(maxlen=self.cfg.next_queue_size + 1)
        self._hold_piece: int = -1
        self._can_hold = True

        self._fill_bag()
        for _ in range(self.cfg.next_queue_size + 1):
            self._next_queue.append(self._next_from_bag())

        self._spawn_piece()
        return self.get_obs() + ({},)

    def step(self, action: Action) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, bool, Dict]:
        """Execute a placement action. Returns (obs, reward, terminated, truncated, info)."""
        if self._terminated:
            return self.get_obs(), 0.0, True, False, {"score": self.score}

        reward = 0.0
        piece_name = self._current_piece_name

        # --- Handle Hold ---
        if action.hold:
            if self._can_hold:
                if self._hold_piece < 0:
                    self._hold_piece = PIECE_NAMES.index(self._current_piece_name) if isinstance(self._current_piece_name, str) else self._current_piece_name
                    self._can_hold = False
                    self._spawn_piece()
                    obs = self.get_obs()
                    return obs, self.cfg.reward.w_survival, False, self.step_count >= self.cfg.max_steps, self._info()
                else:
                    held = self._hold_piece
                    self._hold_piece = PIECE_NAMES.index(self._current_piece_name) if isinstance(self._current_piece_name, str) else self._current_piece_name
                    self._current_piece_name = PIECE_NAMES[held] if isinstance(held, int) else held
                    piece_name = self._current_piece_name
                    self._can_hold = False

        # --- Place piece at ghost position ---
        cells = SHAPES[piece_name][action.rotation]
        ghost_y = self._ghost_y(cells, action.column, self.cfg.hidden_rows - 2)
        drop_distance = ghost_y - (self.cfg.hidden_rows - 2)
        reward += drop_distance * self.cfg.reward.hard_drop_score

        self._place(cells, action.column, ghost_y)
        self._current_piece_name = None

        # --- Game over? ---
        if self._blocks_in_hidden_rows():
            self._terminated = True
            reward += self.cfg.reward.w_death
            self.score += int(reward)
            return self.get_obs(), reward, True, False, self._info()

        # --- Clear lines ---
        cleared = self._clear_lines()
        if cleared > 0:
            reward += SCORE_TABLE[cleared] * self.level
            self.lines_cleared += cleared
            self.level = self.lines_cleared // 10 + 1

        # --- Board quality reward shaping ---
        heights = self._column_heights()
        reward -= self.cfg.reward.w_height * float(np.sum(heights))
        reward -= self.cfg.reward.w_holes * float(self._count_holes())
        reward -= self.cfg.reward.w_bumpiness * float(np.sum(np.abs(np.diff(heights))))
        reward -= self.cfg.reward.w_well * float(self._max_well(heights))
        reward += self.cfg.reward.w_survival

        self.score += int(reward)
        self.step_count += 1

        # --- Spawn next ---
        self._spawn_piece()

        truncated = self.step_count >= self.cfg.max_steps
        return self.get_obs(), reward, False, truncated, self._info()

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (board_tensor, features_vector)."""
        ci = PIECE_NAMES.index(self._current_piece_name) if self._current_piece_name else -1
        return self.encoder.encode(self.board, ci if ci >= 0 else 0,
                                    self._current_rotation,
                                    self._hold_piece if self._hold_piece >= 0 else -1,
                                    list(self._next_queue),
                                    self._can_hold)

    def get_legal_actions(self) -> List[Action]:
        """Get list of legal (rotation, column, hold) actions for current state."""
        if self._terminated or not self._current_piece_name:
            return []

        actions = []
        piece_name = self._current_piece_name
        current = PIECE_NAMES.index(piece_name)

        # Non-hold actions.
        for rot in range(4):
            cells = SHAPES[piece_name][rot]
            for col in range(-2, self.cfg.cols + 2):
                if self._collides(cells, col, self.cfg.hidden_rows - 2):
                    continue
                gy = self._ghost_y(cells, col, self.cfg.hidden_rows - 2)
                if gy >= 0 and not self._collides(cells, col, gy):
                    actions.append(Action(rot, col, hold=False))

        # Hold actions.
        if self._can_hold:
            if self._hold_piece < 0:
                actions.append(Action(0, 0, hold=True))
            else:
                alt_name = PIECE_NAMES[self._hold_piece]
                for rot in range(4):
                    cells = SHAPES[alt_name][rot]
                    for col in range(-2, self.cfg.cols + 2):
                        if self._collides(cells, col, self.cfg.hidden_rows - 2):
                            continue
                        gy = self._ghost_y(cells, col, self.cfg.hidden_rows - 2)
                        if gy >= 0 and not self._collides(cells, col, gy):
                            actions.append(Action(rot, col, hold=True))

        return actions

    def get_legal_actions_mask(self, max_actions: int = 41) -> np.ndarray:
        """Boolean mask of shape (max_actions,) with True for legal actions.

        Action index encoding: idx = rotation * 10 + column + (hold ? 40 : 0).
        column is shifted by +2 to make 0-based.
        """
        mask = np.zeros(max_actions, dtype=bool)
        for a in self.get_legal_actions():
            col_idx = a.column + 2
            if 0 <= col_idx < 10:
                idx = a.rotation * 10 + col_idx
                if a.hold:
                    idx += 40
                if idx < max_actions:
                    mask[idx] = True
            else:
                # Out-of-range columns: assign to nearest bucket.
                col_idx = max(0, min(9, col_idx))
                idx = a.rotation * 10 + col_idx
                if a.hold:
                    idx += 40
                if idx < max_actions:
                    mask[idx] = True
        return mask

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #
    @property
    def terminated(self) -> bool: return self._terminated

    @property
    def _current_piece_name_idx(self) -> int:
        return PIECE_NAMES.index(self._current_piece_name) if self._current_piece_name else -1

    @property
    def observation_space_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return ((1, self.total_rows, self.cfg.cols), (self.encoder.feature_dim,))

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #
    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        lines = []
        for r in range(self.cfg.hidden_rows, self.total_rows):
            row_str = ""
            for c in range(self.cfg.cols):
                row_str += "[]" if self.board[r, c] else " ."
            lines.append(row_str)
        lines.append(f"Score: {self.score}  Level: {self.level}  Lines: {self.lines_cleared}")
        if self._current_piece_name:
            lines.append(f"Current: {self._current_piece_name}  Hold: {PIECE_NAMES[self._hold_piece] if self._hold_piece >= 0 else '-'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Internal: Board operations
    # ------------------------------------------------------------------ #
    def _place(self, cells, col, row):
        for cx, cy in cells:
            r, c = row + cy, col + cx
            if 0 <= r < self.total_rows and 0 <= c < self.cfg.cols:
                self.board[r, c] = True

    def _collides(self, cells, col, row) -> bool:
        for cx, cy in cells:
            r, c = row + cy, col + cx
            if c < 0 or c >= self.cfg.cols or r >= self.total_rows:
                return True
            if r >= 0 and self.board[r, c]:
                return True
        return False

    def _ghost_y(self, cells, col, start_y) -> int:
        y = start_y
        while not self._collides(cells, col, y + 1):
            y += 1
        return y

    def _clear_lines(self) -> int:
        cleared = 0
        r = self.total_rows - 1
        while r >= 0:
            if np.all(self.board[r]):
                self.board[1:r+1] = self.board[:r]
                self.board[0] = False
                cleared += 1
            else:
                r -= 1
        return cleared

    def _blocks_in_hidden_rows(self) -> bool:
        return bool(np.any(self.board[:self.cfg.hidden_rows]))

    def _column_heights(self) -> np.ndarray:
        h = np.full(self.cfg.cols, self.total_rows)
        for c in range(self.cfg.cols):
            filled = np.where(self.board[:, c])[0]
            if len(filled) > 0:
                h[c] = filled[0]
        return self.total_rows - h

    def _count_holes(self) -> int:
        holes = 0
        for c in range(self.cfg.cols):
            found = False
            for r in range(self.total_rows):
                if self.board[r, c]:
                    found = True
                elif found:
                    holes += 1
        return holes

    def _max_well(self, heights: np.ndarray) -> int:
        w = 0
        for c in range(self.cfg.cols):
            ld = heights[c-1] - heights[c] if c > 0 else 0
            rd = heights[c+1] - heights[c] if c < self.cfg.cols - 1 else 0
            w = max(w, max(0, min(ld, rd)))
        return w

    # ------------------------------------------------------------------ #
    #  Internal: Bag & Spawn
    # ------------------------------------------------------------------ #
    def _fill_bag(self):
        self._bag = list(range(7))
        np.random.shuffle(self._bag)

    def _next_from_bag(self) -> int:
        if not self._bag:
            self._fill_bag()
        return self._bag.pop()

    def _spawn_piece(self):
        if self._next_queue:
            idx = self._next_queue.popleft()
            self._next_queue.append(self._next_from_bag())
            self._current_piece_name = PIECE_NAMES[idx]
            self._current_rotation = 0
            self._can_hold = True
            cells = SHAPES[self._current_piece_name][0]
            if self._collides(cells, 3, self.cfg.hidden_rows - 2):
                self._terminated = True
        else:
            self._terminated = True

    def _info(self) -> Dict:
        return {"score": self.score, "level": self.level, "lines": self.lines_cleared,
                "steps": self.step_count, "terminated": self._terminated}


# ------------------------------------------------------------------ #
#  Vectorized environment (simple version)
# ------------------------------------------------------------------ #
class VectorTetrisEnv:
    """Multiple independent Tetris environments for parallel sampling."""

    def __init__(self, num_envs: int, config: EnvConfig = EnvConfig()):
        self.num_envs = num_envs
        self.envs = [TetrisEnv(deepcopy(config)) for _ in range(num_envs)]

    def reset(self, seeds: Optional[List[int]] = None):
        if seeds is None:
            seeds = [None] * self.num_envs
        results = [env.reset(seed=s) for env, seed in zip(self.envs, seeds)]
        boards = np.stack([r[0] for r in results])
        features = np.stack([r[1] for r in results])
        return boards, features

    def step(self, actions: List[Action]):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        boards = np.stack([r[0][0] for r in results])
        features = np.stack([r[0][1] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminated = np.array([r[2] for r in results], dtype=bool)
        truncated = np.array([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        # Auto-reset terminated environments.
        for i, t in enumerate(terminated):
            if t:
                obs = self.envs[i].reset()
                boards[i] = obs[0]
                features[i] = obs[1]
        return boards, features, rewards, terminated, truncated, infos

    def get_legal_actions_masks(self, max_actions: int = 41) -> np.ndarray:
        masks = [env.get_legal_actions_mask(max_actions) for env in self.envs]
        return np.stack(masks)
