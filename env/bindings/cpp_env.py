"""C++-accelerated Tetris environment via pybind11.

Drop-in replacement for env.tetris_env.TetrisEnv. Delegates all heavy
computation (collision detection, feature encoding, action generation)
to the compiled C++ core while keeping the identical Python interface.

Usage:
    from env.bindings.cpp_env import CppTetrisEnv
    env = CppTetrisEnv(config)
    board, features = env.reset()
    (board, features), reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from ..tetris_env import Action, EnvConfig


class CppTetrisEnv:
    """Drop-in replacement for TetrisEnv using C++ backend.

    Implements the same public interface as env.tetris_env.TetrisEnv
    but delegates game logic, action generation, and feature encoding
    to the compiled C++ tetris_core module.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 60}

    def __init__(self, config: EnvConfig = None):
        if config is None:
            config = EnvConfig()

        self.cfg = config

        # Map Python EnvConfig -> C++ EnvConfig.
        # The compiled C++ module. Try multiple import paths:
        # 1. Installed as a package (pip install -e .)
        # 2. Built in build/ directory (add to PYTHONPATH)
        # 3. Copied alongside this file
        try:
            from . import tetris_core as tc
        except ImportError:
            try:
                import tetris_core as tc  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "Cannot import tetris_core C++ module. "
                    "Build it with: mkdir -p build && cd build && "
                    "cmake .. && cmake --build . --target tetris_core\n"
                    "Then add build/env/bindings to PYTHONPATH."
                )
        import sys
        _platform = "Windows" if sys.platform == "win32" else "Linux"
        print(f"[CppTetrisEnv] tetris_core C++ module loaded successfully "
              f"(platform: {_platform}, module: {tc.__name__})")
        cpp_cfg = tc.EnvConfig()
        cpp_cfg.cols = config.cols
        cpp_cfg.rows = config.rows
        cpp_cfg.hidden_rows = config.hidden_rows
        cpp_cfg.lock_delay_ms = config.lock_delay_ms
        cpp_cfg.lock_moves_max = config.lock_moves_max
        cpp_cfg.next_queue_size = config.next_queue_size
        cpp_cfg.bag_type = config.bag_type

        # Reward weights.
        rw = config.reward
        cpp_cfg.w_height = rw.w_height
        cpp_cfg.w_holes = rw.w_holes
        cpp_cfg.w_bumpiness = rw.w_bumpiness
        cpp_cfg.w_well = rw.w_well
        cpp_cfg.w_survival = rw.w_survival
        cpp_cfg.w_death = rw.w_death
        cpp_cfg.hard_drop_score = rw.hard_drop_score
        cpp_cfg.soft_drop_score = rw.soft_drop_score

        self._tc = tc
        self._cpp_env = tc.TetrisEnvCpp(cpp_cfg)
        self._encoder = tc.StateEncoder(
            config.cols,
            config.rows + config.hidden_rows,
            config.hidden_rows,
            config.next_queue_size,
        )

        self._max_steps = config.max_steps
        self._terminated = False
        self._step_count = 0
        self._score = 0

        # Store for board property (updated by _sync_state).
        self._board_np = np.zeros(
            (config.rows + config.hidden_rows, config.cols), dtype=bool
        )

    # ------------------------------------------------------------------ #
    #  Core API (Gymnasium-compatible)
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Reset environment. Returns (board_tensor, features_vector, info)."""
        if seed is not None:
            cpp_cfg = self._cpp_env.config()
            self._cpp_env = self._tc.TetrisEnvCpp(cpp_cfg, seed)
        else:
            self._cpp_env.reset()

        self._terminated = False
        self._step_count = 0
        self._score = 0

        board, features = self._cpp_env.get_obs(self._encoder)
        self._sync_board(board)
        return board, features, {}

    def step(self, action: Action) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, bool, Dict]:
        """Execute a placement action. Returns (obs, reward, terminated, truncated, info)."""
        if self._terminated:
            board, features = self._cpp_env.get_obs(self._encoder)
            return (board, features), 0.0, True, False, self._info()

        # Convert Python Action -> C++ Action.
        cpp_action = self._tc.Action()
        cpp_action.rotation = action.rotation
        cpp_action.column = action.column
        cpp_action.hold = action.hold

        reward = self._cpp_env.step(cpp_action)
        self._terminated = self._cpp_env.is_terminated()
        self._score = self._cpp_env.get_score()
        self._step_count = self._cpp_env.get_state().step_count

        board, features = self._cpp_env.get_obs(self._encoder)
        self._sync_board(board)

        truncated = self._step_count >= self._max_steps
        return (board, features), reward, self._terminated, truncated, self._info()

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (board_tensor, features_vector) for current state."""
        return self._cpp_env.get_obs(self._encoder)

    # ------------------------------------------------------------------ #
    #  Action Interface
    # ------------------------------------------------------------------ #

    def get_legal_actions(self) -> List[Action]:
        """Get list of legal (rotation, column, hold) actions."""
        if self._terminated:
            return []
        cpp_actions = self._cpp_env.get_legal_actions()
        return [Action(a.rotation, a.column, a.hold) for a in cpp_actions]

    def get_legal_actions_mask(self, max_actions: int = 112) -> np.ndarray:
        """Boolean mask of shape (max_actions,) with True for legal actions.

        Uses 10-column bucket encoding (matching Python get_legal_actions_mask):
          idx = rotation * 10 + (column + 2) + (hold ? 40 : 0)
        """
        return self._cpp_env.get_legal_actions_mask(max_actions)

    # ------------------------------------------------------------------ #
    #  State Access
    # ------------------------------------------------------------------ #

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def board(self) -> np.ndarray:
        """Current board as (22, 10) bool array (copy of internal state)."""
        return self._board_np

    @property
    def score(self) -> int:
        return self._score

    @property
    def _current_piece_name_idx(self) -> int:
        """Current piece index (0-6), matching TetrisEnv interface."""
        return int(self._cpp_env.get_state().current_piece)

    # ------------------------------------------------------------------ #
    #  Render
    # ------------------------------------------------------------------ #

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """Render board as ASCII art."""
        state = self._cpp_env.get_state()
        board_np, _ = self._cpp_env.get_obs(self._encoder)
        board_bool = board_np[0] > 0.5

        lines = []
        for r in range(self.cfg.hidden_rows, self.cfg.rows + self.cfg.hidden_rows):
            row_str = "".join("[]" if board_bool[r, c] else " ." for c in range(self.cfg.cols))
            lines.append(row_str)

        hold_idx = int(state.hold_piece)
        cur_idx = int(state.current_piece)
        hold_name = self._tc.PIECE_NAMES[hold_idx] if hold_idx < 7 else "-"
        cur_name = self._tc.PIECE_NAMES[cur_idx] if cur_idx < 7 else "-"

        lines.append(
            f"Score: {self._score}  Level: {self._cpp_env.get_level()}  "
            f"Lines: {self._cpp_env.get_lines_cleared()}"
        )
        lines.append(f"Current: {cur_name}  Hold: {hold_name}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Internal Helpers
    # ------------------------------------------------------------------ #

    def _sync_board(self, board_tensor: np.ndarray):
        """Sync cached board from C++ state."""
        self._board_np = board_tensor[0] > 0.5

    def _info(self) -> Dict:
        return {
            "score": self._score,
            "level": self._cpp_env.get_level(),
            "lines": self._cpp_env.get_lines_cleared(),
            "steps": self._step_count,
            "terminated": self._terminated,
        }
