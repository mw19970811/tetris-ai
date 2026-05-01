"""State encoder: converts game state into feature vectors for the neural network.

Provides three encoding schemes:
  A: Raw bitboard (22, 10) binary grid
  B: Handcrafted features (53-dim Dellacherie-style vector)
  C: Hybrid = A + B (default, recommended)
"""

import numpy as np
from typing import Tuple, List


class StateEncoder:
    """Encodes Tetris game state into feature tensors for RL training."""

    # One-hot dimension for each piece type (7 types + empty = 8)
    PIECE_DIM = 7
    HOLD_DIM = 8  # 7 types + empty

    def __init__(self, cols: int = 10, total_rows: int = 22, hidden_rows: int = 2,
                 next_queue_size: int = 4, scheme: str = "hybrid"):
        self.cols = cols
        self.total_rows = total_rows
        self.hidden_rows = hidden_rows
        self.next_queue_size = next_queue_size
        self.scheme = scheme  # "bitmap", "features", "hybrid"

        # Feature dimension
        self.num_handcrafted = 6
        self.feature_dim = (self.num_handcrafted
                            + self.PIECE_DIM      # current piece one-hot
                            + 4                    # current rotation one-hot
                            + self.HOLD_DIM        # hold piece one-hot
                            + next_queue_size * self.PIECE_DIM)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def encode(self, board: np.ndarray, current_piece: int, current_rotation: int,
               hold_piece: int, next_queue: List[int], can_hold: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Encode state. Returns (board_tensor, features_vector)."""
        board_tensor = self.encode_bitmap(board)
        features = self.encode_features(board, current_piece, current_rotation,
                                        hold_piece, next_queue)
        return board_tensor, features

    def encode_bitmap(self, board: np.ndarray) -> np.ndarray:
        """Scheme A: (22, 10) binary grid → (1, 22, 10) float32 tensor for CNN."""
        return board.astype(np.float32)[np.newaxis, :, :]

    def encode_features(self, board: np.ndarray, current_piece: int,
                        current_rotation: int, hold_piece: int,
                        next_queue: List[int]) -> np.ndarray:
        """Scheme B: handcrafted feature vector (53-dim)."""
        feats = []

        # --- Six Dellacherie features ---
        heights = self._column_heights(board)
        feats.append(np.sum(heights))                             # landing/aggregate height (scaled later)
        feats.append(0.0)                                          # lines cleared placeholder (set by caller if needed)
        feats.append(float(self._count_holes(board)))              # holes
        feats.append(self._bumpiness(heights))                     # bumpiness
        feats.append(float(self._max_well_depth(heights)))         # max well depth
        feats.append(0.0)                                          # mean height change placeholder

        # --- Piece identity one-hots ---
        piece_oh = np.zeros(self.PIECE_DIM, dtype=np.float32)
        if 0 <= current_piece < self.PIECE_DIM:
            piece_oh[current_piece] = 1.0
        feats.extend(piece_oh)

        rot_oh = np.zeros(4, dtype=np.float32)
        rot_oh[current_rotation % 4] = 1.0
        feats.extend(rot_oh)

        hold_oh = np.zeros(self.HOLD_DIM, dtype=np.float32)
        if hold_piece < 0 or hold_piece >= self.PIECE_DIM:
            hold_oh[-1] = 1.0  # empty
        else:
            hold_oh[hold_piece] = 1.0
        feats.extend(hold_oh)

        for i in range(self.next_queue_size):
            oh = np.zeros(self.PIECE_DIM, dtype=np.float32)
            if i < len(next_queue) and 0 <= next_queue[i] < self.PIECE_DIM:
                oh[next_queue[i]] = 1.0
            feats.extend(oh)

        return np.array(feats, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Board analysis helpers
    # ------------------------------------------------------------------ #
    def _column_heights(self, board: np.ndarray) -> np.ndarray:
        """Column heights (0 = empty column, 20 = full column)."""
        heights = np.zeros(self.cols, dtype=np.int32)
        for c in range(self.cols):
            filled = np.where(board[:, c] > 0)[0]
            heights[c] = self.total_rows - filled[0] if len(filled) > 0 else 0
        return heights

    def _count_holes(self, board: np.ndarray) -> int:
        """Count empty cells that have a filled cell above them in the same column."""
        holes = 0
        for c in range(self.cols):
            found_block = False
            for r in range(self.total_rows):
                if board[r, c]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def _bumpiness(self, heights: np.ndarray) -> float:
        """Sum of absolute height differences between adjacent columns."""
        return float(np.sum(np.abs(np.diff(heights))))

    def _max_well_depth(self, heights: np.ndarray) -> int:
        """Maximum well depth across all columns."""
        max_well = 0
        n = len(heights)
        for c in range(n):
            left_diff = heights[c - 1] - heights[c] if c > 0 else 0
            right_diff = heights[c + 1] - heights[c] if c < n - 1 else 0
            well = max(0, min(left_diff, right_diff))
            if well > max_well:
                max_well = well
        return max_well

    def _row_transitions(self, board: np.ndarray) -> int:
        """Count horizontal cell-type transitions across all rows."""
        trans = 0
        for r in range(self.total_rows):
            for c in range(self.cols + 1):
                left = board[r, c - 1] > 0 if c > 0 else False
                right = board[r, c] > 0 if c < self.cols else False
                if left != right:
                    trans += 1
        return trans
