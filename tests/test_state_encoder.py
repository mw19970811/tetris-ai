"""Tests for the state encoder module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from env.state_encoder import StateEncoder


class TestStateEncoder:
    def test_hybrid_encoding_output_shapes(self):
        encoder = StateEncoder()
        board = np.zeros((22, 10), dtype=bool)
        board_t, features = encoder.encode(board, 0, 0, -1, [0, 1, 2, 3], True)
        assert board_t.shape == (1, 22, 10)
        assert features.shape == (53,)

    def test_feature_dimension(self):
        encoder = StateEncoder()
        assert encoder.feature_dim == 53

    def test_bitmap_encoding(self):
        encoder = StateEncoder()
        board = np.ones((22, 10), dtype=bool)
        bm = encoder.encode_bitmap(board)
        assert bm.shape == (1, 22, 10)
        assert bm.dtype == np.float32

    def test_hole_counting(self):
        encoder = StateEncoder()
        board = np.zeros((22, 10), dtype=bool)
        # Block at row 10 → holes in rows 11-21 below it = 11 holes.
        board[10, 0] = True
        holes = encoder._count_holes(board)
        assert holes == 11, f"Expected 11 holes, got {holes}"

    def test_hole_counting_multiple_columns(self):
        encoder = StateEncoder()
        board = np.zeros((22, 10), dtype=bool)
        board[5, 0] = True    # → holes in rows 6-21 = 16 holes
        board[10, 1] = True   # → holes in rows 11-21 = 11 holes
        holes = encoder._count_holes(board)
        assert holes == 27

    def test_no_holes_empty_board(self):
        encoder = StateEncoder()
        board = np.zeros((22, 10), dtype=bool)
        assert encoder._count_holes(board) == 0

    def test_bumpiness(self):
        encoder = StateEncoder()
        heights = np.array([5, 5, 10, 10], dtype=np.int32)
        assert encoder._bumpiness(heights) == 5.0

    def test_bumpiness_flat_surface(self):
        encoder = StateEncoder()
        heights = np.ones(10, dtype=np.int32) * 5
        assert encoder._bumpiness(heights) == 0.0

    def test_max_well_depth(self):
        encoder = StateEncoder()
        heights = np.array([10, 5, 10], dtype=np.int32)
        assert encoder._max_well_depth(heights) == 5

    def test_max_well_no_well(self):
        encoder = StateEncoder()
        heights = np.ones(10, dtype=np.int32) * 5
        assert encoder._max_well_depth(heights) == 0

    def test_row_transitions_empty(self):
        encoder = StateEncoder()
        board = np.zeros((22, 10), dtype=bool)
        # Empty board: all cells identical, so zero cell-type changes.
        assert encoder._row_transitions(board) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
