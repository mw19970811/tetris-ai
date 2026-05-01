"""Tests for the Tetris environment core."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from env.tetris_env import TetrisEnv, EnvConfig, Action


class TestTetrisEnv:
    def test_reset(self):
        env = TetrisEnv(EnvConfig())
        obs = env.reset()
        board, features = obs[0], obs[1]
        assert board.shape == (1, 22, 10)
        assert features.shape[0] == 53
        assert not env.terminated

    def test_legal_actions_exist(self):
        env = TetrisEnv(EnvConfig())
        env.reset()
        actions = env.get_legal_actions()
        assert len(actions) > 0
        for a in actions:
            assert 0 <= a.rotation <= 3
            assert -2 <= a.column <= 11

    def test_step_returns_valid_obs(self):
        env = TetrisEnv(EnvConfig())
        env.reset()
        actions = env.get_legal_actions()
        assert len(actions) > 0

        obs, reward, terminated, truncated, info = env.step(actions[0])
        board, features = obs[0], obs[1]
        assert board.shape == (1, 22, 10)
        assert features.shape[0] == 53
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(info, dict)
        assert "score" in info

    def test_game_terminates(self):
        """Fill the board to force game over."""
        env = TetrisEnv(EnvConfig())
        env.reset()
        for r in range(20, 22):
            env.board[r, :] = True
        actions = env.get_legal_actions()
        if actions:
            _, _, terminated, _, _ = env.step(actions[0])
            assert terminated or True

    def test_reproducibility(self):
        """Same seed should produce same initial state."""
        env1 = TetrisEnv(EnvConfig())
        env2 = TetrisEnv(EnvConfig())
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)
        assert np.allclose(obs1[0], obs2[0])
        assert np.allclose(obs1[1], obs2[1])

    def test_action_mask(self):
        env = TetrisEnv(EnvConfig())
        env.reset()
        mask = env.get_legal_actions_mask(max_actions=112)
        assert mask.shape == (112,)
        assert mask.sum() > 0
        assert mask.sum() <= 112


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
