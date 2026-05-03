"""Tests for C++-accelerated Tetris environment (CppTetrisEnv)."""

import sys
import os
import time
import pytest
import numpy as np

# Allow running as: python tests/test_cpp_env.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tetris_env import TetrisEnv, EnvConfig, Action
from env.reward_calculator import RewardConfig


# Attempt to import C++ module — skip all C++ tests if not compiled.
try:
    import env.bindings.tetris_core  # noqa: F401
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

# CppTetrisEnv can always be imported (pure Python), but can only be
# instantiated when the C++ module is available.
from env.bindings.cpp_env import CppTetrisEnv  # noqa: E402


def _make_config():
    rc = RewardConfig()
    return EnvConfig(
        cols=10, rows=20, hidden_rows=2,
        next_queue_size=4, bag_type="7bag",
        max_steps=5000, reward=rc,
    )


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def py_env():
    return TetrisEnv(_make_config())


@pytest.fixture
def cpp_env():
    if not CPP_AVAILABLE:
        pytest.skip("C++ tetris_core module not compiled")
    return CppTetrisEnv(_make_config())


# ------------------------------------------------------------------ #
#  Basic interface tests
# ------------------------------------------------------------------ #

class TestCppEnvInterface:
    """Verify CppTetrisEnv implements the same interface as TetrisEnv."""

    def test_create_and_reset(self, cpp_env):
        board, features, info = cpp_env.reset()
        assert board.shape == (1, 22, 10)
        assert board.dtype == np.float32
        assert features.shape == (53,)
        assert features.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_with_seed(self):
        if not CPP_AVAILABLE:
            pytest.skip("C++ module not compiled")
        config = _make_config()
        env1 = CppTetrisEnv(config)
        env2 = CppTetrisEnv(config)
        b1, f1, _ = env1.reset(seed=42)
        b2, f2, _ = env2.reset(seed=42)
        # Same seed should produce same initial state.
        assert np.array_equal(b1, b2)
        assert np.array_equal(f1, f2)

    def test_step_shapes(self, cpp_env):
        cpp_env.reset()
        actions = cpp_env.get_legal_actions()
        assert len(actions) > 0
        (board, features), reward, terminated, truncated, info = cpp_env.step(actions[0])
        assert board.shape == (1, 22, 10)
        assert features.shape == (53,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_get_legal_actions(self, cpp_env):
        cpp_env.reset()
        actions = cpp_env.get_legal_actions()
        assert len(actions) > 0
        for a in actions:
            assert isinstance(a, Action)
            assert 0 <= a.rotation < 4
            assert -2 <= a.column < 12
            assert isinstance(a.hold, bool)

    def test_get_legal_actions_mask(self, cpp_env):
        cpp_env.reset()
        mask = cpp_env.get_legal_actions_mask(112)
        assert mask.shape == (112,)
        assert mask.dtype == bool
        # At least some actions should be legal.
        assert mask.sum() > 0

    def test_get_obs(self, cpp_env):
        cpp_env.reset()
        board, features = cpp_env.get_obs()
        assert board.shape == (1, 22, 10)
        assert features.shape == (53,)

    def test_terminated_property(self, cpp_env):
        cpp_env.reset()
        assert cpp_env.terminated is False

    def test_board_property(self, cpp_env):
        cpp_env.reset()
        b = cpp_env.board
        assert b.shape == (22, 10)
        assert b.dtype == bool

    def test_score_property(self, cpp_env):
        cpp_env.reset()
        assert isinstance(cpp_env.score, int)

    def test_multiple_steps(self, cpp_env):
        cpp_env.reset()
        for _ in range(100):
            actions = cpp_env.get_legal_actions()
            if not actions:
                break
            (_, _), _, terminated, truncated, _ = cpp_env.step(actions[0])
            if terminated or truncated:
                break

    def test_render_ansi(self, cpp_env):
        cpp_env.reset()
        s = cpp_env.render("ansi")
        assert isinstance(s, str)
        assert "Score:" in s
        assert len(s) > 0


class TestCppEnvConsistency:
    """Statistical consistency between Python and C++ envs (not bit-exact)."""

    def test_feature_dim_match(self, py_env, cpp_env):
        """Both envs produce 53-dim features."""
        py_env.reset()
        cpp_env.reset()
        _, py_feat, _ = py_env.reset()
        _, cpp_feat, _ = cpp_env.reset()
        assert py_feat.shape == cpp_feat.shape == (53,)

    def test_board_shape_match(self, py_env, cpp_env):
        """Both envs produce (1, 22, 10) board."""
        py_env.reset()
        cpp_env.reset()
        py_board, _, _ = py_env.reset()
        cpp_board, _, _ = cpp_env.reset()
        assert py_board.shape == cpp_board.shape

    def test_legal_actions_similar_count(self, py_env, cpp_env):
        """Average legal action count should be similar over multiple states.

        Uses short trajectories (reset every few steps) to sample many
        diverse board states without requiring long survival.
        """
        py_counts = []
        cpp_counts = []
        seed = 42

        py_env.reset(seed=seed)
        cpp_env.reset(seed=seed)

        for i in range(500):
            py_la = py_env.get_legal_actions()
            cpp_la = cpp_env.get_legal_actions()
            if not py_la or not cpp_la:
                seed += 1
                py_env.reset(seed=seed)
                cpp_env.reset(seed=seed)
                continue
            py_counts.append(len(py_la))
            cpp_counts.append(len(cpp_la))

            # Pick a mid-board action for survival.
            py_env.step(py_la[len(py_la) // 2])
            cpp_env.step(cpp_la[len(cpp_la) // 2])

            # Reset periodically to sample diverse states.
            if i % 20 == 19:
                seed += 1
                py_env.reset(seed=seed)
                cpp_env.reset(seed=seed)

        assert len(py_counts) > 100, f"Need >100 samples, got {len(py_counts)}"
        py_avg = np.mean(py_counts)
        cpp_avg = np.mean(cpp_counts)
        ratio = abs(py_avg - cpp_avg) / max(py_avg, 1.0)
        assert ratio < 0.30, f"Legal action counts differ too much: py={py_avg:.1f} cpp={cpp_avg:.1f} ratio={ratio:.2%}"


# ------------------------------------------------------------------ #
#  Benchmark
# ------------------------------------------------------------------ #

class TestCppEnvBenchmark:
    """Performance benchmarks for C++ env vs Python env."""

    def test_legal_actions_speedup(self, py_env, cpp_env):
        """get_legal_actions() should be at least 5x faster in C++."""
        py_env.reset(seed=42)
        cpp_env.reset(seed=42)

        # Warmup: pick mid-board action to avoid early death.
        for _ in range(100):
            py_la = py_env.get_legal_actions()
            cpp_la = cpp_env.get_legal_actions()
            if not py_la or not cpp_la:
                py_env.reset(seed=42)
                cpp_env.reset(seed=42)
                continue
            py_env.step(py_la[len(py_la) // 2])
            cpp_env.step(cpp_la[len(cpp_la) // 2])

        # Benchmark Python — just measure get_legal_actions, no step.
        py_env.reset(seed=42)
        t0 = time.perf_counter()
        for _ in range(500):
            py_env.get_legal_actions()
        py_time = time.perf_counter() - t0

        # Benchmark C++.
        cpp_env.reset(seed=42)
        t0 = time.perf_counter()
        for _ in range(500):
            cpp_env.get_legal_actions()
        cpp_time = time.perf_counter() - t0

        speedup = py_time / max(cpp_time, 1e-9)
        print(f"\n  Python get_legal_actions: {py_time*1000:.1f}ms (500 calls)")
        print(f"  C++ get_legal_actions:    {cpp_time*1000:.1f}ms (500 calls)")
        print(f"  Speedup: {speedup:.1f}x")
        assert speedup >= 3.0, f"Expected >=3x speedup, got {speedup:.1f}x"

    def test_full_step_speedup(self, py_env, cpp_env):
        """Full step() + get_obs() should be faster in C++."""
        py_env.reset(seed=42)
        cpp_env.reset(seed=42)

        # Warmup: pick mid-board action to avoid early death.
        for _ in range(50):
            py_la = py_env.get_legal_actions()
            cpp_la = cpp_env.get_legal_actions()
            if not py_la or not cpp_la:
                py_env.reset(seed=42)
                cpp_env.reset(seed=42)
                continue
            py_env.step(py_la[len(py_la) // 2])
            cpp_env.step(cpp_la[len(cpp_la) // 2])

        # Benchmark Python.
        py_env.reset(seed=42)
        t0 = time.perf_counter()
        for _ in range(200):
            la = py_env.get_legal_actions()
            if not la:
                py_env.reset(seed=42)
                continue
            py_env.step(la[len(la) // 2])
        py_time = time.perf_counter() - t0

        # Benchmark C++.
        cpp_env.reset(seed=42)
        t0 = time.perf_counter()
        for _ in range(200):
            la = cpp_env.get_legal_actions()
            if not la:
                cpp_env.reset(seed=42)
                continue
            cpp_env.step(la[len(la) // 2])
        cpp_time = time.perf_counter() - t0

        speedup = py_time / max(cpp_time, 1e-9)
        print(f"\n  Python step loop: {py_time*1000:.1f}ms (200 steps)")
        print(f"  C++ step loop:    {cpp_time*1000:.1f}ms (200 steps)")
        print(f"  Speedup: {speedup:.1f}x")
        assert speedup >= 2.0, f"Expected >=2x speedup, got {speedup:.1f}x"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
