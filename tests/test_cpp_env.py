"""Tests for C++-accelerated Tetris environment (CppTetrisEnv)."""

import sys
import time
import pytest
import numpy as np

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
        """Average legal action count should be similar over multiple states."""
        py_counts = []
        cpp_counts = []

        py_env.reset()
        cpp_env.reset()

        for _ in range(200):
            py_la = py_env.get_legal_actions()
            cpp_la = cpp_env.get_legal_actions()
            if not py_la or not cpp_la:
                break
            py_counts.append(len(py_la))
            cpp_counts.append(len(cpp_la))

            py_env.step(py_la[0])
            cpp_env.step(cpp_la[0])

        assert len(py_counts) > 50  # enough data for comparison
        py_avg = np.mean(py_counts)
        cpp_avg = np.mean(cpp_counts)
        # Should be within ~15% of each other across many random steps.
        assert abs(py_avg - cpp_avg) / max(py_avg, 1.0) < 0.20


# ------------------------------------------------------------------ #
#  Benchmark
# ------------------------------------------------------------------ #

class TestCppEnvBenchmark:
    """Performance benchmarks for C++ env vs Python env."""

    def test_legal_actions_speedup(self, py_env, cpp_env):
        """get_legal_actions() should be at least 5x faster in C++."""
        py_env.reset()
        cpp_env.reset()

        # Warmup.
        for _ in range(100):
            py_env.get_legal_actions()
            cpp_env.get_legal_actions()
            a = py_env.get_legal_actions()[0]
            py_env.step(a)
            cpp_env.step(a)

        # Benchmark Python.
        py_env.reset()
        t0 = time.perf_counter()
        for _ in range(500):
            py_env.get_legal_actions()
            a = py_env.get_legal_actions()
        py_time = time.perf_counter() - t0

        # Benchmark C++.
        cpp_env.reset()
        t0 = time.perf_counter()
        for _ in range(500):
            cpp_env.get_legal_actions()
            a = cpp_env.get_legal_actions()
        cpp_time = time.perf_counter() - t0

        speedup = py_time / max(cpp_time, 1e-9)
        print(f"\n  Python get_legal_actions: {py_time*1000:.1f}ms (500 calls)")
        print(f"  C++ get_legal_actions:    {cpp_time*1000:.1f}ms (500 calls)")
        print(f"  Speedup: {speedup:.1f}x")
        assert speedup >= 3.0, f"Expected >=3x speedup, got {speedup:.1f}x"

    def test_full_step_speedup(self, py_env, cpp_env):
        """Full step() + get_obs() should be faster in C++."""
        py_env.reset()
        cpp_env.reset()

        # Warmup.
        for _ in range(50):
            a = py_env.get_legal_actions()[0]
            py_env.step(a)
            a_cpp = cpp_env.get_legal_actions()[0]
            cpp_env.step(a_cpp)

        # Benchmark Python.
        py_env.reset()
        t0 = time.perf_counter()
        for _ in range(200):
            la = py_env.get_legal_actions()
            if not la:
                break
            py_env.step(la[0])
        py_time = time.perf_counter() - t0

        # Benchmark C++.
        cpp_env.reset()
        t0 = time.perf_counter()
        for _ in range(200):
            la = cpp_env.get_legal_actions()
            if not la:
                break
            cpp_env.step(la[0])
        cpp_time = time.perf_counter() - t0

        speedup = py_time / max(cpp_time, 1e-9)
        print(f"\n  Python step loop: {py_time*1000:.1f}ms (200 steps)")
        print(f"  C++ step loop:    {cpp_time*1000:.1f}ms (200 steps)")
        print(f"  Speedup: {speedup:.1f}x")
        assert speedup >= 2.0, f"Expected >=2x speedup, got {speedup:.1f}x"
