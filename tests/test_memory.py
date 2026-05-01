"""Tests for experience replay buffers."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from agent.memory import SumTree, PrioritizedReplayBuffer, UniformReplayBuffer


class TestSumTree:
    def test_add_and_total(self):
        tree = SumTree(100)
        for i in range(50):
            tree.add(1.0, i)
        assert tree.size == 50
        assert abs(tree.total() - 50.0) < 0.01

    def test_sample_distribution(self):
        tree = SumTree(100)
        tree.add(10.0, "high")
        for _ in range(9):
            tree.add(0.1, "low")
        # High-priority item should be sampled more often.
        high_count = 0
        for _ in range(1000):
            s = np.random.uniform(0, tree.total())
            _, _, data = tree.get(s)
            if data == "high":
                high_count += 1
        # Should be sampled roughly 10/(10 + 9*0.1) ≈ 91% of the time.
        assert high_count > 700

    def test_update(self):
        tree = SumTree(100)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        old_total = tree.total()
        tree.update(99, 5.0)  # tree_idx = capacity-1+0 = 99 (first leaf)
        assert abs(tree.total() - (old_total - 1.0 + 5.0)) < 0.01

    def test_capacity_overwrite(self):
        tree = SumTree(5)
        for i in range(10):
            tree.add(float(i + 1), i)
        assert tree.size == 5
        # Only the 5 most recent survive.
        assert float(tree.total()) > 0


class TestPrioritizedReplayBuffer:
    def test_add_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)
        for i in range(100):
            state = (np.zeros((1, 22, 10), dtype=np.float32), np.zeros(53, dtype=np.float32))
            next_state = (np.zeros((1, 22, 10), dtype=np.float32), np.zeros(53, dtype=np.float32))
            buf.add(state, 0, 1.0, next_state, False, td_error=abs(np.random.randn()))

        assert len(buf) == 100
        batch, indices, weights = buf.sample(32, step=1000)
        assert batch["board"].shape[0] == 32
        assert batch["features"].shape[0] == 32
        assert len(batch["actions"]) == 32
        assert weights.shape[0] == 32

    def test_priority_update(self):
        buf = PrioritizedReplayBuffer(capacity=1000, alpha=1.0)
        state = (np.zeros((1, 22, 10), dtype=np.float32), np.zeros(53, dtype=np.float32))
        next_state = (np.zeros((1, 22, 10), dtype=np.float32), np.zeros(53, dtype=np.float32))

        # Add items with given TD errors.
        buf.add(state, 0, 0.0, next_state, False, td_error=10.0)
        buf.add(state, 1, 0.0, next_state, False, td_error=10.0)

        batch, indices, _ = buf.sample(2, step=0)
        assert len(indices) == 2

        # Update with higher TD errors → priorities should increase.
        old_priorities = [buf.tree.tree[int(i)] for i in indices]
        buf.update_priorities(indices, np.array([20.0, 20.0]))
        new_priorities = [buf.tree.tree[int(i)] for i in indices]
        assert all(np > op for np, op in zip(new_priorities, old_priorities))

    def test_beta_annealing(self):
        buf = PrioritizedReplayBuffer(beta_start=0.4, beta_end=1.0, beta_frames=1000)
        assert buf._beta(0) == 0.4
        assert abs(buf._beta(1000) - 1.0) < 0.01
        assert abs(buf._beta(500) - 0.7) < 0.01


class TestUniformReplayBuffer:
    def test_add_and_sample(self):
        buf = UniformReplayBuffer(capacity=1000)
        for i in range(200):
            state = (np.zeros((1, 22, 10), dtype=np.float32), np.zeros(53, dtype=np.float32))
            next_state = (np.zeros((1, 22, 10), dtype=np.float32), np.zeros(53, dtype=np.float32))
            buf.add(state, 0, 1.0, next_state, False)

        assert len(buf) == 200
        batch, _, weights = buf.sample(64)
        assert batch["board"].shape[0] == 64
        assert np.allclose(weights, 1.0)  # uniform weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
