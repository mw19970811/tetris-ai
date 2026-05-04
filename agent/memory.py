"""Prioritized Experience Replay buffer with SumTree for O(log N) sampling.

Reference: Schaul et al. (2016), "Prioritized Experience Replay"
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any


class StoredTransition:
    """Lightweight transition storage using __slots__ (no dict overhead)."""
    __slots__ = ('board', 'features', 'action', 'reward',
                 'next_board', 'next_features', 'done')

    def __init__(self, board, features, action, reward,
                 next_board, next_features, done):
        self.board = board
        self.features = features
        self.action = action
        self.reward = reward
        self.next_board = next_board
        self.next_features = next_features
        self.done = done


class SumTree:
    """Binary sum-tree for efficient weighted sampling.

    Leaf nodes store priorities; internal nodes store sums.
    All operations are O(log N).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0

    def add(self, priority: float, data: Any):
        """Add data with given priority, overwriting oldest if full."""
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority at given tree index."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, value: float) -> Tuple[int, float, Any]:
        """Find leaf node such that cumulative priority >= value.

        Returns (tree_index, priority, data).
        """
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self) -> float:
        return self.tree[0]

    def max_priority(self) -> float:
        return np.max(self.tree[self.capacity - 1:self.capacity - 1 + self.size]) if self.size > 0 else 1.0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent > 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    """PER buffer with hybrid priority and importance-sampling correction.

    Hybrid priority:  p = max( |td|^alpha,  |reward| * reward_weight )
    This prevents high-reward transitions (line clears) from being
    drowned out by high-TD-error transitions (unexpected deaths).
    """

    def __init__(self, capacity: int = 1_000_000, alpha: float = 0.8,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_frames: int = 10_000_000, epsilon: float = 1e-6,
                 reward_weight: float = 0.5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.reward_weight = reward_weight
        self.max_priority = 1.0
        self._init_priority = 1.0  # New transitions start at 1.0, not max

    def _compute_priority(self, td_error: float, reward: float = 0.0) -> float:
        """Hybrid priority: max of TD-based and reward-based priority."""
        td_prio = (abs(td_error) + self.epsilon) ** self.alpha
        rw_prio = abs(reward) * self.reward_weight
        return max(td_prio, rw_prio, 1e-6)

    def add(self, state: Tuple[np.ndarray, np.ndarray], action: int,
            reward: float, next_state: Tuple[np.ndarray, np.ndarray],
            done: bool, td_error: Optional[float] = None):
        """Store transition with initial priority.

        New transitions start at _init_priority (1.0) instead of
        max_priority — this prevents fresh entries from dominating
        sampling before their true TD-error is known.

        state, next_state are (board, features) tuples.
        """
        if td_error is not None:
            priority = self._compute_priority(td_error, reward)
        else:
            priority = self._init_priority

        trans = StoredTransition(
            board=state[0], features=state[1],
            action=action, reward=reward,
            next_board=next_state[0], next_features=next_state[1],
            done=done,
        )
        self.tree.add(priority, trans)

    def sample(self, batch_size: int, step: int = 0
               ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample a batch with importance sampling weights.

        Returns (batch_dict, indices, weights).
        """
        batch = {
            "board": [], "features": [], "actions": [], "rewards": [],
            "next_board": [], "next_features": [], "dones": [],
        }
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)

        total = self.tree.total()
        if total == 0:
            return batch, indices, weights

        segment = total / batch_size
        beta = self._beta(step)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)

            if data is None:
                continue

            prob = priority / total
            weight = (len(self.tree) * prob) ** (-beta)
            weights[i] = weight

            batch["board"].append(data.board)
            batch["features"].append(data.features)
            batch["actions"].append(data.action)
            batch["rewards"].append(data.reward)
            batch["next_board"].append(data.next_board)
            batch["next_features"].append(data.next_features)
            batch["dones"].append(data.done)
            indices[i] = idx

        # Normalise weights.
        max_w = np.max(weights) if np.max(weights) > 0 else 1.0
        weights = weights / max_w

        # Stack into arrays.
        batch["board"] = np.stack(batch["board"])
        batch["features"] = np.stack(batch["features"])
        batch["actions"] = np.array(batch["actions"], dtype=np.int64)
        batch["rewards"] = np.array(batch["rewards"], dtype=np.float32)
        batch["next_board"] = np.stack(batch["next_board"])
        batch["next_features"] = np.stack(batch["next_features"])
        batch["dones"] = np.array(batch["dones"], dtype=bool)

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray,
                          rewards: Optional[np.ndarray] = None):
        """Update priorities using hybrid TD+reward priority."""
        if rewards is None:
            rewards = np.zeros_like(td_errors)
        for idx, td_error, reward in zip(indices, td_errors, rewards):
            priority = self._compute_priority(float(td_error), float(reward))
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)

    def _beta(self, step: int) -> float:
        """Linearly anneal beta from start to end."""
        frac = min(step / self.beta_frames, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def state_dict(self) -> dict:
        """Serialise buffer state for checkpointing."""
        return {
            "tree_data": self.tree.data,
            "tree_tree": self.tree.tree,
            "tree_write_pos": self.tree.write_pos,
            "tree_size": self.tree.size,
            "max_priority": self.max_priority,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_frames": self.beta_frames,
            "epsilon": self.epsilon,
            "reward_weight": self.reward_weight,
        }

    def load_state_dict(self, state: dict):
        """Restore buffer state from checkpoint."""
        self.capacity = state["capacity"]
        self.alpha = state.get("alpha", 0.8)
        self.beta_start = state.get("beta_start", 0.4)
        self.beta_end = state.get("beta_end", 1.0)
        self.beta_frames = state.get("beta_frames", 10_000_000)
        self.epsilon = state.get("epsilon", 1e-6)
        self.reward_weight = state.get("reward_weight", 0.5)
        self.max_priority = state["max_priority"]
        self._init_priority = 1.0
        self.tree = SumTree(self.capacity)
        # Restore tree data — pad/truncate to match capacity.
        n = min(len(state["tree_data"]), self.capacity)
        self.tree.data[:n] = state["tree_data"][:n]
        n_tree = min(len(state["tree_tree"]), len(self.tree.tree))
        self.tree.tree[:n_tree] = state["tree_tree"][:n_tree]
        self.tree.write_pos = min(state["tree_write_pos"], self.capacity - 1)
        self.tree.size = min(state["tree_size"], self.capacity)

    def __len__(self) -> int:
        return len(self.tree)


class UniformReplayBuffer:
    """Simple uniform-sampling replay buffer (for ablation experiments)."""

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer: List[StoredTransition] = []
        self.write_pos = 0

    def add(self, state: Tuple[np.ndarray, np.ndarray], action: int,
            reward: float, next_state: Tuple[np.ndarray, np.ndarray], done: bool):
        trans = StoredTransition(
            board=state[0], features=state[1],
            action=action, reward=reward,
            next_board=next_state[0], next_features=next_state[1],
            done=done,
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(trans)
        else:
            self.buffer[self.write_pos] = trans
        self.write_pos = (self.write_pos + 1) % self.capacity

    def sample(self, batch_size: int, step: int = 0):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = {
            "board": [], "features": [], "actions": [], "rewards": [],
            "next_board": [], "next_features": [], "dones": [],
        }
        for i in indices:
            d = self.buffer[i]
            batch["board"].append(d.board)
            batch["features"].append(d.features)
            batch["actions"].append(d.action)
            batch["rewards"].append(d.reward)
            batch["next_board"].append(d.next_board)
            batch["next_features"].append(d.next_features)
            batch["dones"].append(d.done)
        for k in ("board", "features", "next_board", "next_features"):
            batch[k] = np.stack(batch[k])
        batch["actions"] = np.array(batch["actions"], dtype=np.int64)
        batch["rewards"] = np.array(batch["rewards"], dtype=np.float32)
        batch["dones"] = np.array(batch["dones"], dtype=bool)
        return batch, indices, np.ones(batch_size, dtype=np.float32)  # uniform weights

    def update_priorities(self, indices, td_errors):
        pass  # no-op for uniform buffer

    def __len__(self):
        return len(self.buffer)
