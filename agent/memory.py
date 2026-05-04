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

    def add(self, priority: float, data: Any) -> int:
        """Add data with given priority, overwriting oldest if full.

        Returns the tree index of the new leaf.
        """
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

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
    """PER buffer with newcomer guarantee + blended TD+reward priority.

    Two-stage design ensures every transition is sampled at least once:
      1. Newcomer FIFO (capacity ~2000) — new transitions wait here.
         Up to 50 % of each training batch is drawn from newcomers.
      2. PER SumTree — transitions graduate here after being sampled once,
         with their true priority computed from TD-error + reward.

    Blended priority formula:
        priority = (1 - reward_blend) * |td|^alpha  +  reward_blend * |reward| * reward_weight
    """

    def __init__(self, capacity: int = 1_000_000, alpha: float = 0.3,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_frames: int = 10_000_000, epsilon: float = 1e-6,
                 reward_weight: float = 0.5, reward_blend: float = 0.9):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.reward_weight = reward_weight
        self.reward_blend = reward_blend
        self.max_priority = 1.0
        self._priority_ema = 1.0  # Running average of updated priorities
        self._init_priority = 1.0  # Tracks EMA

        # Newcomer FIFO — guarantees every transition gets sampled once.
        self._newcomers: list = []          # list of StoredTransition
        self._newcomer_capacity = 2000      # ~31 rounds at 64 envs/step
        self._newcomer_ratio = 0.5          # fraction of batch drawn from newcomers

    def _compute_priority(self, td_error: float, reward: float = 0.0) -> float:
        """Blended priority: weighted mix of TD-based and reward-based."""
        td_prio = (abs(td_error) + self.epsilon) ** self.alpha
        rw_prio = abs(reward) * self.reward_weight
        b = self.reward_blend
        return max((1.0 - b) * td_prio + b * rw_prio, 1e-6)

    def add(self, state: Tuple[np.ndarray, np.ndarray], action: int,
            reward: float, next_state: Tuple[np.ndarray, np.ndarray],
            done: bool, td_error: Optional[float] = None):
        """Store transition.  New entries go to the newcomer FIFO first;
        overflow drains to the PER tree with EMA init_priority.
        """
        trans = StoredTransition(
            board=state[0], features=state[1],
            action=action, reward=reward,
            next_board=next_state[0], next_features=next_state[1],
            done=done,
        )
        self._newcomers.append(trans)

        # Drain overflow to PER tree.
        while len(self._newcomers) > self._newcomer_capacity:
            old = self._newcomers.pop(0)
            self.tree.add(self._init_priority, old)

    def sample(self, batch_size: int, step: int = 0
               ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample a batch mixing newcomers (guaranteed exposure) and PER tree.

        Up to ``newcomer_ratio`` of the batch comes from the newcomer FIFO
        so every transition is trained on at least once.  Sampled newcomers
        graduate to the PER tree so their priority can be updated.
        """
        n_new = min(int(batch_size * self._newcomer_ratio), len(self._newcomers))
        # Don't ask the tree for more than it has.
        n_tree = min(batch_size - n_new, len(self.tree))
        # If tree is too small, fill remaining from newcomers.
        n_new = min(batch_size - n_tree, len(self._newcomers))

        batch = {
            "board": [], "features": [], "actions": [], "rewards": [],
            "next_board": [], "next_features": [], "dones": [],
        }
        # Tree indices are negative for newcomers (distinguished in update_priorities).
        indices = np.full(batch_size, -1, dtype=np.int32)
        weights = np.ones(batch_size, dtype=np.float32)

        # === Newcomers: uniform random draw from FIFO ===
        grad_indices = []  # tree indices of graduated newcomers
        if n_new > 0:
            picks = random.sample(range(len(self._newcomers)), n_new)
            picks.sort(reverse=True)
            batch_positions = []  # which batch slot each newcomer fills
            for p in picks:
                t = self._newcomers.pop(p)
                batch["board"].append(t.board)
                batch["features"].append(t.features)
                batch["actions"].append(t.action)
                batch["rewards"].append(t.reward)
                batch["next_board"].append(t.next_board)
                batch["next_features"].append(t.next_features)
                batch["dones"].append(t.done)
                # Graduate to PER tree; record tree index for update_priorities.
                tree_idx = self.tree.add(self._init_priority, t)
                grad_indices.append(tree_idx)

            # Fill indices for newcomer slots.
            for i, ti in enumerate(grad_indices):
                indices[i] = ti

        # === PER tree ===
        tree_total = self.tree.total()
        if n_tree > 0 and tree_total > 0 and len(self.tree) > 0:
            segment = tree_total / n_tree
            beta = self._beta(step)
            for i in range(n_tree):
                s = random.uniform(segment * i, segment * (i + 1))
                tree_idx, priority, data = self.tree.get(s)
                if data is None:
                    continue
                prob = priority / tree_total
                w = (len(self.tree) * prob) ** (-beta)
                weights[n_new + i] = w
                indices[n_new + i] = tree_idx
                batch["board"].append(data.board)
                batch["features"].append(data.features)
                batch["actions"].append(data.action)
                batch["rewards"].append(data.reward)
                batch["next_board"].append(data.next_board)
                batch["next_features"].append(data.next_features)
                batch["dones"].append(data.done)

        # Normalise IS weights.
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
        """Update priorities using blended TD+reward priority.

        Also tracks EMA of priorities so new transitions get a fair
        initial priority (not stuck at 1.0 in a pool of 88.75).
        """
        if rewards is None:
            rewards = np.zeros_like(td_errors)
        batch_mean = 0.0
        for idx, td_error, reward in zip(indices, td_errors, rewards):
            priority = self._compute_priority(float(td_error), float(reward))
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)
            batch_mean += priority
        batch_mean /= max(len(indices), 1)

        # EMA tracking — smooth factor 0.01 balances recency vs stability.
        self._priority_ema = 0.99 * self._priority_ema + 0.01 * batch_mean
        self._init_priority = max(self._priority_ema, 1.0)

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
            "reward_blend": self.reward_blend,
            "priority_ema": self._priority_ema,
            "newcomers": self._newcomers,
        }

    def load_state_dict(self, state: dict):
        """Restore buffer state from checkpoint."""
        self.capacity = state["capacity"]
        self.alpha = state.get("alpha", 0.4)
        self.beta_start = state.get("beta_start", 0.4)
        self.beta_end = state.get("beta_end", 1.0)
        self.beta_frames = state.get("beta_frames", 10_000_000)
        self.epsilon = state.get("epsilon", 1e-6)
        self.reward_weight = state.get("reward_weight", 0.5)
        self.reward_blend = state.get("reward_blend", 0.9)
        self.max_priority = state["max_priority"]
        self._priority_ema = state.get("priority_ema", self.max_priority)
        self._init_priority = max(self._priority_ema, 1.0)
        self._newcomers = state.get("newcomers", [])
        self.tree = SumTree(self.capacity)
        # Restore tree data — pad/truncate to match capacity.
        n = min(len(state["tree_data"]), self.capacity)
        self.tree.data[:n] = state["tree_data"][:n]
        n_tree = min(len(state["tree_tree"]), len(self.tree.tree))
        self.tree.tree[:n_tree] = state["tree_tree"][:n_tree]
        self.tree.write_pos = min(state["tree_write_pos"], self.capacity - 1)
        self.tree.size = min(state["tree_size"], self.capacity)

    def __len__(self) -> int:
        return len(self.tree) + len(self._newcomers)


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
