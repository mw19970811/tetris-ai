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
    """PER buffer with blended TD+reward priority and reward-aware init.

    Every new transition enters with ``max_priority`` (PER standard)
    so it is guaranteed visibility.  High-reward transitions (line
    clears) get an additional reward floor that can exceed max_priority.

    Blended priority formula:
        priority = (1 - reward_blend) * |td|^alpha  +  reward_blend * |reward| * reward_weight
    """

    def __init__(self, capacity: int = 1_000_000, alpha: float = 0.3,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_frames: int = 10_000_000, epsilon: float = 1e-6,
                 reward_weight: float = 0.5, reward_blend: float = 0.3,
                 reward_clip: float = 10.0):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.reward_weight = reward_weight
        self.reward_blend = reward_blend
        self.reward_clip = reward_clip
        self.max_priority = 1.0
        self._priority_ema = 1.0  # Monitoring: running average of updated priorities
        self._update_count = 0

    @staticmethod
    def _clamp_reward(r: float, clip: float) -> float:
        """Clamp reward to [-clip, clip] for priority computation.

        Outlier rewards (e.g. +192 survival bonus from an empty board)
        are capped so they don't dominate sampling.  The true reward
        is still stored for TD-error computation.
        """
        if clip <= 0:
            return r
        return max(-clip, min(clip, r))

    def _compute_priority(self, td_error: float, reward: float = 0.0) -> float:
        """Blended priority with reward clipped to prevent outlier domination."""
        r = self._clamp_reward(reward, self.reward_clip)
        td_prio = (abs(td_error) + self.epsilon) ** self.alpha
        rw_prio = abs(r) * self.reward_weight
        b = self.reward_blend
        return max((1.0 - b) * td_prio + b * rw_prio, 1e-6)

    def _init_priority_for(self, reward: float) -> float:
        """Initial priority for a new transition entering the SumTree.

        Uses ``max_priority`` (PER standard) so every new entry is
        guaranteed visibility.  Reward is clipped to prevent outliers
        from getting an artificially high floor.
        """
        r = self._clamp_reward(reward, self.reward_clip)
        rw_prio = abs(r) * self.reward_weight
        reward_floor = self.reward_blend * rw_prio
        return max(self.max_priority, reward_floor, 1.0)

    def add(self, state: Tuple[np.ndarray, np.ndarray], action: int,
            reward: float, next_state: Tuple[np.ndarray, np.ndarray],
            done: bool, td_error: Optional[float] = None):
        """Store transition directly in the SumTree with high initial priority."""
        trans = StoredTransition(
            board=state[0], features=state[1],
            action=action, reward=reward,
            next_board=next_state[0], next_features=next_state[1],
            done=done,
        )
        prio = self._init_priority_for(reward)
        self.tree.add(prio, trans)

    def sample(self, batch_size: int, step: int = 0
               ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample a batch with importance sampling weights."""
        batch = {
            "board": [], "features": [], "actions": [], "rewards": [],
            "next_board": [], "next_features": [], "dones": [],
        }
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)

        total = self.tree.total()
        if total == 0 or len(self.tree) == 0:
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

        # Normalise IS weights.
        max_w = np.max(weights) if np.max(weights) > 0 else 1.0
        weights = weights / max_w

        # Stack.
        batch["board"] = np.stack(batch["board"])
        batch["features"] = np.stack(batch["features"])
        batch["actions"] = np.array(batch["actions"], dtype=np.int64)
        batch["rewards"] = np.array(batch["rewards"], dtype=np.float32)
        batch["next_board"] = np.stack(batch["next_board"])
        batch["next_features"] = np.stack(batch["next_features"])
        batch["dones"] = np.array(batch["dones"], dtype=bool)

        return batch, indices, weights

    def _refresh_max_priority(self):
        """Recompute max_priority from the SumTree leaves.

        Called periodically so ``max_priority`` reflects the *current*
        buffer, not a stale spike from early training.
        """
        if self.tree.size == 0:
            self.max_priority = 1.0
            return
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.tree.size
        leaves = self.tree.tree[leaf_start:leaf_end]
        # Only active leaves (data not None) have priority > 0.
        active = leaves[leaves > 0]
        self.max_priority = float(np.max(active)) if len(active) > 0 else 1.0

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray,
                          rewards: Optional[np.ndarray] = None):
        """Update priorities using blended TD+reward priority.

        ``max_priority`` tracks the batch ceiling every step, and is
        periodically recalibrated from the full SumTree so it stays
        representative of the current buffer.
        """
        if rewards is None:
            rewards = np.zeros_like(td_errors)
        batch_mean = 0.0
        batch_max = 0.0
        for idx, td_error, reward in zip(indices, td_errors, rewards):
            priority = self._compute_priority(float(td_error), float(reward))
            self.tree.update(int(idx), priority)
            batch_mean += priority
            batch_max = max(batch_max, priority)
        batch_mean /= max(len(indices), 1)

        # Track batch max (prevents stale spikes).
        self.max_priority = max(self.max_priority, batch_max)

        # Every 100 steps, recalibrate from full tree.
        self._update_count += 1
        if self._update_count % 100 == 0:
            self._refresh_max_priority()

        self._priority_ema = 0.99 * self._priority_ema + 0.01 * batch_mean

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
            "reward_clip": self.reward_clip,
            "priority_ema": self._priority_ema,
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
        self.reward_blend = state.get("reward_blend", 0.3)
        self.reward_clip = state.get("reward_clip", 10.0)
        self.max_priority = state.get("max_priority", 1.0)
        self._priority_ema = state.get("priority_ema", self.max_priority)
        # Backward-compat: drain old-format newcomers into tree.
        newcomers = state.get("newcomers", [])
        self.tree = SumTree(self.capacity)
        n = min(len(state["tree_data"]), self.capacity)
        self.tree.data[:n] = state["tree_data"][:n]
        n_tree = min(len(state["tree_tree"]), len(self.tree.tree))
        self.tree.tree[:n_tree] = state["tree_tree"][:n_tree]
        self.tree.write_pos = min(state["tree_write_pos"], self.capacity - 1)
        self.tree.size = min(state["tree_size"], self.capacity)
        for t in newcomers:
            prio = self._init_priority_for(t.reward)
            self.tree.add(prio, t)

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
