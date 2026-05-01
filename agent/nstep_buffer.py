"""N-step return buffer for accelerating credit assignment in Tetris RL.

Accumulates trajectories and computes n-step bootstrapped returns.
For Tetris, n=5 works well => Tetris setup spans ~5-8 placements.
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, List


class NStepBuffer:
    """Stores the last n transitions and computes n-step return when complete.

    R_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k}  +  gamma^n * max_a Q_target(s_{t+n}, a)
    """

    def __init__(self, n: int = 5, gamma: float = 0.99):
        self.n = n
        self.gamma = gamma
        # Each state/next_state is a tuple (board, features).
        self._states: deque = deque(maxlen=n)
        self._actions: deque = deque(maxlen=n)
        self._rewards: deque = deque(maxlen=n)
        self._next_states: deque = deque(maxlen=n)
        self._dones: deque = deque(maxlen=n)

    def add(self, state: Tuple[np.ndarray, np.ndarray], action: int,
            reward: float, next_state: Tuple[np.ndarray, np.ndarray], done: bool
            ) -> Optional[Tuple[Tuple, int, float, Tuple, bool]]:
        """Add transition. Returns None until buffer has n entries,
        then returns oldest transition with n-step discounted reward sum.
        The Q bootstrap term gamma^n * max_a Q(s_{t+n}, a) is applied in the trainer.
        """
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)

        if len(self._states) < self.n:
            return None

        # Compute n-step return (reward sum only, no placeholder).
        n_return = 0.0
        for k in range(self.n):
            n_return += (self.gamma ** k) * self._rewards[k]
            if self._dones[k]:
                break

        # If episode ended within n steps, no Q bootstrap will be added.
        # Flag is conveyed via the next-state's done flag in the trainer.
        result = (
            self._states[0],
            self._actions[0],
            n_return,                 # Sum of discounted rewards only.
            self._next_states[-1],    # s_{t+n}
            self._dones[-1],
        )

        return result

    def flush(self) -> List[Tuple]:
        """Drain remaining transitions (with truncated n-step returns)."""
        results = []
        while len(self._states) >= 1:
            n_return = 0.0
            for k in range(len(self._states)):
                n_return += (self.gamma ** k) * self._rewards[k]
                if self._dones[k]:
                    break
            results.append((
                self._states[0], self._actions[0], n_return,
                self._next_states[-1], self._dones[-1],
            ))
            self._states.popleft()
            self._actions.popleft()
            self._rewards.popleft()
            self._next_states.popleft()
            self._dones.popleft()
        return results

    def reset(self):
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._next_states.clear()
        self._dones.clear()

    def __len__(self):
        return len(self._states)
