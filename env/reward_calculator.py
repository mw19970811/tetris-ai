"""Reward calculator with multi-layer reward shaping for Tetris RL."""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardConfig:
    w_height: float = 0.3
    w_holes: float = 1.5
    w_bumpiness: float = 0.2
    w_well: float = 0.5
    w_survival: float = 0.01
    w_death: float = -100.0

    # Line clear base scores (× level)
    line_scores: tuple = (0, 100, 300, 500, 800)

    hard_drop_score: float = 2.0   # per cell
    soft_drop_score: float = 1.0   # per cell


class RewardCalculator:
    """Computes shaped reward for each step.

    r = r_clear * level + r_death + w_h*(-height) + w_o*(-holes) + w_b*(-bump) + w_w*(-well) + survival
    """

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.cfg = config

    def compute(self, lines_cleared: int, level: int, drop_distance: int = 0,
                board_heights: np.ndarray = None, holes: int = 0,
                bumpiness: float = 0.0, max_well: int = 0,
                terminated: bool = False) -> float:

        reward = 0.0

        # Hard drop bonus.
        reward += drop_distance * self.cfg.hard_drop_score

        # Line clear reward.
        if 0 <= lines_cleared <= 4:
            reward += self.cfg.line_scores[lines_cleared] * level

        # Board quality penalties.
        if board_heights is not None:
            reward -= self.cfg.w_height * float(np.sum(board_heights))
        reward -= self.cfg.w_holes * float(holes)
        reward -= self.cfg.w_bumpiness * float(bumpiness)
        reward -= self.cfg.w_well * float(max_well)

        # Survival bonus.
        if not terminated:
            reward += self.cfg.w_survival
        else:
            reward += self.cfg.w_death

        return reward

    def compute_detailed(self, lines_cleared: int, level: int,
                         drop_distance: int = 0,
                         heights_sum: float = 0.0, holes: int = 0,
                         bumpiness: float = 0.0, max_well: int = 0,
                         terminated: bool = False) -> Dict[str, float]:
        """Compute reward with per-component breakdown for logging."""
        components = {}
        components["hard_drop"] = drop_distance * self.cfg.hard_drop_score
        if 0 <= lines_cleared <= 4:
            components["line_clear"] = self.cfg.line_scores[lines_cleared] * level
        else:
            components["line_clear"] = 0.0
        components["height_penalty"] = -self.cfg.w_height * heights_sum
        components["holes_penalty"] = -self.cfg.w_holes * float(holes)
        components["bumpiness_penalty"] = -self.cfg.w_bumpiness * float(bumpiness)
        components["well_penalty"] = -self.cfg.w_well * float(max_well)
        components["survival"] = self.cfg.w_survival if not terminated else 0.0
        components["death"] = self.cfg.w_death if terminated else 0.0
        components["total"] = sum(components.values())
        return components
