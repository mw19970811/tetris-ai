"""Tests for the reward calculator module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from env.reward_calculator import RewardCalculator, RewardConfig


class TestRewardCalculator:
    def test_line_clear_reward(self):
        calc = RewardCalculator(RewardConfig())
        r = calc.compute(lines_cleared=4, level=1, drop_distance=0, holes=0)
        assert r == pytest.approx(800.01)  # 800 + 0.01 survival

    def test_line_clear_with_level(self):
        calc = RewardCalculator(RewardConfig())
        r = calc.compute(lines_cleared=4, level=5, drop_distance=0, holes=0)
        assert r == pytest.approx(4000.01)  # 800 * 5 + 0.01

    def test_single_line(self):
        calc = RewardCalculator(RewardConfig())
        r = calc.compute(lines_cleared=1, level=1, drop_distance=0, holes=0)
        assert r == pytest.approx(100.01)

    def test_death_penalty(self):
        calc = RewardCalculator(RewardConfig())
        r = calc.compute(lines_cleared=0, level=1, terminated=True)
        assert r == -100.0

    def test_survival_bonus(self):
        calc = RewardCalculator(RewardConfig())
        r = calc.compute(lines_cleared=0, level=1)
        assert r == 0.01

    def test_hole_penalty(self):
        calc = RewardCalculator(RewardConfig(w_holes=1.5))
        r = calc.compute(lines_cleared=0, level=1, holes=10)
        assert r == pytest.approx(0.01 - 1.5 * 10)

    def test_height_penalty(self):
        calc = RewardCalculator(RewardConfig(w_height=0.3))
        heights = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        r = calc.compute(lines_cleared=0, level=1, board_heights=heights, holes=0)
        assert r == pytest.approx(0.01 - 0.3 * 50)  # 50 total height

    def test_detailed_breakdown(self):
        calc = RewardCalculator(RewardConfig())
        components = calc.compute_detailed(
            lines_cleared=2, level=2, drop_distance=5,
            heights_sum=50, holes=5, bumpiness=10, max_well=3,
        )
        assert "total" in components
        assert abs(components["line_clear"] - 600.0) < 0.01   # 2 × 300 × 2
        assert abs(components["hard_drop"] - 10.0) < 0.01      # 5 × 2
        assert abs(components["holes_penalty"] - (-1.5 * 5)) < 0.01
        assert abs(components["height_penalty"] - (-0.3 * 50)) < 0.01

    def test_detailed_breakdown_all_components_present(self):
        calc = RewardCalculator(RewardConfig())
        comp = calc.compute_detailed(0, 1, 0, 0, 0, 0, 0)
        expected_keys = {
            "hard_drop", "line_clear", "height_penalty", "holes_penalty",
            "bumpiness_penalty", "well_penalty", "survival", "death", "total"
        }
        assert set(comp.keys()) == expected_keys

    def test_custom_weights(self):
        cfg = RewardConfig(w_height=1.0, w_holes=2.0, w_bumpiness=0.5,
                           w_well=1.0, w_survival=0.1, w_death=-200.0)
        calc = RewardCalculator(cfg)
        heights = np.array([10, 5, 15])
        r = calc.compute(lines_cleared=0, level=1, board_heights=heights, holes=5,
                         bumpiness=3, max_well=2, terminated=True)
        # death=-200, no survival
        expected = (-200.0 - 1.0 * 30 - 2.0 * 5 - 0.5 * 3 - 1.0 * 2)
        assert r == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
