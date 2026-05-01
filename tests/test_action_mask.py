"""Tests for action masking utilities."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest
from agent.action_mask import (
    encode_action, decode_action, create_action_mask,
    mask_logits, masked_softmax, sample_masked_action,
    NUM_ACTIONS, NUM_COL_BUCKETS, COL_OFFSET,
)


class TestActionEncoding:
    def test_encode_decode_roundtrip(self):
        for rot in range(4):
            for col in range(-2, 12):
                for hold in [False, True]:
                    idx = encode_action(rot, col, hold)
                    r, c, h = decode_action(idx)
                    assert r == rot, f"Rotation mismatch: {rot} vs {r}"
                    # Column clamped to valid range.
                    expected_col = max(-2, min(11, col))
                    assert c == expected_col, f"Column mismatch: {col} vs {c}"
                    assert h == hold

    def test_encode_valid_range(self):
        for rot in range(4):
            for col in range(10):
                idx = encode_action(rot, col, False)
                assert 0 <= idx < NUM_ACTIONS

    def test_decode_valid_range(self):
        for idx in range(NUM_ACTIONS):
            rot, col, hold = decode_action(idx)
            assert 0 <= rot <= 3
            assert -2 <= col <= 11


class TestActionMask:
    def test_create_mask(self):
        legal = [(0, 3, False), (0, 4, False), (1, 3, False)]
        mask = create_action_mask(legal)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.sum() == len(legal)

    def test_mask_logits(self):
        logits = torch.randn(NUM_ACTIONS)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        mask[0] = True
        mask[10] = True

        masked = mask_logits(logits, mask, -1e9)
        assert masked[0] == logits[0]
        assert masked[10] == logits[10]
        assert masked[1] == -1e9

    def test_masked_softmax(self):
        logits = torch.ones(NUM_ACTIONS)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        mask[0] = True
        mask[1] = True

        probs = masked_softmax(logits, mask)
        assert probs.shape == (NUM_ACTIONS,)
        assert abs(probs[0] - 0.5) < 0.01
        assert abs(probs[1] - 0.5) < 0.01
        assert probs[2] == 0.0

    def test_sample_masked_action(self):
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        mask[5] = True  # only one legal action
        probs = torch.rand(NUM_ACTIONS)
        probs[5] = 1.0

        # With only one legal action, should always sample it.
        for _ in range(10):
            action = sample_masked_action(probs, mask)
            assert action.item() == 5

    def test_no_legal_actions(self):
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        probs = torch.rand(NUM_ACTIONS)
        action = sample_masked_action(probs, mask)
        assert action.item() == 0  # fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
