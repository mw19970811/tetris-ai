"""Action masking utilities for placement-based action space.

Encodes actions as: idx = rotation * 13 + (column + 2) + (hold ? 52 : 0)
Max theoretical actions: 4 * 13 * 2 = 104.
In practice we use a compact encoding with col in [-2, 11] mapped to [0, 13].
"""

import torch
import numpy as np
from typing import List


# Action encoding constants.
MAX_COL = 10
COL_OFFSET = 2      # col = -2 maps to idx 0
NUM_COL_BUCKETS = MAX_COL + 2 * COL_OFFSET  # 14 buckets for columns -2..11
NUM_ROTATIONS = 4
NUM_ACTIONS = NUM_ROTATIONS * NUM_COL_BUCKETS * 2  # 4 * 14 * 2 = 112 max


def encode_action(rotation: int, column: int, hold: bool) -> int:
    """Encode (rotation, column, hold) to flat index."""
    col_idx = column + COL_OFFSET
    col_idx = max(0, min(NUM_COL_BUCKETS - 1, col_idx))
    idx = rotation * NUM_COL_BUCKETS + col_idx
    if hold:
        idx += NUM_ROTATIONS * NUM_COL_BUCKETS
    return idx


def decode_action(idx: int) -> tuple:
    """Decode flat index to (rotation, column, hold)."""
    hold = idx >= (NUM_ROTATIONS * NUM_COL_BUCKETS)
    if hold:
        idx -= NUM_ROTATIONS * NUM_COL_BUCKETS
    rotation = idx // NUM_COL_BUCKETS
    col = (idx % NUM_COL_BUCKETS) - COL_OFFSET
    return rotation, int(col), hold


def create_action_mask(legal_actions: List, num_actions: int = NUM_ACTIONS,
                       device: torch.device = None) -> torch.Tensor:
    """Create boolean action mask from list of (rotation, column, hold) tuples."""
    mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    for action in legal_actions:
        if hasattr(action, 'rotation'):
            idx = encode_action(action.rotation, action.column, action.hold)
        else:
            idx = encode_action(*action)
        if 0 <= idx < num_actions:
            mask[idx] = True
    return mask


def mask_logits(logits: torch.Tensor, mask: torch.Tensor,
                fill_value: float = -1e9) -> torch.Tensor:
    """Apply action mask to logits/q-values. Sets illegal actions to fill_value."""
    return torch.where(mask, logits, torch.full_like(logits, fill_value))


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over only legal actions."""
    masked = mask_logits(logits, mask, -1e9)
    return torch.softmax(masked, dim=-1)


def sample_masked_action(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sample an action index from masked probability distribution."""
    if not mask.any():
        return torch.tensor(0)
    # Set illegal action probs to 0 and renormalize.
    masked_probs = probs * mask.float()
    masked_probs = masked_probs / masked_probs.sum()
    return torch.multinomial(masked_probs, 1).squeeze(-1)
