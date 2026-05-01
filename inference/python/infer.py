"""Python inference interface for trained models.

Supports both PyTorch and ONNX Runtime backends.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path


class InferenceEngine:
    """Load a trained model and perform inference for action selection."""

    def __init__(self, model_path: str, backend: str = "auto",
                 num_actions: int = 112, feature_dim: int = 53,
                 device: str = "cpu"):
        """
        Args:
            model_path: Path to .pt checkpoint or .onnx file.
            backend: "pytorch", "onnx", or "auto" (detect from extension).
        """
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.device = device

        if backend == "auto":
            backend = "onnx" if model_path.endswith(".onnx") else "pytorch"

        if backend == "onnx":
            self._load_onnx(model_path)
        else:
            self._load_pytorch(model_path)

    def _load_pytorch(self, path: str):
        """Load PyTorch checkpoint."""
        from agent.model import DuelingDQN

        checkpoint = torch.load(path, map_location=self.device)
        self.model = DuelingDQN(
            num_actions=self.num_actions, feature_dim=self.feature_dim, use_noisy=False
        ).to(self.device)

        # Handle different checkpoint formats.
        if "agent_state" in checkpoint:
            state_dict = checkpoint["agent_state"]["online_net"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "online_net" in checkpoint:
            state_dict = checkpoint["online_net"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._backend = "pytorch"

    def _load_onnx(self, path: str):
        """Load ONNX model."""
        import onnxruntime as ort
        self.session = ort.InferenceSession(path)
        self._backend = "onnx"

    @torch.no_grad()
    def infer(self, board: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Run inference. Returns q-values of shape (num_actions,)."""
        if self._backend == "pytorch":
            board_t = torch.as_tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
            feat_t = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(board_t, feat_t).squeeze(0).cpu().numpy()
        else:
            board_in = board.astype("float32")[np.newaxis, :, :, :] if board.ndim == 3 else board.astype("float32")
            feat_in = features.astype("float32")[np.newaxis, :] if features.ndim == 1 else features.astype("float32")
            out = self.session.run(None, {"board": board_in, "features": feat_in})
            q = out[0].squeeze()
        return q

    def select_action(self, board: np.ndarray, features: np.ndarray,
                      legal_actions: List, deterministic: bool = True
                      ) -> Tuple[int, int, int, int]:
        """Select best action from legal actions.

        Returns (rotation, column, hold, action_idx).
        """
        from agent.action_mask import create_action_mask, mask_logits, encode_action, decode_action

        mask = create_action_mask(legal_actions, self.num_actions)
        if not mask.any():
            return 0, 0, False, 0

        q = self.infer(board, features)
        q_tensor = torch.as_tensor(q)
        mask_tensor = mask

        masked_q = mask_logits(q_tensor, mask_tensor, -1e9)
        action_idx = int(masked_q.argmax().item())
        rotation, col, hold = decode_action(action_idx)
        return rotation, col, hold, action_idx
