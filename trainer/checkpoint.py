"""Checkpoint management for training state persistence."""

import os
import glob
import torch
from typing import Dict, Optional
from datetime import datetime


class CheckpointManager:
    """Manages saving/loading of training checkpoints.

    Cleanup policy: retains the *keep_best* checkpoints with highest
    evaluation scores, plus the *keep_latest* most-recent checkpoints
    (the two sets may overlap).  Older checkpoints are deleted
    automatically.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 keep_best: int = 5, keep_latest: int = 1):
        self.checkpoint_dir = checkpoint_dir
        self.keep_best = keep_best
        self.keep_latest = keep_latest
        self._checkpoint_scores: Dict[str, float] = {}
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, step: int, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             extra_state: Optional[Dict] = None) -> str:
        """Save a checkpoint. Returns the checkpoint path."""
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        if extra_state:
            checkpoint["extra"] = extra_state

        path = os.path.join(self.checkpoint_dir, f"step_{step:09d}.pt")
        torch.save(checkpoint, path)
        self._cleanup()
        return path

    def save_full(self, step: int, agent, replay_buffer=None) -> str:
        """Save full training state including agent, optimizer, and replay buffer."""
        agent_state = agent.state_dict()

        checkpoint = {
            "step": step,
            "agent_state": agent_state,
            "agent_type": type(agent).__name__,
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(self.checkpoint_dir, f"step_{step:09d}.pt")
        torch.save(checkpoint, path)

        # Save replay buffer for DQN (can be large, separate file).
        if replay_buffer is not None:
            try:
                buffer_path = os.path.join(self.checkpoint_dir, f"step_{step:09d}_buffer.pt")
                buf_state = replay_buffer.state_dict() if hasattr(replay_buffer, 'state_dict') else {
                    "tree_data": replay_buffer.tree.data,
                    "tree_tree": replay_buffer.tree.tree,
                    "tree_write_pos": replay_buffer.tree.write_pos,
                    "tree_size": replay_buffer.tree.size,
                    "max_priority": replay_buffer.max_priority,
                }
                torch.save(buf_state, buffer_path)
            except Exception as e:
                print(f"[Checkpoint] Warning: failed to save replay buffer: {e}")

        self._cleanup()
        return path

    def load(self, path: str, model: torch.nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """Load checkpoint. Returns the step number."""
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("step", 0)

    def load_full(self, path: str, agent, replay_buffer=None) -> int:
        """Load full training state. Returns the step number."""
        checkpoint = torch.load(path, map_location="cpu")

        # Support multiple checkpoint formats.
        if "agent_state" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state"])
        elif "model_state_dict" in checkpoint:
            net = getattr(agent, 'online_net', None) or getattr(agent, 'network', None)
            if net is not None:
                net.load_state_dict(checkpoint["model_state_dict"])
            if hasattr(agent, 'target_net'):
                agent.target_net.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Try loading directly as state dict.
            agent.load_state_dict(checkpoint)

        step = checkpoint.get("step", 0)

        # Restore replay buffer.
        buffer_path = path.replace(".pt", "_buffer.pt")
        if replay_buffer is not None and os.path.exists(buffer_path):
            try:
                buf = torch.load(buffer_path, map_location="cpu", weights_only=False)
                if hasattr(replay_buffer, 'load_state_dict'):
                    replay_buffer.load_state_dict(buf)
                else:
                    replay_buffer.tree.data = buf["tree_data"]
                    replay_buffer.tree.tree = buf["tree_tree"]
                    replay_buffer.tree.write_pos = buf["tree_write_pos"]
                    replay_buffer.tree.size = buf["tree_size"]
                    replay_buffer.max_priority = buf["max_priority"]
            except Exception as e:
                print(f"[Checkpoint] Warning: failed to restore replay buffer: {e}")

        return step

    def load_latest(self, agent, replay_buffer=None) -> int:
        """Load the latest checkpoint. Returns step number (0 if none found)."""
        latest = self.find_latest()
        if latest is None:
            return 0
        print(f"[Checkpoint] Resuming from: {latest}")
        return self.load_full(latest, agent, replay_buffer)

    def save_ppo(self, step: int, agent) -> str:
        """Save PPO training state (model + optimizer)."""
        checkpoint = {
            "step": step,
            "agent_state": agent.state_dict(),
            "agent_type": "PPO",
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(self.checkpoint_dir, f"step_{step:09d}.pt")
        torch.save(checkpoint, path)
        self._cleanup()
        return path

    def find_latest(self) -> Optional[str]:
        """Find the latest checkpoint by step number."""
        pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        # Exclude buffer files.
        files = [f for f in glob.glob(pattern) if "_buffer" not in f]
        if not files:
            return None
        files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[1]))
        return files[-1]

    def record_score(self, path: str, score: float):
        """Record the evaluation score for a checkpoint, used for best-N retention."""
        self._checkpoint_scores[path] = score

    def _cleanup(self):
        """Keep top ``keep_best`` by score + most recent ``keep_latest``."""
        pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        files = [f for f in glob.glob(pattern) if "_buffer" not in f]
        if len(files) <= self.keep_best + self.keep_latest:
            return
        files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[1]))

        # Top keep_best by recorded evaluation score.
        scored = [(f, self._checkpoint_scores.get(f, 0.0)) for f in files]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = {f for f, _ in scored[:self.keep_best]}

        # Most recent keep_latest.
        latest_n = set(files[-self.keep_latest:])

        protected = top_n | latest_n

        for f in files:
            if f not in protected:
                os.remove(f)
                buf = f.replace(".pt", "_buffer.pt")
                if os.path.exists(buf):
                    os.remove(buf)
