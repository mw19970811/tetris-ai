"""Logging utilities for training metrics.

Supports: file logging (TeeLogger), TensorBoard, Weights & Biases, and
local JSON metrics history.
"""

import time
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, Optional, Any


class TeeLogger:
    """Duplicates stdout to a log file (like Unix tee).

    Replaces ``sys.stdout`` on construction; call ``close()`` to restore.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"train_{timestamp}.log")
        self._file = open(self.path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


class Logger:
    """Unified logger for training metrics — local, TensorBoard, and WandB."""

    def __init__(self, log_dir: str = "logs", use_wandb: bool = False,
                 wandb_project: str = "tetris-ai", wandb_entity: str = "",
                 use_tensorboard: bool = True,
                 config: Optional[dict] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard.
        self.use_tb = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard")
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                print(f"[Logger] TensorBoard enabled — run: tensorboard --logdir={tb_dir}")
            except ImportError:
                print("[WARN] tensorboard not installed (pip install tensorboard).")
                self.use_tb = False

        # WandB.
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project, entity=wandb_entity or None,
                    config=config, dir=log_dir,
                )
            except ImportError:
                print("[WARN] wandb not installed, falling back to file logging only.")
                self.use_wandb = False

        # Local metrics storage.
        self.metrics_history: Dict[str, list] = defaultdict(list)
        self.start_time = time.time()

    # ------------------------------------------------------------------ #
    def log(self, metrics: Dict[str, Any], step: int):
        """Log a metrics dict at a given training step."""
        metrics["step"] = step
        metrics["wall_time"] = time.time() - self.start_time
        self.metrics_history["step"].append(step)

        for k, v in metrics.items():
            if k == "step":
                continue
            self.metrics_history[k].append(v)

            # TensorBoard.
            if self.use_tb and self.tb_writer and isinstance(v, (int, float)):
                self.tb_writer.add_scalar(k, v, step)

            # WandB.
            if self.use_wandb and self.wandb_run:
                import wandb
                wandb.log({k: v}, step=step)

    def log_scalar(self, tag: str, value: float, step: int):
        self.log({tag: value}, step)

    # ------------------------------------------------------------------ #
    #  High-level logging helpers
    # ------------------------------------------------------------------ #
    def log_train_step(self, step: int, avg_reward: float, avg_lines: float,
                       fps: float, buffer_size: int, elapsed: float,
                       dead_count: int = 0, dead_rate: float = 0.0,
                       avg_steps: float = 0.0, **extra):
        """Log per-log-interval training scalars (the main curves)."""
        self.log({
            "train/avg_reward": avg_reward,
            "train/avg_lines": avg_lines,
            "train/avg_steps": avg_steps,
            "train/fps": fps,
            "train/buffer_size": buffer_size,
            "train/elapsed_h": elapsed / 3600.0,
            "train/stale_dead_count": dead_count,
            "train/stale_dead_rate": dead_rate,
            **{f"train/{k}": v for k, v in extra.items()},
        }, step)

    def log_eval(self, step: int, avg_score: float, max_score: float,
                 min_score: float, std_score: float, avg_lines: float,
                 avg_level: float, avg_steps: float, tetris_rate: float = 0.0,
                 **kwargs):
        """Log evaluation results."""
        self.log({
            "eval/avg_score": avg_score,
            "eval/max_score": max_score,
            "eval/min_score": min_score,
            "eval/std_score": std_score,
            "eval/avg_lines": avg_lines,
            "eval/avg_level": avg_level,
            "eval/avg_steps": avg_steps,
            "eval/tetris_rate": tetris_rate,
        }, step)

    def log_train(self, step: int, **kwargs):
        """Log training step metrics (DQN-specific: loss, q_mean, etc.)."""
        self.log({f"train/{k}": v for k, v in kwargs.items()}, step)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save_metrics(self, filename: str = "metrics.json"):
        """Save all logged metrics to a JSON file."""
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(dict(self.metrics_history), f, indent=2)

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()

    def __del__(self):
        self.close()
