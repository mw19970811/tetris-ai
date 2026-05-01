"""Logging utilities for training metrics.

Supports both TensorBoard and Weights & Biases backends.
"""

import time
import json
import os
from collections import defaultdict
from typing import Dict, Optional, Any


class Logger:
    """Unified logger for training metrics."""

    def __init__(self, log_dir: str = "logs", use_wandb: bool = False,
                 wandb_project: str = "tetris-ai", wandb_entity: str = "",
                 config: Optional[dict] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

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
                print("[WARN] wandb not installed, falling back to TensorBoard only.")
                self.use_wandb = False

        # Local metrics storage.
        self.metrics_history: Dict[str, list] = defaultdict(list)
        self.start_time = time.time()

    def log(self, metrics: Dict[str, Any], step: int):
        """Log a metrics dict at a given training step."""
        metrics["step"] = step
        metrics["wall_time"] = time.time() - self.start_time
        self.metrics_history["step"].append(step)

        for k, v in metrics.items():
            self.metrics_history[k].append(v)
            if self.use_wandb and self.wandb_run:
                import wandb
                wandb.log({k: v}, step=step)

    def log_scalar(self, tag: str, value: float, step: int):
        self.log({tag: value}, step)

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
        """Log training step metrics."""
        self.log({f"train/{k}": v for k, v in kwargs.items()}, step)

    def save_metrics(self, filename: str = "metrics.json"):
        """Save all logged metrics to a JSON file."""
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(dict(self.metrics_history), f, indent=2)

    def close(self):
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()

    def __del__(self):
        self.close()
