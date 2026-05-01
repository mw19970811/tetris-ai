#!/usr/bin/env python3
"""Batch evaluation script: evaluate a trained model on multiple episodes.

Usage:
    python scripts/eval.py --model checkpoints/step_010000000.pt --episodes 100
    python scripts/eval.py --model tetris_ai.onnx --backend onnx --episodes 50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from collections import defaultdict


def evaluate(model_path: str, num_episodes: int = 100, backend: str = "auto"):
    """Evaluate a model over multiple episodes and print stats."""
    from env.tetris_env import TetrisEnv, EnvConfig, Action
    from inference.python.infer import InferenceEngine

    engine = InferenceEngine(model_path, backend=backend)
    stats = defaultdict(list)

    for ep in range(num_episodes):
        env = TetrisEnv(EnvConfig())
        obs = env.reset()
        done = False
        ep_lines = 0
        ep_tetris = 0

        while not done:
            board, features = obs[0], obs[1]
            legal = env.get_legal_actions()
            if not legal:
                break

            rot, col, hold, _ = engine.select_action(board, features, legal)
            obs, reward, terminated, truncated, info = env.step(Action(rot, col, hold))
            done = terminated or truncated

            ep_lines = info.get("lines", 0)

        stats["score"].append(info.get("score", 0))
        stats["lines"].append(info.get("lines", 0))
        stats["level"].append(info.get("level", 1))
        stats["steps"].append(info.get("steps", 0))

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{num_episodes}  "
                  f"Avg Score: {np.mean(stats['score'][-20:]):,.0f}  "
                  f"Best: {np.max(stats['score']):,.0f}")

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Avg Score:     {np.mean(stats['score']):>15,.1f}")
    print(f"  Max Score:     {np.max(stats['score']):>15,.1f}")
    print(f"  Min Score:     {np.min(stats['score']):>15,.1f}")
    print(f"  Std Score:     {np.std(stats['score']):>15,.1f}")
    print(f"  Avg Lines:     {np.mean(stats['lines']):>15,.1f}")
    print(f"  Avg Level:     {np.mean(stats['level']):>15,.1f}")
    print(f"  Avg Steps:     {np.mean(stats['steps']):>15,.1f}")
    print(f"{'='*60}")

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tetris AI model")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint or ONNX path")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx"])
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.backend)


if __name__ == "__main__":
    main()
