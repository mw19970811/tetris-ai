#!/usr/bin/env python3
"""Interactive play: watch the AI play Tetris, or play against it.

Usage:
    python scripts/play.py --model checkpoints/step_005000000.pt   # AI plays
    python scripts/play.py --model tetris_ai.onnx --backend onnx   # ONNX inference
    python scripts/play.py --human                                  # Human plays (keyboard)
"""

import sys
import os
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def play_ai(model_path: str, backend: str = "auto", delay_ms: int = 50):
    """Watch AI play Tetris."""
    from env.tetris_env import TetrisEnv, EnvConfig, Action
    from inference.python.infer import InferenceEngine

    env = TetrisEnv(EnvConfig())
    engine = InferenceEngine(model_path, backend=backend)

    obs = env.reset()
    done = False
    step = 0

    while not done:
        board, features = obs[0], obs[1]
        legal = env.get_legal_actions()

        if not legal:
            break

        rot, col, hold, _ = engine.select_action(board, features, legal)
        action = Action(rot, col, hold)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        # Render.
        print("\033[H\033[J")  # clear screen
        print(env.render("ansi"))
        print(f"Step: {step}  Score: {info['score']:,}  Level: {info['level']}  Lines: {info['lines']}")

        time.sleep(delay_ms / 1000.0)

    print(f"\nGame Over! Final Score: {info['score']:,}  Lines: {info['lines']}  Steps: {step}")


def play_human():
    """Human player mode using keyboard."""
    from env.tetris_env import TetrisEnv, EnvConfig, Action

    env = TetrisEnv(EnvConfig())
    obs = env.reset()

    import keyboard
    print("\nControls: ← → arrows=move, ↑=rotate CW, Z=rotate CCW, ↓=soft drop, Space=hard drop, C=hold, Q=quit\n")
    print(env.render("ansi"))

    done = False
    while not done:
        action = None
        key = keyboard.read_key()

        if key == 'left':
            action = Action(0, 3, False)  # Simplified: just move
        elif key == 'right':
            action = Action(0, 4, False)
        elif key == 'up':
            action = Action(1, 3, False)
        elif key == 'z':
            action = Action(3, 3, False)
        elif key == 'space':
            # Hard drop — for placement-based, we handle via legal actions.
            legal = env.get_legal_actions()
            if legal:
                action = legal[len(legal) // 2]  # pick middle action
        elif key == 'c':
            legal = [a for a in env.get_legal_actions() if a.hold]
            if legal:
                action = legal[0]
        elif key == 'q':
            break

        if action:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print("\033[H\033[J")
            print(env.render("ansi"))
            print(f"Score: {info['score']:,}  Level: {info['level']}  Lines: {info['lines']}")

    print(f"\nGame Over! Final Score: {info.get('score', 0):,}")


def main():
    parser = argparse.ArgumentParser(description="Play Tetris (AI or human)")
    parser.add_argument("--model", type=str, default="", help="Model checkpoint/ONNX path")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx"])
    parser.add_argument("--human", action="store_true", help="Human player mode")
    parser.add_argument("--delay", type=int, default=50, help="Delay between AI moves (ms)")
    args = parser.parse_args()

    if args.human:
        play_human()
    elif args.model:
        play_ai(args.model, args.backend, args.delay)
    else:
        print("Usage: python play.py --model <checkpoint.pt>   OR   python play.py --human")
        print("Example: python play.py --model checkpoints/step_005000000.pt --delay 100")


if __name__ == "__main__":
    main()
