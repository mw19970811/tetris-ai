#!/usr/bin/env python3
"""Training entry point for Tetris RL agent.

Usage:
    python scripts/train.py                          # DQN with defaults
    python scripts/train.py --probe                   # Hardware probe + train
    python scripts/train.py --algo ppo                # PPO
    python scripts/train.py --steps 10000000 --wandb  # 10M steps with logging
    python scripts/train.py --help                    # Full options
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.trainer import main

if __name__ == "__main__":
    # Run hardware probe before training if requested.
    if "--probe" in sys.argv:
        sys.argv.remove("--probe")
        from trainer.hardware_probe import probe
        from trainer.resource_planner import plan, apply_plan
        from scripts.probe import print_report
        info = probe()
        rp = plan(info)
        print_report(info, rp)

    main()
