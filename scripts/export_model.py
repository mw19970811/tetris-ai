#!/usr/bin/env python3
"""Export model to ONNX format.

Usage:
    python scripts/export_model.py checkpoints/step_010000000.pt -o tetris_ai.onnx
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.python.export import main

if __name__ == "__main__":
    main()
