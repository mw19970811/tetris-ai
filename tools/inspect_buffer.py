#!/usr/bin/env python3
"""Interactive replay-buffer inspector for Tetris AI training.

Load a ``*_buffer.pt`` checkpoint and browse transitions with keyboard
navigation.  Renders before/after boards, decodes actions and features.

Usage::

    python tools/inspect_buffer.py checkpoints/step_000010000_buffer.pt
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path so torch.load can unpickle agent.* classes.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_CELL = {"I": "█", "O": "█", "T": "█", "S": "█", "Z": "█", "J": "█", "L": "█"}

COLS = 10
TOTAL_ROWS = 22
HIDDEN_ROWS = 2
NUM_ACTIONS = 112
NUM_ROTATIONS = 4
NUM_COL_BUCKETS = 14
COL_OFFSET = 2

# ANSI escapes
CLEAR = "\033[2J\033[H"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[37m"
GRAY = "\033[90m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"


# ------------------------------------------------------------------ #
#  Cross-platform keyboard input
# ------------------------------------------------------------------ #
def _get_key_unix() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Escape sequence (arrow keys etc.)
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "UP"
            elif seq == "[B":
                return "DOWN"
            elif seq == "[C":
                return "RIGHT"
            elif seq == "[D":
                return "LEFT"
            elif seq == "[H":
                return "HOME"
            elif seq == "[F":
                return "END"
            elif seq == "[5":
                return "PGUP"
            elif seq == "[6":
                return "PGDN"
            return "ESC"
        elif ch == "\x7f":
            return "BACKSPACE"
        elif ch == "\r" or ch == "\n":
            return "ENTER"
        elif ord(ch) == 3:
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _get_key_windows() -> str:
    import msvcrt

    ch = msvcrt.getch()
    if ch == b"\xe0" or ch == b"\x00":
        # Extended key
        ch2 = msvcrt.getch()
        mapping = {
            b"H": "UP",
            b"P": "DOWN",
            b"M": "RIGHT",
            b"K": "LEFT",
            b"G": "HOME",
            b"O": "END",
            b"I": "PGUP",
            b"Q": "PGDN",
        }
        return mapping.get(ch2, f"\\x{ch2[0]:02x}")
    if ch == b"\x1b":
        return "ESC"
    if ch == b"\x08":
        return "BACKSPACE"
    if ch == b"\r":
        return "ENTER"
    if ord(ch) == 3:
        raise KeyboardInterrupt
    try:
        return ch.decode("utf-8")
    except UnicodeDecodeError:
        return "?"


def get_key() -> str:
    if os.name == "nt":
        return _get_key_windows()
    else:
        return _get_key_unix()


# ------------------------------------------------------------------ #
#  Action codec
# ------------------------------------------------------------------ #
def decode_action(idx: int) -> Tuple[int, int, bool]:
    hold = idx >= (NUM_ROTATIONS * NUM_COL_BUCKETS)
    if hold:
        idx -= NUM_ROTATIONS * NUM_COL_BUCKETS
    rotation = idx // NUM_COL_BUCKETS
    col = (idx % NUM_COL_BUCKETS) - COL_OFFSET
    return rotation, int(col), hold


# ------------------------------------------------------------------ #
#  Feature decoding
# ------------------------------------------------------------------ #
def decode_features(features: np.ndarray):
    feats = np.asarray(features, dtype=np.float32)
    result = {
        "height_sum": float(feats[0]),
        "lines_placeholder": float(feats[1]),
        "holes": float(feats[2]),
        "bumpiness": float(feats[3]),
        "max_well": float(feats[4]),
        "height_change_placeholder": float(feats[5]),
    }
    piece_idx = int(np.argmax(feats[6:13]))
    result["current_piece"] = PIECE_NAMES[piece_idx] if piece_idx < 7 else "?"
    result["rotation"] = int(np.argmax(feats[13:17]))
    hold_idx = int(np.argmax(feats[17:25]))
    result["hold"] = PIECE_NAMES[hold_idx] if hold_idx < 7 else "none"
    result["next"] = []
    for i in range(4):
        oh = feats[25 + i * 7 : 25 + (i + 1) * 7]
        nidx = int(np.argmax(oh))
        result["next"].append(PIECE_NAMES[nidx] if nidx < 7 else "?")
    return result


# ------------------------------------------------------------------ #
#  Board rendering
# ------------------------------------------------------------------ #
def render_board(board_2d: np.ndarray, highlight: Optional[List[Tuple[int, int]]] = None,
                 cleared_rows: Optional[set] = None) -> str:
    """Render a (22, 10) board as ANSI-coloured ASCII.

    Parameters
    ----------
    board_2d : shape (22, 10)
    highlight : cells to colour green (newly placed piece).
    cleared_rows : rows that were cleared (yellow).
    """
    highlight = highlight or set()
    cleared_rows = cleared_rows or set()
    lines: List[str] = []
    lines.append(f"      {DIM}col → 0 1 2 3 4 5 6 7 8 9{RESET}")
    lines.append(f"      ┌─────────────────────┐")

    for r in range(TOTAL_ROWS):
        line = " " * 6
        if r == 0:
            line += f"{BOLD}row{RESET} {DIM}{r:>2}{RESET} │"
        elif r == 1:
            line += f"     {DIM}{r:>2}{RESET} │"
        elif r == TOTAL_ROWS - 1:
            line += f"     {BOLD}{r:>2}{RESET} │"
        else:
            line += f"     {DIM}{r:>2}{RESET} │"

        for c in range(COLS):
            if (r, c) in highlight:
                line += f" {GREEN}■{RESET}"
            elif r in cleared_rows and board_2d[r, c]:
                line += f" {YELLOW}■{RESET}"
            elif board_2d[r, c]:
                line += f" {WHITE}■{RESET}"
            else:
                line += f" {DIM}·{RESET}"
        line += " │"
        if r == 0:
            line += f"  ← {DIM}hidden rows (0-1){RESET}"
        elif r == 2:
            line += f"  ← {DIM}visible rows start{RESET}"
        lines.append(line)

    lines.append(f"      └─────────────────────┘")
    return "\n".join(lines)


def compute_diff_cells(before: np.ndarray, after: np.ndarray) -> List[Tuple[int, int]]:
    """Return cells that changed from 0→1 (newly placed)."""
    cells = []
    for r in range(TOTAL_ROWS):
        for c in range(COLS):
            if not before[r, c] and after[r, c]:
                cells.append((r, c))
    return cells


def compute_cleared_rows(before: np.ndarray, after: np.ndarray) -> set:
    """Return rows that went from full to empty (cleared lines)."""
    cleared = set()
    for r in range(TOTAL_ROWS):
        if before[r].all() and not after[r].any():
            cleared.add(r)
    return cleared


# ------------------------------------------------------------------ #
#  Buffer loading
# ------------------------------------------------------------------ #
def load_buffer(path: str) -> dict:
    import torch

    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    file_size = os.path.getsize(path)
    if file_size < 1024:
        print(f"Error: file is too small ({file_size} bytes) — likely truncated or corrupt.")
        sys.exit(1)

    try:
        buf = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error: failed to load buffer file: {e}")
        print(f"  File: {path}  ({file_size / 1024**2:.1f} MB)")
        print(f"  The file may be truncated (training interrupted mid-save).")
        print(f"  Try an earlier checkpoint, e.g.:")
        import re
        m = re.search(r'step_(\d+)_buffer', os.path.basename(path))
        if m:
            step = int(m.group(1))
            prev = max(step - 20000, 0)
            alt = path.replace(f'step_{step:09d}', f'step_{prev:09d}')
            print(f"    python tools/inspect_buffer.py {alt}")
        sys.exit(1)

    if not isinstance(buf, dict):
        print(f"Error: unexpected buffer format (got {type(buf).__name__}, expected dict).")
        sys.exit(1)
    return buf


def extract_transitions_with_priorities(buf: dict) -> list:
    """Extract valid transitions paired with their SumTree priorities.

    Returns list of (transition, priority) where priority is the raw
    SumTree leaf value (TD-error ** alpha).
    """
    data = buf.get("tree_data", []) or buf.get("data", [])
    tree = buf.get("tree_tree", None)
    cap = buf.get("capacity", len(data))

    priorities = _extract_leaf_priorities(tree, cap, len(data))

    results = []
    for i, t in enumerate(data):
        if t is not None:
            prio = priorities[i] if i < len(priorities) else 1.0
            results.append((t, float(prio)))
    return results


def _extract_leaf_priorities(tree: np.ndarray, capacity: int, data_len: int) -> np.ndarray:
    """Extract per-leaf priorities from SumTree array.

    SumTree layout: leaves at indices [capacity-1, 2*capacity-2).
    Returns array of length data_len with priority for each position.
    """
    if tree is None or capacity <= 0:
        return np.ones(data_len, dtype=np.float64)

    leaf_start = capacity - 1
    leaf_end = min(leaf_start + data_len, len(tree))
    n_leaves = leaf_end - leaf_start
    if n_leaves <= 0:
        return np.ones(data_len, dtype=np.float64)

    result = np.ones(data_len, dtype=np.float64)
    result[:n_leaves] = np.asarray(tree[leaf_start:leaf_end], dtype=np.float64)
    return result


def transition_to_dict(t) -> dict:
    """Convert a StoredTransition to plain dict with decoded fields."""
    board = np.asarray(t.board, dtype=bool).squeeze(0)  # (22, 10)
    next_board = np.asarray(t.next_board, dtype=bool).squeeze(0)
    features = np.asarray(t.features, dtype=np.float32)
    next_features = np.asarray(t.next_features, dtype=np.float32)
    rot, col, hold = decode_action(int(t.action))

    before_info = decode_features(features)
    after_info = decode_features(next_features)
    diff = compute_diff_cells(board, next_board)
    cleared = compute_cleared_rows(board, next_board)

    return {
        "board": board,
        "next_board": next_board,
        "features": features,
        "next_features": next_features,
        "action": int(t.action),
        "rotation": rot,
        "column": col,
        "hold": hold,
        "reward": float(t.reward),
        "done": bool(t.done),
        "before_info": before_info,
        "after_info": after_info,
        "diff_cells": diff,
        "cleared_rows": cleared,
    }


# ------------------------------------------------------------------ #
#  Interactive viewer
# ------------------------------------------------------------------ #
class BufferViewer:
    def __init__(self, entries: list, filepath: str, buf_meta: dict):
        # entries: list of (StoredTransition, priority)
        self.entries = entries
        self.filepath = filepath
        self.capacity = buf_meta.get("capacity", 0)
        self.total = len(entries)

        # Per-entry metrics arrays.
        self._priorities = np.array([p for _, p in entries], dtype=np.float64)
        self._rewards = np.array([float(t.reward) for t, _ in entries], dtype=np.float32)
        self._probs = self._compute_probs(buf_meta.get("alpha", 0.6))
        self._weights = self._compute_is_weights()

        # Currently displayed index (into self._visible list).
        self._cursor = 0
        self._filter_mode = "all"
        self._status_msg = ""
        self._rebuild_filter()

    def _compute_probs(self, alpha: float) -> np.ndarray:
        """Compute sampling probability for each transition."""
        p = self._priorities.copy()
        p = np.maximum(p, 1e-12)
        if alpha != 1.0:
            p = p ** (1.0 / alpha) if alpha > 0 else p  # undo alpha exponent
            # raw priority = |td|^alpha → |td| = priority^(1/alpha)
            # sampling prob using the same alpha
            p = np.maximum(p, 1e-12) ** alpha
        total = p.sum()
        return p / total if total > 0 else np.ones_like(p) / len(p)

    def _transition(self, idx: int):
        """Return the StoredTransition at global index."""
        return self.entries[idx][0]

    def _priority(self, idx: int) -> float:
        return self._priorities[idx]

    def _prob(self, idx: int) -> float:
        return self._probs[idx]

    def _weight(self, idx: int) -> float:
        return self._weights[idx]

    def _compute_is_weights(self) -> np.ndarray:
        """Compute importance-sampling correction factors (beta=1.0).

        Returns the raw IS multiplier:  w_i = 1 / (N * P(i)).
        w_i > 1 → transition is under-sampled (loss up-weighted).
        w_i < 1 → transition is over-sampled (loss down-weighted).
        """
        n = len(self._probs)
        if n == 0:
            return np.ones(1, dtype=np.float64)
        probs = np.maximum(self._probs, 1e-12)
        return 1.0 / (n * probs)

    def _rebuild_filter(self):
        """Build list of visible indices based on current filter."""
        if self._filter_mode == "all":
            self._visible = list(range(self.total))
        elif self._filter_mode == "done":
            self._visible = [i for i, (t, _) in enumerate(self.entries) if t.done]
        elif self._filter_mode == "live":
            self._visible = [i for i, (t, _) in enumerate(self.entries) if not t.done]
        elif self._filter_mode == "reward+":
            self._visible = [i for i in range(self.total) if self._rewards[i] > 0]
        elif self._filter_mode == "reward-":
            self._visible = [i for i in range(self.total) if self._rewards[i] < 0]
        elif self._filter_mode == "top-reward":
            self._visible = sorted(range(self.total),
                                   key=lambda i: self._rewards[i], reverse=True)
        elif self._filter_mode == "top-prio":
            self._visible = sorted(range(self.total),
                                   key=lambda i: self._priorities[i], reverse=True)
        else:
            self._visible = list(range(self.total))

        if not self._visible:
            self._visible = [0]
        self._cursor = max(0, min(self._cursor, len(self._visible) - 1))

    @property
    def current_idx(self) -> int:
        return self._visible[self._cursor]

    @property
    def visible_count(self) -> int:
        return len(self._visible)

    def nav_next(self, step: int = 1):
        self._cursor = min(self._cursor + step, len(self._visible) - 1)

    def nav_prev(self, step: int = 1):
        self._cursor = max(self._cursor - step, 0)

    def nav_goto(self, idx: int):
        """Go to specific global index (find nearest in visible list)."""
        idx = max(0, min(idx, self.total - 1))
        # Find closest visible
        if idx in self._visible:
            self._cursor = self._visible.index(idx)
        else:
            # Find nearest
            best = 0
            best_dist = 999999
            for vi, global_idx in enumerate(self._visible):
                d = abs(global_idx - idx)
                if d < best_dist:
                    best_dist = d
                    best = vi
            self._cursor = best

    def nav_home(self):
        self._cursor = 0

    def nav_end(self):
        self._cursor = len(self._visible) - 1

    def cycle_filter(self):
        modes = ["all", "done", "live", "reward+", "reward-", "top-reward", "top-prio"]
        try:
            cur = modes.index(self._filter_mode)
        except ValueError:
            cur = 0
        self._filter_mode = modes[(cur + 1) % len(modes)]
        self._rebuild_filter()

    # ------------------------------------------------------------------ #
    #  Episode tracing
    # ------------------------------------------------------------------ #
    def trace_forward(self) -> bool:
        """Find a transition whose *board* matches current *next_board*."""
        current = self._transition(self.current_idx)
        target = np.asarray(current.next_board, dtype=bool).squeeze(0)

        for i in range(self.current_idx + 1, self.total):
            t = self._transition(i)
            if np.array_equal(np.asarray(t.board, dtype=bool).squeeze(0), target):
                self._ensure_all_filter()
                self._cursor = self._visible.index(i)
                return True
        for i in range(0, self.current_idx):
            t = self._transition(i)
            if np.array_equal(np.asarray(t.board, dtype=bool).squeeze(0), target):
                self._ensure_all_filter()
                self._cursor = self._visible.index(i)
                return True
        return False

    def trace_backward(self) -> bool:
        """Find a transition whose *next_board* matches current *board*."""
        current = self._transition(self.current_idx)
        target = np.asarray(current.board, dtype=bool).squeeze(0)

        for i in range(self.current_idx - 1, -1, -1):
            t = self._transition(i)
            if np.array_equal(np.asarray(t.next_board, dtype=bool).squeeze(0), target):
                self._ensure_all_filter()
                self._cursor = self._visible.index(i)
                return True
        for i in range(self.total - 1, self.current_idx, -1):
            t = self._transition(i)
            if np.array_equal(np.asarray(t.next_board, dtype=bool).squeeze(0), target):
                self._ensure_all_filter()
                self._cursor = self._visible.index(i)
                return True
        return False

    def _ensure_all_filter(self):
        """Switch filter to 'all' if not already."""
        if self._filter_mode != "all":
            self._filter_mode = "all"
            self._rebuild_filter()

    # ------------------------------------------------------------------ #
    #  Jump-to-max
    # ------------------------------------------------------------------ #
    def jump_max_reward(self):
        """Jump to transition with highest reward in visible set."""
        if not self._visible:
            return
        best = max(self._visible, key=lambda i: self._rewards[i])
        self._cursor = self._visible.index(best)

    def jump_max_priority(self):
        """Jump to transition with highest priority in visible set."""
        if not self._visible:
            return
        best = max(self._visible, key=lambda i: self._priorities[i])
        self._cursor = self._visible.index(best)

    def jump_max_prob(self):
        """Jump to transition with highest sampling probability."""
        if not self._visible:
            return
        best = max(self._visible, key=lambda i: self._probs[i])
        self._cursor = self._visible.index(best)


# ------------------------------------------------------------------ #
#  Rendering
# ------------------------------------------------------------------ #
def render_viewer(viewer: BufferViewer) -> str:
    idx = viewer.current_idx
    d = transition_to_dict(viewer._transition(idx))
    priority = viewer._priority(idx)
    prob = viewer._prob(idx)
    reward = viewer._rewards[idx]

    lines: List[str] = []
    # Header
    lines.append(CLEAR)
    lines.append(f"{BOLD}{'═' * 67}{RESET}")
    lines.append(
        f"  {BOLD}Buffer:{RESET} {os.path.basename(viewer.filepath)}"
    )
    lines.append(
        f"  Samples: {viewer.total:,} / {viewer.capacity:,}"
        f"    max reward: {viewer._rewards.max():+.1f}"
        f"    max priority: {viewer._priorities.max():.2f}"
    )
    # Reward distribution summary
    pos_count = int((viewer._rewards > 0).sum())
    neg_count = int((viewer._rewards < 0).sum())
    zero_count = int((viewer._rewards == 0).sum())
    lines.append(
        f"  Rewards: +{pos_count:,}  -{neg_count:,}  ={zero_count:,}"
        f"    range [{viewer._rewards.min():+.1f}, {viewer._rewards.max():+.1f}]"
    )
    # IS weight colour coding.
    weight = viewer._weight(idx)
    if weight > 1.0:
        w_color, w_label = GREEN, f"x{weight:.2f}"
    elif weight > 0.1:
        w_color, w_label = YELLOW, f"x{weight:.3f}"
    else:
        w_color, w_label = DIM, f"x{weight:.4f}"

    # Rank info for sorted modes.
    rank_str = ""
    if viewer._filter_mode == "top-reward":
        rank_str = f"    {GREEN}rank #{viewer._cursor + 1} by reward{RESET}"
    elif viewer._filter_mode == "top-prio":
        rank_str = f"    {GREEN}rank #{viewer._cursor + 1} by priority{RESET}"

    lines.append(
        f"  {BOLD}Transition #{idx:,} / {viewer.total - 1:,}"
        f"    (visible: {viewer._cursor + 1} / {viewer.visible_count}){rank_str}"
    )
    # Compact reward + priority summary line.
    r_color = GREEN if reward > 0 else (RED if reward < 0 else "")
    pct = 100.0 * int((viewer._priorities >= priority).sum()) / viewer.total
    lines.append(
        f"  {BOLD}Reward:{RESET} {r_color}{reward:+.1f}{RESET}"
        f"    {BOLD}Priority:{RESET} {priority:.4f}"
        f"  (top {pct:.1f}%)"
        f"    {BOLD}IS:{RESET} {w_color}{w_label}{RESET}"
    )
    lines.append(f"{BOLD}{'═' * 67}{RESET}")

    # BEFORE board
    lines.append(f"\n  {CYAN}{BOLD}◀── BEFORE (state){RESET}  {DIM}board[{idx}].board{RESET}")
    lines.append(render_board(d["board"], highlight=set()))
    lines.append("")

    # Action + metadata line
    rot, col, hold_flag = d["rotation"], d["column"], d["hold"]
    action_str = f"rot={rot}, col={col:>3}, hold={hold_flag}"
    reward_color = GREEN if reward > 0 else (RED if reward < 0 else "")
    lines.append(
        f"  {YELLOW}{BOLD}Action:{RESET} {action_str:<28}"
        f"  |  {BOLD}Reward:{RESET} {reward_color}{reward:+.1f}{RESET}"
        f"  |  {BOLD}Done:{RESET} {d['done']}"
    )
    # Piece info
    bi = d["before_info"]
    lines.append(
        f"  {BOLD}Current:{RESET} {bi['current_piece']} (rot {bi['rotation']})"
        f"    {BOLD}Hold:{RESET} [{bi['hold']}]"
        f"    {BOLD}Next:{RESET} {' '.join(bi['next'])}"
    )
    lines.append(f"  {BOLD}Features:{RESET} height={bi['height_sum']:.0f}  "
                 f"holes={bi['holes']:.0f}  bump={bi['bumpiness']:.0f}  "
                 f"well={bi['max_well']:.0f}")

    # AFTER board
    lines.append(f"\n  {CYAN}{BOLD}──▶ AFTER (next_state){RESET}  {DIM}board[{idx}].next_board{RESET}")
    if d["cleared_rows"]:
        lines.append(f"  {YELLOW}Lines cleared: {len(d['cleared_rows'])}{RESET}")
    highlight_set = set(d["diff_cells"]) if d["diff_cells"] else set()
    lines.append(render_board(d["next_board"], highlight=highlight_set,
                              cleared_rows=d["cleared_rows"]))
    lines.append("")

    # After stats
    ai = d["after_info"]
    lines.append(f"  {BOLD}After features:{RESET} height={ai['height_sum']:.0f}  "
                 f"holes={ai['holes']:.0f}  bump={ai['bumpiness']:.0f}  "
                 f"well={ai['max_well']:.0f}")
    lines.append(f"  {BOLD}Next piece after:{RESET} {' '.join(ai['next'])}")

    # Footer
    lines.append("")
    lines.append(f"{BOLD}{'═' * 65}{RESET}")
    lines.append(
        f"  {DIM}[←→/↑↓]{RESET} prev/next 1   "
        f"  {DIM}[PgUp/PgDn]{RESET} ±10   "
        f"  {DIM}[Home/End]{RESET} first/last"
    )
    fmode = viewer._filter_mode
    if fmode in ("top-reward", "top-prio"):
        fm_display = f"{GREEN}{fmode}{RESET}"
    else:
        fm_display = fmode
    lines.append(
        f"  {DIM}[g]{RESET} goto idx   "
        f"  {DIM}[f]{RESET} filter: {fm_display:<14}"
        f"  {DIM}[r]{RESET} max reward   "
        f"  {DIM}[p]{RESET} max prio"
    )
    lines.append(
        f"  {DIM}[t]{RESET} trace fwd   "
        f"  {DIM}[T]{RESET} trace bwd   "
        f"  {DIM}[q]{RESET} quit"
    )
    if viewer._status_msg:
        lines.append(f"  {YELLOW}{viewer._status_msg}{RESET}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Main loop
# ------------------------------------------------------------------ #
def interactive_loop(viewer: BufferViewer):
    """Run the interactive viewer loop."""
    print(render_viewer(viewer))

    while True:
        try:
            key = get_key()
        except KeyboardInterrupt:
            break

        if key in ("q", "Q", "ESC"):
            break
        elif key in ("RIGHT", "DOWN"):
            viewer._status_msg = ""
            viewer.nav_next(1)
        elif key in ("LEFT", "UP"):
            viewer._status_msg = ""
            viewer.nav_prev(1)
        elif key in ("PGDN", "]"):
            viewer._status_msg = ""
            viewer.nav_next(10)
        elif key in ("PGUP", "["):
            viewer._status_msg = ""
            viewer.nav_prev(10)
        elif key in ("HOME", "^"):
            viewer._status_msg = ""
            viewer.nav_home()
        elif key in ("END", "$"):
            viewer._status_msg = ""
            viewer.nav_end()
        elif key in ("f", "F"):
            viewer._status_msg = ""
            viewer.cycle_filter()
        elif key in ("g", "G"):
            viewer._status_msg = ""
            _goto_input(viewer)
        elif key in ("r", "R"):
            viewer._status_msg = ""
            viewer.jump_max_reward()
            viewer._status_msg = f"max reward ({viewer._rewards[viewer.current_idx]:+.1f}) @ #{viewer.current_idx:,}"
        elif key in ("p", "P"):
            viewer._status_msg = ""
            viewer.jump_max_priority()
            viewer._status_msg = f"max priority ({viewer._priority(viewer.current_idx):.4f}) @ #{viewer.current_idx:,}"
        elif key == "t":
            if viewer.trace_forward():
                viewer._status_msg = f"traced forward → #{viewer.current_idx:,}"
            else:
                viewer._status_msg = "no matching next transition found"
        elif key == "T":
            if viewer.trace_backward():
                viewer._status_msg = f"traced backward → #{viewer.current_idx:,}"
            else:
                viewer._status_msg = (
                    "no matching prev transition — episode start, "
                    "overwritten, or N-step gap crosses env boundary"
                )
        elif key in ("h", "H"):
            # Quick help overlay
            pass

        print(render_viewer(viewer))

    # Clean exit
    print(CLEAR)
    print(f"Exited.  Buffer: {os.path.basename(viewer.filepath)}  "
          f"Samples: {viewer.total:,}")


def _goto_input(viewer: BufferViewer):
    """Ask user for an index to jump to."""
    # Move cursor to bottom line
    sys.stdout.write(f"\033[{24 + 15}H")  # approximate
    sys.stdout.write(f"\033[K")  # clear line
    sys.stdout.write(f"  Goto index (0-{viewer.total - 1}): ")
    sys.stdout.flush()

    # Read digits
    digits = ""
    while True:
        k = get_key()
        if k in ("ENTER", "\r", "\n"):
            break
        if k in ("ESC", "q", "Q"):
            return
        if k == "BACKSPACE":
            digits = digits[:-1]
            sys.stdout.write(f"\b \b")
        elif k.isdigit():
            digits += k
            sys.stdout.write(k)
        sys.stdout.flush()

    if digits:
        try:
            target = int(digits)
            viewer.nav_goto(target)
        except ValueError:
            pass


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Interactive replay-buffer inspector for Tetris AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/inspect_buffer.py checkpoints/step_000010000_buffer.pt
  python tools/inspect_buffer.py checkpoints/step_000010000_buffer.pt --list 20
        """,
    )
    parser.add_argument("buffer_path", help="Path to *_buffer.pt checkpoint file")
    parser.add_argument(
        "--list", "-l", type=int, default=0, metavar="N",
        help="Non-interactive: print first N transitions and exit",
    )
    args = parser.parse_args()

    buf = load_buffer(args.buffer_path)
    entries = extract_transitions_with_priorities(buf)

    if not entries:
        print("Buffer is empty (no transitions stored).")
        return

    buf_meta = {
        "capacity": buf.get("capacity", 0),
        "tree_tree": buf.get("tree_tree", None),
        "max_priority": buf.get("max_priority", 1.0),
        "alpha": buf.get("alpha", 0.6),
    }

    if args.list > 0:
        n = min(args.list, len(entries))
        for i in range(n):
            t, prio = entries[i]
            d = transition_to_dict(t)
            print(f"--- Transition {i} (priority={prio:.4f}) ---")
            print(f"  action idx={d['action']}  rot={d['rotation']}  "
                  f"col={d['column']}  hold={d['hold']}")
            print(f"  reward={d['reward']:+.1f}  done={d['done']}")
            bi = d["before_info"]
            print(f"  before: piece={bi['current_piece']} rot={bi['rotation']}  "
                  f"hold=[{bi['hold']}]  next={' '.join(bi['next'])}")
            print(f"  before features: height={bi['height_sum']:.0f}  "
                  f"holes={bi['holes']:.0f}  bump={bi['bumpiness']:.0f}  "
                  f"well={bi['max_well']:.0f}")
            print(render_board(d["board"]))
            print(render_board(d["next_board"],
                               highlight=set(d["diff_cells"] or []),
                               cleared_rows=d["cleared_rows"] or set()))
            print()
        return

    viewer = BufferViewer(entries, args.buffer_path, buf_meta)
    interactive_loop(viewer)


if __name__ == "__main__":
    main()
