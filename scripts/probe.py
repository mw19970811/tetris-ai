#!/usr/bin/env python3
"""Hardware probe: detect system capabilities and recommend optimal training config.

Usage:
    python scripts/probe.py              # Full diagnostic report
    python scripts/probe.py --json       # Machine-readable JSON output
    python scripts/probe.py --brief      # One-line recommendation only
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.hardware_probe import probe, SystemInfo, CPUInfo, GPUInfo
from trainer.resource_planner import plan, ResourcePlan, apply_plan


def _format_gb(val_gb: float) -> str:
    return f"{val_gb:.1f} GB"


def _format_mb(val_mb: float) -> str:
    if val_mb >= 1024:
        return f"{val_mb/1024:.1f} GB"
    return f"{val_mb:.0f} MB"


def print_report(info: SystemInfo, rp: ResourcePlan):
    """Print a formatted diagnostic report."""
    sep = "─" * 62

    print(f"\n{'='*62}")
    print("  Tetris AI — Hardware Probe & Resource Planner")
    print(f"{'='*62}")

    # ── System ──
    print(f"\n  System")
    print(f"  {sep}")
    print(f"  OS           : {info.os}")
    print(f"  Python       : {info.python_version}")
    print(f"  PyTorch      : {info.pytorch_version}")
    print(f"  CUDA         : {info.cuda_version or 'N/A'}")

    # ── CPU ──
    cpu = info.cpu
    print(f"\n  CPU")
    print(f"  {sep}")
    print(f"  Model        : {cpu.model}")
    print(f"  Architecture : {cpu.arch}")
    print(f"  Physical cores : {cpu.physical_cores}")
    print(f"  Logical cores  : {cpu.logical_cores}")
    print(f"  Total RAM    : {_format_gb(cpu.total_ram_gb)}")
    print(f"  Available    : {_format_gb(cpu.available_ram_gb)}")

    # ── GPU ──
    print(f"\n  GPU  ({info.gpu_count} detected)")
    print(f"  {sep}")
    if info.gpu_count == 0:
        print("  (none — will train on CPU)")
    else:
        for g in info.gpus:
            cc = f"  CC={g.compute_capability}" if g.compute_capability else ""
            print(f"  GPU {g.index}: {g.name}")
            print(f"       Memory: {_format_gb(g.memory_gb)} total, {_format_gb(g.free_memory_gb)} free{cc}")

    # ── Recommendation ──
    print(f"\n  Recommendation")
    print(f"  {sep}")
    mode_label = {
        "cpu": "CPU-only",
        "cuda": "Single-GPU CUDA",
        "cuda_multi_gpu": f"Multi-GPU CUDA ({info.gpu_count} GPUs)",
    }
    print(f"  Mode         : {mode_label.get(rp.mode, rp.mode)}")
    print(f"  Num envs     : {rp.recommended_num_envs}")
    print(f"  Batch size   : {rp.recommended_batch_size}")
    print(f"  Replay cap   : {rp.recommended_replay_capacity:,}")
    print(f"  CPU threads  : {rp.recommended_num_threads}")

    # ── Memory Budget ──
    print(f"\n  Memory Budget (estimated)")
    print(f"  {sep}")
    for key, label in [
        ("environments", "Environments"),
        ("replay_buffer", "Replay buffer"),
        ("model_weights", "Model weights"),
        ("batch_training", "Batch training peak"),
        ("pytorch_overhead", "PyTorch overhead"),
        ("os_buffer", "OS buffer"),
    ]:
        val = rp.memory_budget_mb.get(key, 0)
        pct = (val / rp.total_estimated_mb * 100) if rp.total_estimated_mb > 0 else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"  {label:<18s}  {_format_mb(val):>8s}  {bar}")

    print(f"  {sep}")
    print(f"  {'Total estimated':<18s}  {_format_mb(rp.total_estimated_mb):>8s}")

    # ── Warnings ──
    if rp.warnings:
        print(f"\n  Warnings")
        print(f"  {sep}")
        for w in rp.warnings:
            print(f"  ⚠  {w}")

    # ── Notes ──
    if rp.notes:
        print(f"\n  Notes")
        print(f"  {sep}")
        for n in rp.notes:
            print(f"  - {n}")

    # ── Quick start ──
    print(f"\n  Quick start")
    print(f"  {sep}")
    device = "cuda" if rp.mode != "cpu" else "cpu"
    print(f"  python scripts/train.py --device {device} --envs {rp.recommended_num_envs}")
    if rp.mode == "cpu":
        print(f"  # CPU mode: set OMP_NUM_THREADS={rp.recommended_num_threads} for optimal BLAS perf.")
    print(f"{'='*62}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hardware probe for Tetris AI training")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--brief", action="store_true", help="One-line recommendation only")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--envs", type=int, default=None, help="Override num_envs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch_size")
    parser.add_argument("--steps", type=int, default=50_000_000, help="Total training steps")
    args = parser.parse_args()

    # Force UTF-8 output on Windows.
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("[Probe] Detecting hardware ...", file=sys.stderr)
    info = probe()
    rp = plan(info, algorithm=args.algo,
              user_envs=args.envs, user_batch=args.batch,
              user_total_steps=args.steps)

    if args.brief:
        mode_map = {"cpu": "CPU", "cuda": "CUDA", "cuda_multi_gpu": "Multi-GPU"}
        print(f"{mode_map.get(rp.mode, rp.mode)} | envs={rp.recommended_num_envs} "
              f"batch={rp.recommended_batch_size} replay={rp.recommended_replay_capacity:,} "
              f"threads={rp.recommended_num_threads} mem={rp.total_estimated_mb:.0f}MB")
        return

    if args.json:
        out = {
            "os": info.os,
            "python": info.python_version,
            "pytorch": info.pytorch_version,
            "cuda": info.cuda_version,
            "cpu": {
                "model": info.cpu.model,
                "physical_cores": info.cpu.physical_cores,
                "logical_cores": info.cpu.logical_cores,
                "total_ram_gb": info.cpu.total_ram_gb,
                "available_ram_gb": info.cpu.available_ram_gb,
            },
            "gpus": [{"index": g.index, "name": g.name,
                      "memory_gb": g.memory_gb, "free_memory_gb": g.free_memory_gb,
                      "compute_capability": g.compute_capability} for g in info.gpus],
            "recommendation": {
                "mode": rp.mode,
                "num_envs": rp.recommended_num_envs,
                "batch_size": rp.recommended_batch_size,
                "replay_capacity": rp.recommended_replay_capacity,
                "num_threads": rp.recommended_num_threads,
                "total_estimated_mb": rp.total_estimated_mb,
                "memory_budget": rp.memory_budget_mb,
            },
            "warnings": rp.warnings,
            "notes": rp.notes,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print_report(info, rp)


if __name__ == "__main__":
    main()
