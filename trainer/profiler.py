"""Per-phase cumulative timing for the training loop.

Integrates into Trainer.train() to record wall-clock time spent in each
phase of the step loop: env reset, legal-actions query, action selection,
env step, and model update.  Reports cumulative breakdown every log interval.

Usage:
    profiler = TrainingProfiler(enabled=True)
    with profiler.phase("env_step"):
        env.step(action)
    profiler.report(step, elapsed)  # prints breakdown
"""

import time
from contextlib import contextmanager
from typing import Dict, Optional


class TrainingProfiler:
    """Cumulative wall-clock profiler for the training loop.

    Each ``phase(name)`` context manager records the elapsed time under
    *name*.  Call ``report()`` to print cumulative absolute + percentage
    breakdown.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._accum: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    @contextmanager
    def phase(self, name: str):
        """Context manager that records elapsed time under *name*."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._accum[name] = self._accum.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1

    # ------------------------------------------------------------------ #
    def reset(self):
        """Clear all accumulated timings."""
        self._accum.clear()
        self._counts.clear()

    # ------------------------------------------------------------------ #
    def report(self, step: int, total_elapsed: float,
               env_count: int = 1) -> Optional[str]:
        """Return a one-line summary string, or None when disabled.

        *total_elapsed* is the wall-clock time since training started;
        *env_count* is used to derive per-env-step averages.
        """
        if not self.enabled or not self._accum:
            return None

        total_tracked = sum(self._accum.values())
        parts = []
        # Keep ordering deterministic.
        for name in sorted(self._accum.keys()):
            t = self._accum[name]
            pct = (t / total_elapsed * 100) if total_elapsed > 0 else 0.0
            n = self._counts.get(name, 0)
            us_per = (t / n * 1e6) if n > 0 else 0.0
            parts.append(f"{name}: {t:6.1f}s ({pct:5.1f}%) [{us_per:6.0f}us/call]")

        uncovered = max(0, total_elapsed - total_tracked)
        if uncovered > 0.01:
            pct_u = uncovered / total_elapsed * 100
            parts.append(f"other: {uncovered:.1f}s ({pct_u:.1f}%)")

        return f"[Profiler @ step {step:>9,}]  " + "  |  ".join(parts)
