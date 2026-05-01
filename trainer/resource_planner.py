"""Resource planner: estimates memory and recommends optimal training config.

Analyses hardware probe results and produces a tuned TrainingConfig
tailored to the available CPU / GPU / RAM resources.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from .hardware_probe import SystemInfo, CPUInfo, GPUInfo
from .config import TrainingConfig, EnvConfig, NetworkConfig, DQNConfig, PPOConfig


# --- Memory Budget Constants (per unit, in MB) ---
MB_PER_ENV = 2.0             # board + features + workspace per env instance
MB_PER_TRANSITION = 0.002    # ~2 KB per stored (s,a,r,s',d) in replay buffer
MB_MODEL_WEIGHTS = 10.0      # DuelingDQN weights (~2-5 MB; padded overhead)
MB_BATCH_PEAK = 200.0        # Peak for forward+backward pass with batch_size=32
MB_PYTORCH_OVERHEAD = 300.0  # CUDA context, allocator overhead
MB_OS_BUFFER = 500.0         # Leave room for OS + other processes

# --- Recommended defaults by tier ---
TIER_HIGH = "high"        # >= 24 GB GPU, >= 64 GB RAM
TIER_MEDIUM = "medium"    # >= 8 GB GPU, >= 16 GB RAM
TIER_LOW = "low"          # CPU-only or < 8 GB GPU
TIER_MINIMAL = "minimal"  # < 8 GB RAM total


@dataclass
class ResourcePlan:
    """Recommended resource allocation for training."""

    mode: str                        # "cpu", "cuda", "cuda_multi_gpu"
    recommended_num_envs: int
    recommended_batch_size: int
    recommended_replay_capacity: int
    recommended_num_threads: int
    memory_budget_mb: Dict[str, float]  # breakdown
    total_estimated_mb: float
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _determine_tier(info: SystemInfo) -> str:
    """Classify system into a resource tier."""
    gpu_mem = max((g.memory_gb for g in info.gpus), default=0)
    ram = info.cpu.available_ram_gb

    if gpu_mem >= 20 and ram >= 48:
        return TIER_HIGH
    elif gpu_mem >= 6 and ram >= 12:
        return TIER_MEDIUM
    elif gpu_mem > 0 or ram >= 8:
        return TIER_LOW
    return TIER_MINIMAL


def _estimate_memory(num_envs: int, batch_size: int,
                     replay_capacity: int) -> Dict[str, float]:
    """Estimate peak memory usage in MB."""
    env_mb = num_envs * MB_PER_ENV
    replay_mb = replay_capacity * MB_PER_TRANSITION
    batch_mb = (batch_size / 32) * MB_BATCH_PEAK
    return {
        "environments": env_mb,
        "replay_buffer": replay_mb,
        "model_weights": MB_MODEL_WEIGHTS,
        "batch_training": batch_mb,
        "pytorch_overhead": MB_PYTORCH_OVERHEAD,
        "os_buffer": MB_OS_BUFFER,
    }


def plan(info: SystemInfo, algorithm: str = "dqn",
         user_envs: Optional[int] = None,
         user_batch: Optional[int] = None,
         user_total_steps: int = 50_000_000) -> ResourcePlan:
    """Generate a resource plan based on hardware probe results."""

    tier = _determine_tier(info)
    gpu_count = info.gpu_count
    gpu_mem = max((g.memory_gb for g in info.gpus), default=0)
    gpu_free = max((g.free_memory_gb for g in info.gpus), default=0)
    ram_avail = info.cpu.available_ram_gb
    cpu_cores = info.cpu.physical_cores
    warnings = []
    notes = []

    # --- Determine training mode ---
    if gpu_count >= 2 and gpu_mem >= 10:
        mode = "cuda_multi_gpu"
        notes.append(f"Detected {gpu_count} GPUs — can use DataParallel/DDP.")
    elif gpu_count == 1 and gpu_mem >= 2:
        mode = "cuda"
    else:
        mode = "cpu"
        if gpu_count == 0:
            warnings.append("No CUDA GPU detected. Training on CPU — expect 10-50x slower.")
        else:
            warnings.append(f"GPU '{info.gpus[0].name}' has only {gpu_mem:.1f} GB VRAM (min 2 GB needed). Falling back to CPU.")

    # --- Determine resource limits ---
    if mode == "cpu":
        # CPU training: memory-bound by RAM.
        usable_ram_mb = (ram_avail - 1.0) * 1024  # leave 1 GB for OS
        max_envs = max(1, int(usable_ram_mb / MB_PER_ENV))
        recommended_envs = min(user_envs or cpu_cores, max_envs, 64)
        batch_size = min(user_batch or 32, 32)
        replay_cap = min(500_000, int(usable_ram_mb / MB_PER_TRANSITION))
        threads = min(cpu_cores, recommended_envs)
        notes.append(f"CPU mode: {cpu_cores} physical cores → {recommended_envs} envs, {threads} threads.")

        # Check if threads are suboptimal.
        if cpu_cores < 4:
            warnings.append(f"Only {cpu_cores} CPU cores — training will be slow. Consider a machine with 8+ cores.")

    elif mode == "cuda":
        # Single GPU: VRAM-bound.
        usable_vram_mb = (gpu_free - 0.3) * 1024  # leave 300 MB headroom
        max_envs = max(1, int(usable_vram_mb * 0.3 / MB_PER_ENV))  # envs take ~30% of budget
        recommended_envs = min(user_envs or 64, max_envs, 128)
        batch_size = min(user_batch or 32, 64)
        replay_cap = min(1_000_000, int(ram_avail * 1024 / MB_PER_TRANSITION))
        threads = cpu_cores
        notes.append(f"CUDA mode: {info.gpus[0].name} ({gpu_mem:.1f} GB VRAM, {gpu_free:.1f} GB free).")

        if gpu_mem < 8:
            warnings.append(f"GPU has {gpu_mem:.1f} GB VRAM — consider reducing num_envs to {max(1, recommended_envs//2)}.")
            recommended_envs = max(1, recommended_envs // 2)

    elif mode == "cuda_multi_gpu":
        usable_vram_mb = (gpu_free - 0.5) * 1024
        max_envs = max(1, int(usable_vram_mb * 0.2 / MB_PER_ENV))
        recommended_envs = min(user_envs or gpu_count * 32, max_envs, 256)
        batch_size = min(user_batch or 64, 128)
        replay_cap = min(2_000_000, int(ram_avail * 1024 / MB_PER_TRANSITION))
        threads = cpu_cores
        notes.append(f"Multi-GPU mode: {gpu_count} × {info.gpus[0].name} ({gpu_mem:.1f} GB each).")

    else:
        recommended_envs = 1
        batch_size = 16
        replay_cap = 100_000
        threads = 1

    # Apply user overrides.
    if user_envs:
        recommended_envs = user_envs
        notes.append(f"User override: num_envs = {user_envs}.")
    if user_batch:
        batch_size = user_batch

    # Clip to safe minimums.
    recommended_envs = max(1, recommended_envs)
    batch_size = max(8, batch_size)
    replay_cap = max(100_000, min(replay_cap, 2_000_000))

    # Estimate memory.
    mem_budget = _estimate_memory(recommended_envs, batch_size, replay_cap)
    total_mb = sum(mem_budget.values())

    if mode == "cpu":
        total_ram_mb = ram_avail * 1024
        if total_mb > total_ram_mb * 0.85:
            warnings.append(
                f"Estimated memory ({total_mb:.0f} MB) exceeds 85% of available RAM "
                f"({total_ram_mb:.0f} MB). Reduce num_envs or replay capacity."
            )
    else:
        total_vram_mb = gpu_free * 1024
        if mem_budget["batch_training"] > total_vram_mb * 0.8:
            warnings.append("Batch size may exceed GPU VRAM. Reduce batch_size.")

    return ResourcePlan(
        mode=mode,
        recommended_num_envs=recommended_envs,
        recommended_batch_size=batch_size,
        recommended_replay_capacity=replay_cap,
        recommended_num_threads=threads,
        memory_budget_mb=mem_budget,
        total_estimated_mb=round(total_mb, 1),
        warnings=warnings,
        notes=notes,
    )


def apply_plan(plan: ResourcePlan, algorithm: str = "dqn",
               total_steps: int = 50_000_000) -> TrainingConfig:
    """Apply a resource plan to produce a concrete TrainingConfig."""
    env_cfg = EnvConfig()
    net_cfg = NetworkConfig()

    dqn_cfg = DQNConfig(
        batch_size=plan.recommended_batch_size,
        replay_capacity=plan.recommended_replay_capacity,
    )
    ppo_cfg = PPOConfig(
        batch_size=plan.recommended_batch_size,
        num_envs=plan.recommended_num_envs,
    )

    device = "cuda" if plan.mode.startswith("cuda") else "cpu"

    return TrainingConfig(
        algorithm=algorithm,
        total_steps=total_steps,
        num_envs=plan.recommended_num_envs,
        device=device,
        env=env_cfg,
        network=net_cfg,
        dqn=dqn_cfg,
        ppo=ppo_cfg,
    )
