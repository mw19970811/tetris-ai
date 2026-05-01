"""Hardware capability probe for Tetris AI training.

Detects CPU, RAM, GPU(s), CUDA, and PyTorch configuration to inform
resource allocation decisions before training begins.
"""

import os
import sys
import platform
import multiprocessing
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class CPUInfo:
    physical_cores: int
    logical_cores: int
    model: str
    arch: str           # x86_64 / arm64 / ...
    total_ram_gb: float
    available_ram_gb: float


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_gb: float
    free_memory_gb: float
    compute_capability: str = ""
    is_cuda_available: bool = False


@dataclass
class SystemInfo:
    os: str
    python_version: str
    pytorch_version: str
    cuda_version: Optional[str]
    cpu: CPUInfo
    gpus: List[GPUInfo]
    gpu_count: int


def _get_cpu_info() -> CPUInfo:
    """Detect CPU cores and memory."""
    physical = multiprocessing.cpu_count()

    # Try to get physical vs logical cores on Linux/Windows.
    logical = physical
    model = platform.processor() or "Unknown"
    arch = platform.machine()

    try:
        if sys.platform == "linux":
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "Model name" in line:
                    model = line.split(":", 1)[1].strip()
                elif "Thread(s) per core" in line:
                    threads_per = int(line.split(":", 1)[1].strip())
                    physical = logical // max(threads_per, 1)
                elif "Architecture" in line:
                    arch = line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors"],
                text=True, stderr=subprocess.DEVNULL
            )
            lines = out.strip().splitlines()
            if len(lines) >= 2:
                parts = lines[1].strip().split()
                if len(parts) >= 3:
                    model = " ".join(parts[:-2])
                    logical = int(parts[-1])
                    physical = int(parts[-2])
            # Also try to get from environment.
            physical = os.cpu_count() or physical
            logical = physical  # fallback
    except Exception:
        pass

    # RAM detection.
    total_ram = 0.0
    available_ram = 0.0
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_ram = mem.total / (1024 ** 3)
        available_ram = mem.available / (1024 ** 3)
    except ImportError:
        # Fallback: rough estimate via OS commands or environment.
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
                mem = MEMORYSTATUSEX()
                mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
                total_ram = mem.ullTotalPhys / (1024 ** 3)
                available_ram = mem.ullAvailPhys / (1024 ** 3)
        except Exception:
            total_ram = 16.0
            available_ram = 8.0

    return CPUInfo(
        physical_cores=physical,
        logical_cores=logical,
        model=model,
        arch=arch,
        total_ram_gb=round(total_ram, 1),
        available_ram_gb=round(available_ram, 1),
    )


def _get_gpu_info() -> List[GPUInfo]:
    """Detect GPU(s) via PyTorch and nvidia-smi."""
    gpus = []
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            name = props.name
            total_mem = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
            total_mem /= (1024 ** 3)
            cap = f"{props.major}.{props.minor}"

            # Try to get free memory via nvidia-smi.
            free_mem = 0.0
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={i}"],
                    text=True, stderr=subprocess.DEVNULL
                )
                free_mem = float(out.strip()) / 1024.0
            except Exception:
                free_mem = total_mem * 0.7  # estimate

            gpus.append(GPUInfo(
                index=i,
                name=name,
                memory_gb=round(total_mem, 1),
                free_memory_gb=round(free_mem, 1),
                compute_capability=cap,
                is_cuda_available=True,
            ))
    except ImportError:
        pass

    return gpus


def probe() -> SystemInfo:
    """Run full hardware probe and return system info."""
    import torch

    cpu = _get_cpu_info()
    gpus = _get_gpu_info()

    cuda_ver = None
    if gpus:
        cuda_ver = torch.version.cuda

    return SystemInfo(
        os=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        cuda_version=cuda_ver,
        cpu=cpu,
        gpus=gpus,
        gpu_count=len(gpus),
    )
