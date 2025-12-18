"""
System resource management and multiprocessing safety.

Provides safe defaults and monitoring for CPU/GPU parallel workloads.
"""

import os
import platform
from typing import Any

import psutil
import torch


def get_safe_num_workers(
    max_cpu_load: float = 0.5,
    reserve_cores: int = 1,
    min_workers: int = 0,
    max_workers: int | None = None,
) -> int:
    """
    Calculate safe number of DataLoader workers based on system resources.

    Safety mechanisms:
    - Respects max_cpu_load to avoid system overload
    - Reserves cores for main process and OS
    - Caps workers based on available memory
    - Platform-specific adjustments

    Args:
        max_cpu_load: Maximum CPU utilization (0.0-1.0), default 0.5 (50%)
        reserve_cores: Cores to reserve for main process/OS, default 1
        min_workers: Minimum workers (default 0)
        max_workers: Maximum workers cap (default None = auto)

    Returns:
        Safe number of workers
    """
    cpu_count = os.cpu_count() or 1
    available_memory_gb = psutil.virtual_memory().available / 1e9

    # Reserve cores for main process and OS
    usable_cores = max(1, cpu_count - reserve_cores)

    # Apply max_cpu_load constraint
    workers_from_cpu = int(usable_cores * max_cpu_load)

    # Memory constraint: ~2GB per worker for preprocessing + pin_memory buffers
    workers_from_memory = int(available_memory_gb / 2.0)

    # Take minimum of constraints
    safe_workers = max(min_workers, min(workers_from_cpu, workers_from_memory))

    # Apply user-specified cap if provided
    if max_workers is not None:
        safe_workers = min(safe_workers, max_workers)

    # Platform-specific adjustments
    if platform.system() == "Windows":
        # Windows has higher multiprocessing overhead
        safe_workers = min(safe_workers, 4)

    return safe_workers


def get_system_resources() -> dict[str, Any]:
    """
    Get current system resource utilization.

    Returns:
        Dict with CPU, memory, GPU stats
    """
    resources = {
        "cpu_count": os.cpu_count() or 1,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total_gb": psutil.virtual_memory().total / 1e9,
        "memory_available_gb": psutil.virtual_memory().available / 1e9,
        "memory_percent": psutil.virtual_memory().percent,
    }

    if torch.cuda.is_available():
        resources.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            }
        )

    return resources


def recommend_num_workers(
    batch_size: int,
    sequence_length: int,
    dataset_size: int,
    max_cpu_load: float = 0.5,
) -> dict[str, Any]:
    """
    Recommend optimal num_workers based on dataset characteristics.

    Strategy:
    - Small datasets (< 500 samples): 0 workers (overhead dominates)
    - Medium datasets (500-5000): 2-4 workers
    - Large datasets (> 5000): 4-8 workers
    - Long sequences (> 1000): Reduce workers (expensive preprocessing)

    Args:
        batch_size: Training batch size
        sequence_length: Time series length
        dataset_size: Number of training samples
        max_cpu_load: Maximum CPU utilization

    Returns:
        Dict with recommended workers and reasoning
    """
    base_workers = get_safe_num_workers(max_cpu_load=max_cpu_load)

    # Small dataset heuristic
    if dataset_size < 500:
        recommended = 0
        reason = "Small dataset: main process overhead lower than multiprocessing"

    # Long sequence heuristic
    elif sequence_length > 2000:
        recommended = min(2, base_workers)
        reason = "Long sequences: reduce workers to limit memory pressure"

    # Large dataset
    elif dataset_size > 5000:
        recommended = min(base_workers, 8)
        reason = "Large dataset: maximize workers for throughput"

    # Medium dataset
    else:
        recommended = min(base_workers, 4)
        reason = "Medium dataset: moderate worker count"

    # GPU bottleneck check
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        estimated_batch_memory_gb = (batch_size * sequence_length * 4) / 1e9  # float32

        if estimated_batch_memory_gb > gpu_memory_gb * 0.3:
            # GPU will be bottleneck, CPU workers add minimal value
            if recommended > 2:
                recommended = 2
                reason += " (GPU memory bottleneck detected)"

    return {
        "recommended_workers": recommended,
        "max_safe_workers": base_workers,
        "reason": reason,
        "system_resources": get_system_resources(),
    }


def validate_worker_config(num_workers: int, max_cpu_load: float = 0.5) -> tuple[int, str]:
    """
    Validate and clamp num_workers to safe range.

    Args:
        num_workers: Requested number of workers
        max_cpu_load: Maximum CPU utilization

    Returns:
        Tuple of (safe_num_workers, warning_message)
    """
    max_safe = get_safe_num_workers(max_cpu_load=max_cpu_load)

    if num_workers > max_safe:
        warning = (
            f"⚠️  Requested num_workers={num_workers} exceeds safe limit ({max_safe})\n"
            f"   Clamping to {max_safe} to prevent system overload\n"
            f"   Adjust max_cpu_load in config to override"
        )
        return max_safe, warning

    if num_workers < 0:
        warning = f"⚠️  Invalid num_workers={num_workers}, using 0"
        return 0, warning

    return num_workers, ""
