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


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


def clear_cuda_memory() -> None:
    """Aggressively clear CUDA cache and Python garbage collection."""
    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def aggressive_memory_cleanup() -> None:
    """Perform aggressive memory cleanup before model creation."""
    clear_cuda_memory()
    if torch.cuda.is_available() and platform.system() != "Windows":
        # Force garbage collection (Linux/Mac only - not available on Windows)
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass  # Silently skip if not available


def estimate_model_memory_gb(
    model_name: str,
    d_model: int,
    num_layers: int,
    sequence_length: int,
    batch_size: int,
) -> float:
    """
    Estimate GPU memory needed for model + activations (rough estimate).

    Args:
        model_name: Name of model
        d_model: Model dimension
        num_layers: Number of layers
        sequence_length: Sequence length
        batch_size: Batch size

    Returns:
        Estimated memory in GB
    """
    # Model parameters (float32 = 4 bytes per param)
    if model_name in ("transformer", "cats", "autoformer", "patchtst"):
        # Transformer: ~d_model * seq_len * num_layers parameters + embeddings
        param_memory = (d_model * sequence_length * num_layers * 4) / 1e9
    elif model_name == "cnn":
        # CNN: ~filters * kernel_size * layers
        param_memory = (64 * 128 * 256 * 9 * 4) / 1e9  # rough estimate
    else:  # fcn
        param_memory = (256 * 128 * 4) / 1e9

    # Activation memory (batch_size * seq_len * d_model * num_layers)
    activation_memory = (batch_size * sequence_length * d_model * num_layers * 4) / 1e9

    # Optimizer state (Adam: 2x parameters for momentum + variance)
    optimizer_memory = param_memory * 2

    # Total with safety margin (1.3x)
    total = (param_memory + activation_memory + optimizer_memory) * 1.3

    return total


def adjust_hyperparameters_for_memory(
    dataset_name: str,
    dataset_info: dict[str, Any],
    batch_size: int,
    model_name: str,
    model_config: dict[str, Any],
    gpu_memory_gb: float,
) -> tuple[int, int, dict[str, Any]]:
    """
    Dynamically adjust batch size, sequence length, and model config based on GPU memory.

    Returns:
        (adjusted_batch_size, adjusted_max_length, adjusted_model_config)
    """
    seq_length = dataset_info["sequence_length"]
    dataset_size = dataset_info["n_train"]
    adjusted_config = model_config.copy()
    adjusted_max_length = seq_length
    adjusted_batch_size = batch_size

    # Available memory (reserve 1GB for safety)
    available_memory = max(0.5, gpu_memory_gb - 1.0)

    # Estimate memory for current config
    d_model = adjusted_config.get("d_model", 128)
    num_layers = adjusted_config.get("num_layers", 2)
    estimated_memory = estimate_model_memory_gb(model_name, d_model, num_layers, seq_length, batch_size)

    # Strategy 1: If memory estimate exceeds available, reduce aggressively
    if estimated_memory > available_memory:
        reduction_needed = estimated_memory / available_memory

        # First: truncate sequence length
        if seq_length > 256:
            target_seq_length = max(128, int(seq_length / (reduction_needed**0.4)))
            adjusted_max_length = target_seq_length
            estimated_memory = estimate_model_memory_gb(model_name, d_model, num_layers, target_seq_length, batch_size)

        # Second: reduce batch size
        if estimated_memory > available_memory and batch_size > 4:
            target_batch = max(4, int(batch_size / (estimated_memory / available_memory)))
            adjusted_batch_size = target_batch
            estimated_memory = estimate_model_memory_gb(
                model_name, d_model, num_layers, adjusted_max_length, target_batch
            )

        # Third: reduce model capacity
        if estimated_memory > available_memory:
            reduction_factor = available_memory / estimated_memory

            if "d_model" in adjusted_config:
                adjusted_config["d_model"] = max(64, int(d_model * reduction_factor))

            if "d_ff" in adjusted_config:
                adjusted_config["d_ff"] = max(256, int(adjusted_config.get("d_ff", 512) * reduction_factor))

            if "num_layers" in adjusted_config:
                adjusted_config["num_layers"] = max(1, int(num_layers * reduction_factor))

            if "num_heads" in adjusted_config:
                old_heads = adjusted_config["num_heads"]
                new_heads = max(1, int(old_heads * reduction_factor))
                # Ensure d_model is divisible by num_heads
                adjusted_d_model = adjusted_config["d_model"]
                while new_heads > 0 and adjusted_d_model % new_heads != 0:
                    new_heads -= 1
                adjusted_config["num_heads"] = max(1, new_heads)

            if "num_filters" in adjusted_config and isinstance(adjusted_config["num_filters"], list):
                adjusted_config["num_filters"] = [
                    max(16, int(f * reduction_factor)) for f in adjusted_config["num_filters"]
                ]

            if "hidden_dims" in adjusted_config and isinstance(adjusted_config["hidden_dims"], list):
                adjusted_config["hidden_dims"] = [
                    max(64, int(d * reduction_factor)) for d in adjusted_config["hidden_dims"]
                ]

    # KeplerLightCurves specific: very aggressive truncation
    if dataset_name == "KeplerLightCurves" and seq_length == 512:
        adjusted_max_length = min(adjusted_max_length, 256)
        adjusted_batch_size = min(adjusted_batch_size, 8)

    # Long sequence datasets
    elif seq_length > 1000:
        adjusted_max_length = min(adjusted_max_length, 512)
        adjusted_batch_size = min(adjusted_batch_size, batch_size // 2)

    # Small dataset: optimize for gradient quality
    elif dataset_size < 100:
        adjusted_batch_size = min(adjusted_batch_size, 16)

    return adjusted_batch_size, adjusted_max_length, adjusted_config


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


def recommend_batch_size(
    dataset_size: int,
    sequence_length: int,
    gpu_memory_gb: float = 6.0,
) -> int:
    """
    Recommend batch size for max throughput while respecting memory.

    Strategy:
    - Large sequences + low memory → smaller batch
    - Small sequences + high memory → larger batch
    - Small datasets → optimize for gradient quality over speed
    """
    # Base on available memory and sequence complexity
    memory_factor = int(max(8, min(48, int(gpu_memory_gb * 4))))

    # Adjust for sequence length
    if sequence_length > 2000:
        memory_factor = max(8, memory_factor // 2)
    elif sequence_length < 200:
        memory_factor = min(memory_factor * 2, 64)

    # For very small datasets, don't go too large (hurts gradient diversity)
    if dataset_size < 200:
        return min(memory_factor, 16)

    return int(memory_factor)
