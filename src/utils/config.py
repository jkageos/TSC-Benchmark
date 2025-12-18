"""
Configuration management for TSC-Benchmark.

Handles:
- YAML config loading and validation
- Model factory pattern for instantiation
- Optimizer creation
- Config extraction utilities
- Memory-aware model capacity adjustment
"""

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration structure.

    Ensures all required keys exist and have valid types.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["datasets", "models", "training"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate datasets list
    if not isinstance(config["datasets"], list) or len(config["datasets"]) == 0:
        raise ValueError("datasets must be a non-empty list")

    # Validate models dict
    if not isinstance(config["models"], dict):
        raise ValueError("models must be a dictionary")

    # Validate training config
    training_config = config["training"]
    required_training_keys = ["epochs", "batch_size", "learning_rate"]

    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training config key: {key}")


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info()[0] / 1e9
    return 0.0


def reduce_model_capacity(model_config: dict[str, Any], reduction_factor: float = 0.5) -> dict[str, Any]:
    """
    Reduce model capacity to fit in memory.

    Args:
        model_config: Original model configuration
        reduction_factor: Factor to reduce capacity (0.0-1.0)

    Returns:
        Reduced model configuration
    """
    reduced = model_config.copy()

    # Reduce transformer-specific parameters
    if "d_model" in reduced:
        reduced["d_model"] = max(64, int(reduced["d_model"] * reduction_factor))

    if "d_ff" in reduced:
        reduced["d_ff"] = max(256, int(reduced["d_ff"] * reduction_factor))

    if "num_layers" in reduced:
        reduced["num_layers"] = max(1, int(reduced["num_layers"] * reduction_factor))

    if "num_heads" in reduced:
        reduced["num_heads"] = max(1, int(reduced["num_heads"] * reduction_factor))

    # Reduce CNN filters
    if "num_filters" in reduced and isinstance(reduced["num_filters"], list):
        reduced["num_filters"] = [max(16, int(f * reduction_factor)) for f in reduced["num_filters"]]

    # Reduce FCN hidden dims
    if "hidden_dims" in reduced and isinstance(reduced["hidden_dims"], list):
        reduced["hidden_dims"] = [max(64, int(d * reduction_factor)) for d in reduced["hidden_dims"]]

    return reduced


def create_model(
    model_name: str,
    model_config: dict[str, Any],
    num_classes: int,
    input_length: int,
    input_channels: int = 1,
    auto_reduce: bool = True,
) -> nn.Module:
    """
    Factory function to create models from config.

    Args:
        model_name: Name of the model ('fcn', 'cnn', 'transformer', 'cats', 'autoformer', 'patchtst')
        model_config: Model-specific hyperparameters
        num_classes: Number of classification classes
        input_length: Input sequence length
        input_channels: Number of input channels
        auto_reduce: Automatically reduce capacity if GPU memory is low

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not recognized
    """
    from src.models.autoformer import Autoformer
    from src.models.cats import CATS
    from src.models.cnn import CNN
    from src.models.fcn import FCN
    from src.models.patchtst import PatchTST
    from src.models.transformer import Transformer

    # Registry pattern for model instantiation
    model_registry = {
        "fcn": FCN,
        "cnn": CNN,
        "transformer": Transformer,
        "cats": CATS,
        "autoformer": Autoformer,
        "patchtst": PatchTST,
    }

    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_registry.keys())}")

    # Auto-reduce capacity if GPU memory is low
    config = model_config.copy()
    if auto_reduce and torch.cuda.is_available():
        gpu_memory_gb = get_gpu_memory_gb()

        # Aggressive reduction for very low memory
        if gpu_memory_gb < 3.0:
            print(f"⚠️  Low GPU memory ({gpu_memory_gb:.2f}GB). Reducing model capacity by 50%.")
            config = reduce_model_capacity(config, reduction_factor=0.5)
        elif gpu_memory_gb < 5.0:
            print(f"⚠️  Medium GPU memory ({gpu_memory_gb:.2f}GB). Reducing model capacity by 25%.")
            config = reduce_model_capacity(config, reduction_factor=0.75)

    model_class = model_registry[model_name]

    # Instantiate with standard args + config overrides
    return model_class(
        num_classes=num_classes,
        input_length=input_length,
        input_channels=input_channels,
        **config,
    )


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    **kwargs: Any,
) -> optim.Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments (e.g., weight_decay)

    Returns:
        Configured optimizer

    Raises:
        ValueError: If optimizer_name is not recognized
    """
    optimizer_lower = optimizer_name.lower()

    if optimizer_lower == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_lower == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_lower == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Extract model configuration from main config."""
    return config["models"].get(model_name, {})


def get_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract training configuration from main config."""
    return config["training"]
