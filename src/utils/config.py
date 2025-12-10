"""
Configuration management for TSC-Benchmark.

Handles YAML config loading, validation, and model factory pattern.
"""

from pathlib import Path
from typing import Any

import torch.nn as nn
import torch.optim as optim
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["datasets", "models", "training"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate datasets
    if not isinstance(config["datasets"], list) or len(config["datasets"]) == 0:
        raise ValueError("datasets must be a non-empty list")

    # Validate models
    if not isinstance(config["models"], dict):
        raise ValueError("models must be a dictionary")

    # Validate training config
    training_config = config["training"]
    required_training_keys = ["epochs", "batch_size", "learning_rate"]

    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training config key: {key}")


def create_model(
    model_name: str,
    model_config: dict[str, Any],
    num_classes: int,
    input_length: int,
    input_channels: int = 1,
) -> nn.Module:
    """
    Factory function to create models from config.

    Args:
        model_name: Name of the model ('fcn', 'cnn', 'transformer')
        model_config: Model-specific configuration
        num_classes: Number of classification classes
        input_length: Input sequence length
        input_channels: Number of input channels

    Returns:
        Instantiated model
    """
    from src.models.cnn import CNN
    from src.models.fcn import FCN
    from src.models.transformer import Transformer

    model_registry = {
        "fcn": FCN,
        "cnn": CNN,
        "transformer": Transformer,
    }

    if model_name.lower() not in model_registry:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(model_registry.keys())}"
        )

    model_class = model_registry[model_name.lower()]

    # Combine required parameters with model config
    model_params = {
        "num_classes": num_classes,
        "input_length": input_length,
        "input_channels": input_channels,
        **model_config,
    }

    return model_class(**model_params)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    **kwargs: Any,
) -> optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters

    Returns:
        Optimizer instance
    """
    optimizer_registry = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
    }

    if optimizer_name.lower() not in optimizer_registry:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available optimizers: {list(optimizer_registry.keys())}"
        )

    optimizer_class = optimizer_registry[optimizer_name.lower()]

    return optimizer_class(model.parameters(), lr=learning_rate, **kwargs)


def get_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """
    Extract model-specific configuration.

    Args:
        config: Full configuration dictionary
        model_name: Name of the model

    Returns:
        Model configuration dictionary
    """
    if model_name not in config["models"]:
        raise ValueError(f"Model '{model_name}' not found in config")

    return config["models"][model_name]


def get_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract training configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Training configuration dictionary
    """
    return config["training"]
