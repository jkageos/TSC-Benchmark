"""
K-fold stratified cross-validation for small datasets.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.loader import TimeSeriesDataset
from src.training.trainer import Trainer
from src.utils.system import clear_cuda_memory, get_safe_num_workers


def cross_validate_dataset(
    model_fn: Callable[[], nn.Module],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
    criterion: nn.Module,
    num_classes: int,
    k_folds: int = 5,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 15,
    hardware_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    save_checkpoints: bool = True,
    checkpoint_base_dir: str | None = None,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation on a dataset.

    Returns:
        Dictionary with mean/std metrics across folds, including:
        - mean_accuracy, std_accuracy
        - mean_f1_macro, std_f1_macro
        - mean_f1_weighted, std_f1_weighted
        - mean_precision, std_precision
        - mean_recall, std_recall
    """
    if hardware_config is None:
        hardware_config = {}
    if training_config is None:
        training_config = {}

    device_name = hardware_config.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    use_compile = hardware_config.get("use_compile", True)
    compile_mode = hardware_config.get("compile_mode", "default")
    use_amp = hardware_config.get("use_amp", True)
    max_cpu_load = hardware_config.get("max_cpu_load", 0.5)
    max_workers_override = hardware_config.get("max_workers_override")

    # Extract training settings
    use_scheduler = training_config.get("use_scheduler", True)
    warmup_epochs = training_config.get("warmup_epochs", 5)
    use_augmentation = training_config.get("use_augmentation", True)
    augmentation_params = training_config.get("augmentation_params", {})
    use_tta = training_config.get("use_tta", False)
    tta_augmentations = training_config.get("tta_augmentations", 5)
    use_swa = training_config.get("use_swa", False)
    swa_start = training_config.get("swa_start", 60)
    num_workers = training_config.get("num_workers", 0)

    # Validate num_workers
    num_workers = min(
        num_workers,
        get_safe_num_workers(
            max_cpu_load=max_cpu_load,
            max_workers=max_workers_override,
        ),
    )

    # Validate input shape
    if X.ndim != 2:
        raise ValueError(f"Expected 2D input (batch, seq_len), got shape {X.shape}")

    expected_length = X.shape[1]

    if expected_length < 1:
        raise ValueError(f"Invalid sequence length: {expected_length}")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies: list[float] = []
    fold_f1_macro: list[float] = []
    fold_f1_weighted: list[float] = []
    fold_precision: list[float] = []
    fold_recall: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Verify shapes are consistent
        assert X_train.shape[1] == expected_length
        assert X_val.shape[1] == expected_length

        # Create datasets and loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        dl_kwargs: dict[str, Any] = {
            "pin_memory": True,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2

        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **dl_kwargs,
        )
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **dl_kwargs,
        )

        # Create fresh model for this fold
        try:
            model = model_fn()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                fold_accuracies.append(0.0)
                fold_f1_macro.append(0.0)
                fold_f1_weighted.append(0.0)
                fold_precision.append(0.0)
                fold_recall.append(0.0)
                clear_cuda_memory()  # Use centralized memory cleanup
                continue
            raise

        optimizer = optimizer_fn(model)

        # Handle checkpoint directory
        checkpoint_dir: str | Path = "checkpoints"
        if save_checkpoints and checkpoint_base_dir is not None:
            checkpoint_dir = str(Path(checkpoint_base_dir) / f"fold_{fold + 1}")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_classes=num_classes,
            epochs=epochs,
            device=device.type,  # Pass device.type (str) not device object
            patience=patience,
            use_scheduler=use_scheduler,
            warmup_epochs=warmup_epochs,
            use_amp=use_amp,
            use_compile=use_compile,
            compile_mode=compile_mode,
            use_augmentation=use_augmentation,
            augmentation_params=augmentation_params,
            use_tta=use_tta,
            tta_augmentations=tta_augmentations,
            use_swa=use_swa,
            swa_start=swa_start,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=checkpoint_dir,
        )

        try:
            history = trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Skip this fold silently
                fold_accuracies.append(0.0)
                fold_f1_macro.append(0.0)
                fold_f1_weighted.append(0.0)
                fold_precision.append(0.0)
                fold_recall.append(0.0)
                clear_cuda_memory()  # Use centralized memory cleanup
                continue
            raise

        # Record ALL metrics
        best_metrics = history["best_metrics"]
        fold_accuracies.append(best_metrics["accuracy"])
        fold_f1_macro.append(best_metrics["f1_macro"])
        fold_f1_weighted.append(best_metrics["f1_weighted"])
        fold_precision.append(best_metrics["precision"])
        fold_recall.append(best_metrics["recall"])

        # Clean up after each fold
        clear_cuda_memory()

    # Aggregate results with ALL metrics
    if fold_accuracies:
        mean_acc = float(np.mean(fold_accuracies))
        std_acc = float(np.std(fold_accuracies))
        mean_f1_macro = float(np.mean(fold_f1_macro))
        std_f1_macro = float(np.std(fold_f1_macro))
        mean_f1_weighted = float(np.mean(fold_f1_weighted))
        std_f1_weighted = float(np.std(fold_f1_weighted))
        mean_precision = float(np.mean(fold_precision))
        std_precision = float(np.std(fold_precision))
        mean_recall = float(np.mean(fold_recall))
        std_recall = float(np.std(fold_recall))
    else:
        mean_acc = std_acc = mean_f1_macro = std_f1_macro = 0.0
        mean_f1_weighted = std_f1_weighted = 0.0
        mean_precision = std_precision = 0.0
        mean_recall = std_recall = 0.0

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1_macro": mean_f1_macro,
        "std_f1_macro": std_f1_macro,
        "mean_f1_weighted": mean_f1_weighted,
        "std_f1_weighted": std_f1_weighted,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "fold_accuracies": fold_accuracies,
        "fold_f1_macro": fold_f1_macro,
        "fold_f1_weighted": fold_f1_weighted,
        "fold_precision": fold_precision,
        "fold_recall": fold_recall,
    }
