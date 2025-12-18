"""
K-fold stratified cross-validation for small datasets.
"""

import gc
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
from src.utils.system import get_safe_num_workers


def cross_validate_dataset(
    model_fn: Callable[[], nn.Module],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
    criterion: nn.Module,
    num_classes: int,
    k_folds: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10,
    hardware_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    save_checkpoints: bool = True,
    **trainer_kwargs: Any,
) -> dict[str, Any]:
    """
    Perform k-fold stratified cross-validation.

    CRITICAL: Verifies consistent sequence length before CV split
    """
    # Default configs
    hardware_config = hardware_config or {}
    training_config = training_config or {}

    # Extract hardware settings
    device = hardware_config.get("device", "cuda")
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

    # Ensure X has consistent sequence length BEFORE splitting
    if X.ndim != 2:
        raise ValueError(f"Expected 2D input (batch, seq_len), got shape {X.shape}")

    expected_length = X.shape[1]
    print(f"Cross-validation dataset shape: {X.shape}")
    print(f"  Samples: {X.shape[0]}, Sequence length: {expected_length}")

    if expected_length < 1:
        raise ValueError(f"Invalid sequence length: {expected_length}")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies: list[float] = []
    fold_f1_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'=' * 60}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Verify shapes are consistent (should always pass now)
        assert X_train.shape[1] == expected_length, (
            f"Train fold shape mismatch: {X_train.shape[1]} vs {expected_length}"
        )
        assert X_val.shape[1] == expected_length, f"Val fold shape mismatch: {X_val.shape[1]} vs {expected_length}"

        print(f"  Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

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
                print(f"❌ OOM during model creation for fold {fold + 1}: {e}")
                # Skip this fold
                fold_accuracies.append(0.0)
                fold_f1_scores.append(0.0)
                continue
            raise

        optimizer = optimizer_fn(model)

        # Trainer setup
        checkpoint_path: str | None = None
        if save_checkpoints:
            checkpoint_path = str(Path(f"checkpoints/fold_{fold + 1}"))

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_classes=num_classes,
            epochs=epochs,
            device=device,
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
            checkpoint_dir=checkpoint_path if checkpoint_path else "checkpoints",
            **trainer_kwargs,
        )

        # Train this fold
        try:
            history = trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM during training for fold {fold + 1}: {e}")
                # Skip this fold
                fold_accuracies.append(0.0)
                fold_f1_scores.append(0.0)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                continue
            raise

        # Record metrics
        best_metrics = history["best_metrics"]
        fold_accuracies.append(best_metrics["accuracy"])
        fold_f1_scores.append(best_metrics["f1_macro"])

        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {best_metrics['f1_macro']:.4f}")

    # Aggregate results
    if fold_accuracies:
        mean_acc = float(np.mean(fold_accuracies))
        std_acc = float(np.std(fold_accuracies))
        mean_f1 = float(np.mean(fold_f1_scores))
        std_f1 = float(np.std(fold_f1_scores))
    else:
        mean_acc = std_acc = mean_f1 = std_f1 = 0.0

    print(f"\n{'=' * 60}")
    print(f"Cross-Validation Results ({k_folds} folds)")
    print(f"{'=' * 60}")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1 (macro): {mean_f1:.4f} ± {std_f1:.4f}")

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1_macro": mean_f1,
        "std_f1_macro": std_f1,
        "fold_accuracies": fold_accuracies,
        "fold_f1_scores": fold_f1_scores,
    }
