"""K-fold cross-validation for more robust evaluation on small datasets."""

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.loader import TimeSeriesDataset
from src.training.trainer import Trainer


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
    device: str = "cuda",
    use_scheduler: bool = True,
    warmup_epochs: int = 5,
    use_amp: bool = True,
    use_compile: bool = True,
    compile_mode: str = "default",
    use_augmentation: bool = True,
    augmentation_params: dict[str, Any] | None = None,
    num_workers: int = 0,
    use_tta: bool = False,
    tta_augmentations: int = 5,
    use_swa: bool = False,
    swa_start: int = 60,
    save_checkpoints: bool = True,
    **trainer_kwargs: Any,
) -> dict[str, float]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model_fn: Function that returns a new model instance
        X: Full dataset features
        y: Full dataset labels
        optimizer_fn: Function that creates optimizer given model
        criterion: Loss function
        num_classes: Number of classes
        k_folds: Number of folds
        batch_size: Batch size for training
        epochs: Number of epochs per fold
        patience: Early stopping patience
        device: Device to use
        use_scheduler: Whether to use learning rate scheduler
        warmup_epochs: Number of warmup epochs
        use_amp: Whether to use automatic mixed precision
        use_compile: Whether to use TorchScript compilation
        compile_mode: Compilation mode for TorchScript
        use_augmentation: Whether to use data augmentation
        augmentation_params: Augmentation parameters
        num_workers: Number of workers for data loading
        use_tta: Whether to use test-time augmentation
        tta_augmentations: Number of test-time augmentations
        use_swa: Whether to use Stochastic Weight Averaging
        swa_start: Epoch to start SWA
        save_checkpoints: Whether to save model checkpoints
        **trainer_kwargs: Additional arguments for Trainer

    Returns:
        Dictionary with mean and std of metrics
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies: list[float] = []
    fold_f1_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔄 Fold {fold + 1}/{k_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        # Create new model and optimizer for this fold
        model = model_fn()
        optimizer = optimizer_fn(model)

        # Create trainer with full config
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_classes=num_classes,
            epochs=epochs,
            patience=patience,
            device=device,
            checkpoint_dir=f"checkpoints/fold_{fold + 1}",
            use_scheduler=use_scheduler,
            warmup_epochs=warmup_epochs,
            use_amp=use_amp,
            use_compile=use_compile,
            compile_mode=compile_mode,
            use_augmentation=use_augmentation,
            augmentation_params=augmentation_params or {},
            use_tta=use_tta,
            tta_augmentations=tta_augmentations,
            use_swa=use_swa,
            swa_start=swa_start,
            save_checkpoints=save_checkpoints,
            **trainer_kwargs,
        )

        # Train and evaluate on this fold
        results = trainer.train()

        val_accuracy = results["best_val_accuracy"]
        val_f1 = results["final_metrics"]["f1_macro"]

        fold_accuracies.append(val_accuracy)
        fold_f1_scores.append(val_f1)

        print(f"   Fold {fold + 1} → Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        # Clean up
        del model
        del optimizer
        del trainer
        del train_loader
        del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_f1": float(np.mean(fold_f1_scores)),
        "std_f1": float(np.std(fold_f1_scores)),
    }
