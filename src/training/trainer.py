"""
Main training loop and orchestration.

Handles:
- Epoch training and validation
- Metrics tracking
- Early stopping
- Checkpoint management
- Device management (GPU/CPU)
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import TimeSeriesAugmentation
from src.training.metrics import MetricsTracker
from src.training.tta import TestTimeAugmentation


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_precision: float
    val_recall: float


class Trainer:
    """
    Main training orchestrator for time series classification.

    Features:
    - Mixed precision training (AMP)
    - Learning rate warmup + cosine annealing
    - Early stopping with patience
    - Optional: torch.compile, TTA, SWA, augmentation
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_classes: int,
        epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        use_scheduler: bool = True,
        warmup_epochs: int = 5,
        use_amp: bool = True,
        use_compile: bool = True,
        compile_mode: str = "default",
        use_augmentation: bool = True,
        augmentation_params: dict[str, Any] | None = None,
        use_tta: bool = False,
        tta_augmentations: int = 5,
        use_swa: bool = False,
        swa_start: int = 60,
        save_checkpoints: bool = True,
    ):
        # Core components
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_classes = num_classes
        self.epochs = epochs
        self.patience = patience

        # Setup checkpoint directory - convert to Path for consistency
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device setup (CUDA required)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required but not available. "
                "Please ensure PyTorch with CUDA is installed: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130"
            )
        if device is None:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device)
            if "cpu" in str(self.device):
                raise RuntimeError("CPU device specified, but CUDA is required")

        self.model.to(self.device)

        # Training features
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.use_augmentation = use_augmentation
        self.use_tta = use_tta
        self.use_swa = use_swa
        self.swa_start = swa_start
        self.save_checkpoints = save_checkpoints

        # Mixed precision scaler - use modern API
        self.scaler: GradScaler | None = None
        if use_amp:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            self.scaler = GradScaler(device_type)

        # Learning rate scheduler
        self.scheduler: CosineAnnealingLR | None = None
        if use_scheduler:
            self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

        # Data augmentation
        self.augmentation: TimeSeriesAugmentation | None = None
        if use_augmentation:
            self.augmentation = TimeSeriesAugmentation(**augmentation_params or {})

        # Test-time augmentation
        self.tta: TestTimeAugmentation | None = None
        if use_tta:
            self.tta = TestTimeAugmentation(model, n_augmentations=tta_augmentations)

        # Early stopping state
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state: dict[str, Any] | None = None

        # Track training metrics (lightweight)
        self.train_metrics = MetricsTracker(num_classes, compute_confusion=False)

        # Track validation metrics (full)
        self.val_metrics = MetricsTracker(num_classes, compute_confusion=True)

        # Optimize dataloaders for repeated iteration
        if hasattr(train_loader, "num_workers") and train_loader.num_workers > 0:
            train_loader.persistent_workers = True
            test_loader.persistent_workers = True

    # Optimize DataLoader creation
    def _create_optimized_dataloaders(
        self, train_loader: DataLoader, test_loader: DataLoader
    ) -> tuple[DataLoader, DataLoader]:
        """
        Wrap dataloaders with persistent_workers for speed.
        Reduces overhead on repeated epoch iterations.
        """
        # Only enable if num_workers > 0 to avoid issues with single-process
        if hasattr(train_loader, "num_workers") and train_loader.num_workers > 0:
            train_loader.persistent_workers = True
            test_loader.persistent_workers = True

        return train_loader, test_loader

    def train(self) -> dict[str, Any]:
        """
        Main training loop with early stopping.

        Returns:
            Dictionary with training history and best metrics
        """
        history = {
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "val_f1_scores": [],
            "best_metrics": {},
            "best_epoch": 0,
        }

        for epoch in range(self.epochs):
            # Warmup phase
            if self.use_scheduler and epoch < self.warmup_epochs:
                lr = 1e-6 + (self.optimizer.defaults["lr"] - 1e-6) * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            # Training epoch
            train_loss, train_accuracy = self.train_epoch()
            history["train_losses"].append(train_loss)
            history["train_accuracies"].append(train_accuracy)

            # Validation epoch
            val_loss, val_metrics = self.validate_epoch()
            history["val_losses"].append(val_loss)
            history["val_accuracies"].append(val_metrics["accuracy"])
            history["val_f1_scores"].append(val_metrics["f1_macro"])

            # Learning rate step (after warmup)
            if self.use_scheduler and epoch >= self.warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.best_model_state = deepcopy(self.model.state_dict())
            else:
                self.patience_counter += 1

            # Progress
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1_macro']:.4f}"
            )

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Get best metrics
        best_val_metrics = self.validate_epoch()[1]

        history["best_metrics"] = best_val_metrics
        history["best_epoch"] = self.best_epoch

        return history

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0

        # Accumulate gradients to simulate larger batch without memory cost
        accumulation_steps = 1  # Set to 2-4 if OOM and model is large

        # Convert device to string type for autocast
        device_type = self.device.type  # "cuda" or "cpu"

        for batch_idx, (x, y) in enumerate(tqdm(self.train_loader, desc="Training")):
            x, y = x.to(self.device), y.to(self.device)

            # Data augmentation
            if self.augmentation and x.dim() == 2:
                x = x.unsqueeze(1)
                x = self.augmentation(x)
                x = x.squeeze(1)

            # Forward pass
            with torch.autocast(device_type, enabled=self.use_amp):
                outputs = self.model(x)
                loss = self.criterion(outputs, y) / accumulation_steps

            # Backward with proper scaler handling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            self.train_metrics.update(outputs.detach(), y.detach())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = self.train_metrics.get_accuracy()

        return avg_loss, accuracy

    def validate_epoch(self) -> tuple[float, dict[str, Any]]:
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0

        # Convert device to string type for autocast
        device_type = self.device.type  # "cuda" or "cpu"

        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Validating"):
                x, y = x.to(self.device), y.to(self.device)

                with torch.autocast(device_type, enabled=self.use_amp):
                    if self.use_tta and self.tta is not None:
                        outputs = self.tta.predict(x)
                    else:
                        outputs = self.model(x)

                    loss = self.criterion(outputs, y)

                total_loss += loss.item()
                self.val_metrics.update(outputs, y)

        avg_loss = total_loss / len(self.test_loader)
        metrics = self.val_metrics.compute()

        return avg_loss, metrics
