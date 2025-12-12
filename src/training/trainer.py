"""
Training orchestration for time series classification models.

Handles:
- Training loop with progress bars
- Validation and metric computation
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Device management (GPU/CPU)
"""

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import TimeSeriesAugmentation
from src.training.metrics import EpochMetrics, MetricsTracker
from src.training.tta import TestTimeAugmentation


def get_gpu_info() -> dict[str, Any]:
    """
    Detect GPU type and capabilities for torch.compile support.

    Returns:
        Dict with gpu_name, supports_compile, vram_gb
    """
    if not torch.cuda.is_available():
        return {"gpu_name": "CPU", "supports_compile": False, "vram_gb": 0}

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # torch.compile is supported on Linux only
    is_linux = sys.platform != "win32"
    supports_compile = is_linux

    return {
        "gpu_name": gpu_name,
        "supports_compile": supports_compile,
        "vram_gb": vram_gb,
        "is_linux": is_linux,
    }


class Trainer:
    """
    Main training orchestrator for time series classification.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: Optimizer,
        criterion: nn.Module,
        num_classes: int,
        epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        checkpoint_dir: str = "checkpoints",
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
        """
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            test_loader: DataLoader for test/validation data
            optimizer: Optimizer instance
            criterion: Loss function
            num_classes: Number of classification classes
            epochs: Maximum number of training epochs
            patience: Early stopping patience (epochs without improvement)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            checkpoint_dir: Directory to save model checkpoints
            use_scheduler: Whether to use learning rate scheduler
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_classes = num_classes
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        # Always use CUDA (enforce requirement)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required but not available. "
                "Please ensure PyTorch with CUDA is installed: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
        if device is None:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device)
            if "cpu" in str(self.device):
                raise RuntimeError("CPU device specified, but CUDA is required")

        self.model.to(self.device)

        # Compile model for faster execution (PyTorch 2.0+)
        gpu_info = get_gpu_info()
        if use_compile and hasattr(torch, "compile"):
            if gpu_info["supports_compile"]:
                print(f"📦 GPU detected: {gpu_info['gpu_name']} ({gpu_info['vram_gb']:.1f}GB VRAM)")
                print("✅ torch.compile supported on Linux")
                print(f"   Compiling with mode='{compile_mode}'...")
                try:
                    self.model = cast(nn.Module, torch.compile(self.model, mode=compile_mode))
                    print("✓ Model compiled successfully\n")
                except RuntimeError as e:
                    print(f"⚠️  torch.compile failed, falling back to eager mode: {e}\n")
            else:
                print(f"📦 GPU detected: {gpu_info['gpu_name']} ({gpu_info['vram_gb']:.1f}GB VRAM)")
                print(f"⚠️  torch.compile not supported on {sys.platform}. Using eager mode.\n")
        else:
            if gpu_info["vram_gb"] < 2:
                print(f"📦 GPU detected: {gpu_info['gpu_name']} ({gpu_info['vram_gb']:.1f}GB VRAM)")
                print(f"⚠️  Skipping torch.compile due to very limited VRAM\n")

        # Improved learning rate scheduler with warmup
        self.scheduler = None
        if use_scheduler:
            # Warmup scheduler
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

            # Cosine annealing after warmup
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)

            # Combine warmup + cosine annealing
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
            )

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.history: list[EpochMetrics] = []

        self.use_amp = use_amp and torch.cuda.is_available()

        # Initialize gradient scaler for mixed precision
        # Always use CUDA device for scaler (since CUDA is required)
        self.scaler: GradScaler | None = GradScaler(device=str(self.device)) if self.use_amp else None

        # Time series augmentation
        self.augment: TimeSeriesAugmentation | None = None
        if use_augmentation:
            params = augmentation_params or {}
            self.augment = TimeSeriesAugmentation(**params)

        # Test Time Augmentation settings
        self.use_tta = use_tta
        self.tta_augmentations = tta_augmentations

        # Stochastic Weight Averaging
        self.use_swa = use_swa
        self.swa_start = swa_start
        self.swa_model: AveragedModel | None = None
        self.swa_scheduler: SWALR | None = None

        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.0001)

        # Checkpointing settings
        self.save_checkpoints = save_checkpoints
        self.best_state: dict[str, Any] | None = None

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch with optional mixed precision."""
        self.model.train()
        total_loss = 0.0
        metrics_tracker = MetricsTracker(self.num_classes)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]",
            leave=False,
        )

        for batch_X, batch_y in progress_bar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            # Apply augmentation on training batches (expects [B,C,L])
            augment = self.augment
            if augment is not None:
                if batch_X.dim() == 2:
                    batch_X = batch_X.unsqueeze(1)
                # Ensure shape (B,C,L)
                if batch_X.dim() == 3 and batch_X.shape[1] != 1 and batch_X.shape[1] != batch_X.shape[-1]:
                    pass
                batch_X = augment(batch_X)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.use_amp and self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                self.scaler.scale(loss).backward()

                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

            total_loss += loss.item()
            metrics_tracker.update(outputs.detach(), batch_y)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = metrics_tracker.get_accuracy()

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Validate on test set with optional TTA."""
        self.model.eval()
        total_loss = 0.0
        metrics_tracker = MetricsTracker(self.num_classes)

        # Create TTA wrapper if enabled
        tta_model = TestTimeAugmentation(self.model, self.tta_augmentations) if self.use_tta else None
        if tta_model is not None:
            print(f"🔄 Using TTA with {self.tta_augmentations} augmentations")

        progress_bar = tqdm(
            self.test_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Val{'+ TTA' if self.use_tta else ''}]",
            leave=False,
        )

        for batch_X, batch_y in progress_bar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Use TTA or standard inference
            if tta_model is not None:
                outputs = tta_model.predict(batch_X)
                loss = self.criterion(outputs, batch_y)
            elif self.use_amp and self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
            else:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

            total_loss += loss.item()
            metrics_tracker.update(outputs, batch_y)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.test_loader)
        all_metrics = metrics_tracker.compute()

        return {
            "loss": avg_loss,
            "accuracy": all_metrics["accuracy"],
            "f1_macro": all_metrics["f1_macro"],
            "f1_weighted": all_metrics["f1_weighted"],
            "precision": all_metrics["precision"],
            "recall": all_metrics["recall"],
            "confusion_matrix": all_metrics["confusion_matrix"].tolist(),
        }

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "history": [m.to_dict() for m in self.history],
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def train(self) -> dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dictionary with final metrics and training history
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_accuracy = self.train_epoch()

            # Update SWA model after swa_start
            if self.use_swa and epoch >= self.swa_start and self.swa_model is not None:
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()

            # Validate
            val_metrics = self.validate()

            # Track metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                val_f1=val_metrics["f1_macro"],
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
            )
            self.history.append(epoch_metrics)

            # Learning rate scheduling (update at end of epoch)
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                if epoch % 10 == 0:
                    print(f"Current LR: {current_lr:.6f}")

            # Print epoch summary
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            # Check for improvement
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
                if self.save_checkpoints:
                    self.save_checkpoint("latest.pt", is_best=True)
                else:
                    self.best_state = deepcopy(self.model.state_dict())
                print(f"✓ New best model! Accuracy: {self.best_val_accuracy:.4f}")
            else:
                self.epochs_without_improvement += 1
                if self.save_checkpoints:
                    self.save_checkpoint("latest.pt", is_best=False)

        # Load best model for final evaluation
        if self.save_checkpoints:
            self.load_checkpoint("best_model.pt")
        elif self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # Use SWA model for final evaluation if enabled
        if self.use_swa and self.swa_model is not None:
            # Update batch norm statistics for SWA model
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            # Use SWA model for final metrics
            original_model = self.model
            self.model = self.swa_model.module
            final_metrics = self.validate()
            self.model = original_model  # Restore for checkpoint saving
        else:
            final_metrics = self.validate()

        return {
            "best_val_accuracy": self.best_val_accuracy,
            "final_metrics": final_metrics,
            "history": [m.to_dict() for m in self.history],
            "total_epochs": self.current_epoch + 1,
        }

    def save_training_history(self, filepath: str) -> None:
        """
        Save training history to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        history_data = {
            "epochs": [m.to_dict() for m in self.history],
            "best_val_accuracy": self.best_val_accuracy,
            "total_epochs": self.current_epoch + 1,
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)
