"""
UCR dataset loading and preprocessing with multiprocessing safety.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aeon.datasets import load_classification
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from src.utils.system import get_safe_num_workers, recommend_num_workers


class UCRDataLoader:
    """Handles UCR dataset loading and preprocessing."""

    CACHE_DIR = Path("data/cache")

    def __init__(
        self,
        dataset_name: str,
        normalize: bool = True,
        padding: str = "zero",
        max_length: int | None = None,
        cache_dir: str | Path = "./data/cache",
    ):
        """
        Args:
            dataset_name: UCR dataset name
            normalize: Apply z-score normalization
            padding: Padding strategy ('zero', 'edge', 'wrap')
            max_length: Maximum sequence length (truncate if longer)
            cache_dir: Directory for caching downloaded datasets
        """
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.padding = padding
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_data(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
        """Load and preprocess UCR dataset."""
        # Load raw data - explicitly unpack 2-tuple, ignore optional metadata
        X_train_raw, y_train_raw = load_classification(self.dataset_name, split="train", return_metadata=False)[:2]
        X_test_raw, y_test_raw = load_classification(self.dataset_name, split="test", return_metadata=False)[:2]

        # Convert to numpy arrays with explicit types
        X_train: NDArray[np.float64] = np.array(X_train_raw, dtype=np.float64)
        X_test: NDArray[np.float64] = np.array(X_test_raw, dtype=np.float64)
        y_train: NDArray[np.int64] = np.array(y_train_raw, dtype=np.int64)
        y_test: NDArray[np.int64] = np.array(y_test_raw, dtype=np.int64)

        # Handle multivariate: flatten or average channels
        if X_train.ndim == 3:  # (n_samples, n_channels, seq_len)
            X_train = X_train.mean(axis=1)
            X_test = X_test.mean(axis=1)

        # Ensure 2D
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # **CRITICAL FIX**: Determine target length BEFORE any truncation/padding
        # This ensures train and test have identical sequence length
        train_length = X_train.shape[1]
        test_length = X_test.shape[1]

        # Use the maximum length from both splits as baseline
        baseline_length = max(train_length, test_length)

        # Apply max_length constraint if specified
        if self.max_length is not None:
            target_length = min(baseline_length, self.max_length)
        else:
            target_length = baseline_length

        # Truncate BOTH to target length (in same order to preserve consistency)
        X_train = X_train[:, :target_length]
        X_test = X_test[:, :target_length]

        # Pad BOTH to exact target length using the SAME strategy
        X_train = self._pad_sequences(X_train, target_length)
        X_test = self._pad_sequences(X_test, target_length)

        # Verify shapes match after preprocessing
        assert X_train.shape[1] == X_test.shape[1], (
            f"Shape mismatch after preprocessing: train={X_train.shape[1]}, test={X_test.shape[1]}"
        )

        # Normalize using combined statistics (better for cross-validation)
        if self.normalize:
            X_train, X_test = self._normalize(X_train, X_test)

        # Encode labels to 0-based indices
        y_train_encoded = self._encode_labels(y_train)
        y_test_encoded = self._encode_labels(y_test)

        return X_train, y_train_encoded, X_test, y_test_encoded

    def _pad_sequences(self, X: NDArray[np.float64], target_length: int) -> NDArray[np.float64]:
        """Pad sequences to target length."""
        current_length = X.shape[1]
        if current_length >= target_length:
            return X

        pad_width = target_length - current_length

        if self.padding == "zero":
            padding = np.zeros((X.shape[0], pad_width))
        elif self.padding == "edge":
            padding = np.tile(X[:, -1:], (1, pad_width))
        elif self.padding == "wrap":
            repeats = (pad_width // current_length) + 1
            padding = np.tile(X, (1, repeats))[:, :pad_width]
        else:
            raise ValueError(f"Unknown padding mode: {self.padding}")

        return np.concatenate([X, padding], axis=1)

    def _normalize(
        self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Z-score normalization using training statistics."""
        mean = X_train.mean()
        std = X_train.std()
        std = std if std > 0 else 1.0

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        return X_train, X_test

    def _encode_labels(self, y: NDArray[Any]) -> NDArray[np.int64]:
        """Encode string labels to integers."""
        unique_labels = np.unique(y)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return np.array([label_to_idx[label] for label in y], dtype=np.int64)

    def load_data_for_cross_validation(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and preprocess UCR dataset, merging train/test for CV.

        Returns:
            Tuple of (X_combined, y_combined) with guaranteed consistent sequence length
        """
        cache_file = self.CACHE_DIR / f"{self.dataset_name}_combined.pkl"

        # Return cached if exists
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                # Handle both old (4-tuple) and new (2-tuple) cache formats
                if len(cached_data) == 4:
                    X_train, y_train, X_test, y_test = cached_data
                    X_combined = np.vstack([X_train, X_test])
                    y_combined = np.concatenate([y_train, y_test])
                    return X_combined, y_combined
                else:
                    return cached_data

        # Load fresh, then cache
        X_train, y_train, X_test, y_test = self.load_data()

        # Combine train and test
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test])

        # Cache the combined data (not the separate splits)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((X_combined, y_combined), f)

        return X_combined, y_combined


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for time series."""

    def __init__(self, X: NDArray[np.float64], y: NDArray[np.int64]):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataset_info(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
) -> dict[str, int]:
    """Extract dataset metadata."""
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(np.unique(y_train)),
        "sequence_length": X_train.shape[1],
    }


def load_ucr_dataset(
    dataset_name: str,
    batch_size: int = 32,
    normalize: bool = True,
    padding: str = "zero",
    max_length: int | None = None,
    num_workers: int = 0,
    max_cpu_load: float = 0.5,
    auto_workers: bool = False,
    max_workers_override: int | None = None,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    dict[str, int],
]:
    """
    End-to-end dataset loading with DataLoader creation.

    Args:
        dataset_name: UCR dataset name
        batch_size: Training batch size
        normalize: Apply z-score normalization
        padding: Padding strategy
        max_length: Maximum sequence length
        num_workers: DataLoader workers (0 for main process)
        max_cpu_load: Maximum CPU utilization for auto worker selection (0.0-1.0)
        auto_workers: Automatically select optimal num_workers based on dataset
        max_workers_override: Hard cap on workers (overrides safety limits)

    Returns:
        Tuple of (train_loader, test_loader, dataset_info)
    """
    loader = UCRDataLoader(
        dataset_name=dataset_name,
        normalize=normalize,
        padding=padding,
        max_length=max_length,
    )

    X_train, y_train, X_test, y_test = loader.load_data()
    dataset_info = get_dataset_info(X_train, y_train, X_test, y_test)

    # Automatic worker selection based on dataset characteristics
    if auto_workers:
        recommendation = recommend_num_workers(
            batch_size=batch_size,
            sequence_length=dataset_info["sequence_length"],
            dataset_size=dataset_info["n_train"],
            max_cpu_load=max_cpu_load,
        )
        num_workers = recommendation["recommended_workers"]
    else:
        # Validate and clamp manual num_workers
        max_safe = get_safe_num_workers(
            max_cpu_load=max_cpu_load,
            max_workers=max_workers_override,
        )
        if num_workers > max_safe:
            print(
                f"⚠️  Requested num_workers={num_workers} exceeds safe limit ({max_safe})\n"
                f"   Clamping to {max_safe} to prevent system overload"
            )
            num_workers = max_safe

    # Create PyTorch datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # DataLoaders with optional multiprocessing
    common_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, test_loader, dataset_info
