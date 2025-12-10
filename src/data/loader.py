"""
Dataset loading and preprocessing for UCR time series classification datasets.

This module handles:
- Loading datasets via aeon library
- Z-score normalization
- Padding/truncation for variable-length sequences
- Train/test split management
"""

from typing import Any, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset wrapper for time series classification data.

    Handles both univariate and multivariate time series.
    """

    def __init__(self, X: NDArray[np.float64], y: NDArray[np.int64]):
        """
        Args:
            X: Time series data of shape (n_samples, n_channels, length) or (n_samples, length)
            y: Labels of shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class UCRDataLoader:
    """
    Loader for UCR time series classification datasets.

    Provides preprocessing including normalization and padding.
    """

    def __init__(
        self,
        dataset_name: str,
        normalize: bool = True,
        padding: str = "none",
        max_length: int | None = None,
    ):
        """
        Args:
            dataset_name: Name of the UCR dataset (e.g., "Adiac", "ArrowHead")
            normalize: Whether to apply z-score normalization (default: True)
            padding: Padding strategy - "none", "zero", "repeat" (default: "none")
            max_length: Maximum sequence length for padding/truncation (default: None)
        """
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.padding = padding
        self.max_length = max_length

        self.scaler: StandardScaler | None = None
        self.label_encoder = LabelEncoder()

        # Dataset metadata
        self.n_classes: int = 0
        self.n_channels: int = 0
        self.sequence_length: int = 0
        self.n_train: int = 0
        self.n_test: int = 0

    def load_data(
        self,
    ) -> Tuple[
        NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]
    ]:
        """
        Load and preprocess UCR dataset.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
            X shape: (n_samples, n_channels, sequence_length) or (n_samples, sequence_length)
            y shape: (n_samples,)
        """
        # Import here to avoid circular imports and handle dynamic loading
        from aeon.datasets import load_classification

        # Load dataset using aeon - handling variable return types
        train_data = load_classification(self.dataset_name, split="train")
        test_data = load_classification(self.dataset_name, split="test")

        # Extract X and y from tuple, ignoring metadata if present
        X_train_raw = train_data[0]
        y_train_raw = train_data[1]
        X_test_raw = test_data[0]
        y_test_raw = test_data[1]

        # Convert to numpy arrays with explicit types
        X_train: NDArray[np.float64] = np.asarray(X_train_raw, dtype=np.float64)
        X_test: NDArray[np.float64] = np.asarray(X_test_raw, dtype=np.float64)

        # Handle 2D (univariate) vs 3D (multivariate) data
        # aeon returns shape (n_samples, n_channels, length) for multivariate
        # or (n_samples, length) for univariate
        if X_train.ndim == 2:
            # Univariate: (n_samples, length)
            self.n_channels = 1
            self.sequence_length = X_train.shape[1]
        else:
            # Multivariate: (n_samples, n_channels, length)
            self.n_channels = X_train.shape[1]
            self.sequence_length = X_train.shape[2]

        # Encode labels to integers
        y_train_encoded = self.label_encoder.fit_transform(y_train_raw)
        y_test_encoded = self.label_encoder.transform(y_test_raw)

        # Convert to numpy arrays with explicit int type
        y_train: NDArray[np.int64] = np.asarray(y_train_encoded, dtype=np.int64)
        y_test: NDArray[np.int64] = np.asarray(y_test_encoded, dtype=np.int64)

        self.n_classes = len(self.label_encoder.classes_)
        self.n_train = len(X_train)
        self.n_test = len(X_test)

        # Apply padding/truncation if specified
        if self.max_length is not None:
            X_train = self._apply_padding(X_train, self.max_length)
            X_test = self._apply_padding(X_test, self.max_length)
            self.sequence_length = self.max_length

        # Apply normalization
        if self.normalize:
            X_train = self._normalize(X_train, fit=True)
            X_test = self._normalize(X_test, fit=False)

        return X_train, y_train, X_test, y_test

    def _normalize(
        self, X: NDArray[np.float64], fit: bool = False
    ) -> NDArray[np.float64]:
        """
        Apply z-score normalization to time series data.

        Args:
            X: Input data
            fit: Whether to fit the scaler (True for train, False for test)

        Returns:
            Normalized data with same shape as input
        """
        original_shape = X.shape

        # Reshape to 2D for StandardScaler: (n_samples, features)
        if X.ndim == 2:
            # Univariate: (n_samples, length) → already 2D
            X_reshaped = X
        else:
            # Multivariate: (n_samples, n_channels, length) → (n_samples, n_channels * length)
            X_reshaped = X.reshape(X.shape[0], -1)

        if fit:
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_normalized = self.scaler.transform(X_reshaped)

        # Reshape back to original shape
        return X_normalized.reshape(original_shape)

    def _apply_padding(
        self, X: NDArray[np.float64], target_length: int
    ) -> NDArray[np.float64]:
        """
        Apply padding or truncation to standardize sequence length.

        Args:
            X: Input data
            target_length: Target sequence length

        Returns:
            Padded/truncated data
        """
        if X.ndim == 2:
            # Univariate: (n_samples, length)
            current_length = X.shape[1]

            if current_length == target_length:
                return X
            elif current_length > target_length:
                # Truncate
                return X[:, :target_length]
            else:
                # Pad
                return self._pad_sequences(X, target_length)
        else:
            # Multivariate: (n_samples, n_channels, length)
            current_length = X.shape[2]

            if current_length == target_length:
                return X
            elif current_length > target_length:
                # Truncate
                return X[:, :, :target_length]
            else:
                # Pad
                return self._pad_sequences(X, target_length)

    def _pad_sequences(
        self, X: NDArray[np.float64], target_length: int
    ) -> NDArray[np.float64]:
        """
        Pad sequences to target length using specified strategy.

        Args:
            X: Input data
            target_length: Target sequence length

        Returns:
            Padded data
        """
        if self.padding == "zero":
            # Zero padding
            if X.ndim == 2:
                pad_width = ((0, 0), (0, target_length - X.shape[1]))
            else:
                pad_width = ((0, 0), (0, 0), (0, target_length - X.shape[2]))
            return np.pad(X, pad_width, mode="constant", constant_values=0)

        elif self.padding == "repeat":
            # Repeat last value
            if X.ndim == 2:
                pad_width = ((0, 0), (0, target_length - X.shape[1]))
            else:
                pad_width = ((0, 0), (0, 0), (0, target_length - X.shape[2]))
            return np.pad(X, pad_width, mode="edge")

        else:
            raise ValueError(f"Unknown padding strategy: {self.padding}")

    def get_dataloaders(
        self,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 0,
    ) -> Tuple[
        DataLoader[Tuple[torch.Tensor, torch.Tensor]],
        DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Create PyTorch DataLoaders for train and test sets.

        Args:
            batch_size: Batch size for training
            shuffle_train: Whether to shuffle training data
            num_workers: Number of worker processes for data loading

        Returns:
            Tuple of (train_loader, test_loader)
        """
        X_train, y_train, X_test, y_test = self.load_data()

        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, test_loader

    def get_dataset_info(self) -> dict[str, Any]:
        """
        Get dataset metadata.

        Returns:
            Dictionary with dataset information
        """
        return {
            "name": self.dataset_name,
            "n_classes": self.n_classes,
            "n_channels": self.n_channels,
            "sequence_length": self.sequence_length,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "class_names": self.label_encoder.classes_.tolist(),
        }


def load_ucr_dataset(
    dataset_name: str,
    batch_size: int = 32,
    normalize: bool = True,
    padding: str = "none",
    max_length: int | None = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    dict[str, Any],
]:
    """
    Convenience function to load UCR dataset with default settings.

    Args:
        dataset_name: Name of the UCR dataset
        batch_size: Batch size for DataLoaders
        normalize: Whether to apply z-score normalization
        padding: Padding strategy ("none", "zero", "repeat")
        max_length: Maximum sequence length
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, test_loader, dataset_info)

    Example:
        >>> train_loader, test_loader, info = load_ucr_dataset("Adiac", batch_size=64)
        >>> print(f"Dataset: {info['name']}, Classes: {info['n_classes']}")
        >>> for batch_X, batch_y in train_loader:
        >>>     # Training loop
        >>>     pass
    """
    loader = UCRDataLoader(
        dataset_name=dataset_name,
        normalize=normalize,
        padding=padding,
        max_length=max_length,
    )

    train_loader, test_loader = loader.get_dataloaders(
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
    )

    dataset_info = loader.get_dataset_info()

    return train_loader, test_loader, dataset_info
