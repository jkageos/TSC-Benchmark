"""
Main benchmarking script for time series classification.

Coordinates training and evaluation across multiple models and datasets.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import load_ucr_dataset
from src.training.trainer import Trainer
from src.utils.config import (
    create_model,
    create_optimizer,
    get_model_config,
    get_training_config,
    load_config,
    validate_config,
)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_cuda_memory() -> None:
    """
    Clear CUDA cache to free up memory between runs.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        import gc

        gc.collect()


def get_available_memory() -> float:
    """
    Get available CUDA memory in GB.

    Returns:
        Available memory in GB
    """
    # CUDA is required, so this will always be available
    return torch.cuda.mem_get_info()[0] / 1e9


def get_optimal_batch_size(
    model_name: str,
    sequence_length: int,
    input_channels: int,
    num_classes: int,
    base_batch_size: int = 32,
) -> int:
    """
    Calculate optimal batch size based on sequence length and model type.

    Args:
        model_name: Name of the model
        sequence_length: Length of time series
        input_channels: Number of input channels
        num_classes: Number of classes
        base_batch_size: Base batch size from config

    Returns:
        Optimal batch size
    """
    # Memory usage estimates for different architectures

    if model_name in ["transformer", "cats", "autoformer"]:
        # Transformers are memory intensive (quadratic attention)
        if sequence_length > 2000:
            return max(2, base_batch_size // 16)
        elif sequence_length > 1500:
            return max(4, base_batch_size // 8)
        elif sequence_length > 1000:
            return max(8, base_batch_size // 4)
        elif sequence_length > 500:
            return max(16, base_batch_size // 2)
        else:
            return base_batch_size

    elif model_name == "patchtst":
        # PatchTST reduces sequence length via patching, less memory intensive
        if sequence_length > 2000:
            return max(4, base_batch_size // 8)
        elif sequence_length > 1000:
            return max(8, base_batch_size // 4)
        else:
            return base_batch_size

    elif model_name == "cnn":
        # CNNs are moderately memory intensive
        if sequence_length > 2000:
            return max(4, base_batch_size // 8)
        elif sequence_length > 1000:
            return max(8, base_batch_size // 4)
        else:
            return base_batch_size

    else:  # FCN
        # FCNs are less memory intensive
        if sequence_length > 5000:
            return max(8, base_batch_size // 4)
        else:
            return base_batch_size


def benchmark_model_on_dataset(
    model_name: str,
    dataset_name: str,
    config: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    """
    Benchmark a single model on a single dataset.
    Uses k-fold cross-validation for small datasets (n_train < 300).
    """
    # Clear CUDA memory before starting
    clear_cuda_memory()

    # Dataset-specific overrides (no skipping)
    dataset_overrides = config.get("dataset_overrides", {}).get(dataset_name, {})

    print(f"\n{'=' * 80}")
    print(f"Benchmarking {model_name.upper()} on {dataset_name}")
    print(f"{'=' * 80}\n")

    # Load dataset with overrides
    training_config = get_training_config(config)

    if training_config.get("device") == "cpu":
        training_config["device"] = "cuda"
    elif "device" not in training_config:
        training_config["device"] = "cuda"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Please install PyTorch with CUDA support.")

    # Apply dataset-specific overrides
    max_length = dataset_overrides.get("max_length", training_config["max_length"])
    batch_size = dataset_overrides.get("batch_size", training_config["batch_size"])
    epochs = dataset_overrides.get("epochs", training_config["epochs"])
    patience = dataset_overrides.get("patience", training_config["patience"])
    num_workers = dataset_overrides.get("num_workers", training_config.get("num_workers", 0))

    if dataset_overrides:
        print(f"📋 Applying dataset-specific overrides for {dataset_name}:")
        if max_length != training_config["max_length"]:
            print(f"   max_length: {training_config['max_length']} → {max_length}")
        if batch_size != training_config["batch_size"]:
            print(f"   batch_size: {training_config['batch_size']} → {batch_size}")
        if epochs != training_config["epochs"]:
            print(f"   epochs: {training_config['epochs']} → {epochs}")
        if patience != training_config["patience"]:
            print(f"   patience: {training_config['patience']} → {patience}")
        print()

    try:
        train_loader, test_loader, dataset_info = load_ucr_dataset(
            dataset_name=dataset_name,
            batch_size=batch_size,
            normalize=training_config["normalize"],
            padding=training_config["padding"],
            max_length=max_length,
            num_workers=num_workers,
        )
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        clear_cuda_memory()
        return {"error": str(e)}

    print(f"Dataset Info:")
    print(f"  Classes: {dataset_info['n_classes']}")
    print(f"  Channels: {dataset_info['n_channels']}")
    print(f"  Sequence Length: {dataset_info['sequence_length']}")
    print(f"  Train Samples: {dataset_info['n_train']}")
    print(f"  Test Samples: {dataset_info['n_test']}\n")

    # Determine if cross-validation should be used
    use_cross_val = dataset_info["n_train"] < 300
    if use_cross_val:
        print(f"📊 Small dataset detected ({dataset_info['n_train']} train samples)")
        print(f"   Using 5-fold stratified cross-validation\n")

        result = benchmark_model_with_cross_validation(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            dataset_info=dataset_info,
            training_config=training_config,
            dataset_overrides=dataset_overrides,
            k_folds=dataset_overrides.get("cv_folds", training_config.get("cv_folds", 3)),
        )
    else:
        # Standard train/test split
        print(f"📈 Larger dataset ({dataset_info['n_train']} train samples)")
        print(f"   Using standard train/test split\n")

        adjusted_batch_size = get_optimal_batch_size(
            model_name=model_name,
            sequence_length=dataset_info["sequence_length"],
            input_channels=dataset_info["n_channels"],
            num_classes=dataset_info["n_classes"],
            base_batch_size=batch_size,
        )

        if adjusted_batch_size != batch_size:
            print(f"⚠️  Adjusting batch size for memory optimization")
            print(f"   Original: {batch_size} → Adjusted: {adjusted_batch_size}\n")

            train_loader, test_loader, dataset_info = load_ucr_dataset(
                dataset_name=dataset_name,
                batch_size=adjusted_batch_size,
                normalize=training_config["normalize"],
                padding=training_config["padding"],
                max_length=max_length,
                num_workers=num_workers,
            )

        result = benchmark_model_standard(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            dataset_info=dataset_info,
            training_config=training_config,
            train_loader=train_loader,
            test_loader=test_loader,
            dataset_overrides=dataset_overrides,
            results_dir=results_dir,
        )

    # Clean up memory
    clear_cuda_memory()
    return result


def benchmark_model_standard(
    model_name: str,
    dataset_name: str,
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    training_config: dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader,
    dataset_overrides: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    """
    Standard benchmark with train/test split (for larger datasets).
    """
    batch_size = train_loader.batch_size or 32
    epochs = dataset_overrides.get("epochs", training_config["epochs"])
    patience = dataset_overrides.get("patience", training_config["patience"])

    # Create model
    model_config = get_model_config(config, model_name)
    dataset_model_overrides = dataset_overrides.get("models", {}).get(model_name, {})

    if dataset_model_overrides:
        print(f"📋 Applying model-specific overrides for {model_name} on {dataset_name}:")
        for key, value in dataset_model_overrides.items():
            old_value = model_config.get(key, "not set")
            model_config[key] = value
            print(f"   {key}: {old_value} → {value}")
        print()

    model = create_model(
        model_name=model_name,
        model_config=model_config,
        num_classes=dataset_info["n_classes"],
        input_length=dataset_info["sequence_length"],
        input_channels=dataset_info["n_channels"],
    )

    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_name=training_config["optimizer"],
        learning_rate=training_config["learning_rate"],
        **training_config.get("optimizer_params", {}),
    )

    # Loss function with class weights
    n_train = dataset_info["n_train"]
    if n_train > 1000:
        label_smoothing = 0.1
    elif n_train > 200:
        label_smoothing = 0.05
    else:
        label_smoothing = 0.0

    class_counts = torch.zeros(dataset_info["n_classes"], dtype=torch.long)
    for _, batch_y in train_loader:
        for c in batch_y.unique():
            class_counts[c] += (batch_y == c).sum()

    ratio = class_counts.max().float() / class_counts.clamp(min=1).float()
    class_weights = ratio.to("cuda")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Create checkpoint directory
    checkpoint_dir = results_dir / "checkpoints" / f"{model_name}_{dataset_name}"

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=dataset_info["n_classes"],
        epochs=epochs,
        patience=patience,
        device=training_config["device"],
        checkpoint_dir=str(checkpoint_dir),
        use_scheduler=training_config["use_scheduler"],
        warmup_epochs=training_config["warmup_epochs"],
        use_amp=training_config.get("use_amp", True),
        use_compile=training_config.get("use_compile", True),
        compile_mode=config.get("hardware", {}).get("compile_mode", "default"),
        use_augmentation=config["training"].get("use_augmentation", True),
        augmentation_params=config["training"].get("augmentation_params", {}),
        use_tta=training_config.get("use_tta", False),
        tta_augmentations=training_config.get("tta_augmentations", 5),
        use_swa=training_config.get("use_swa", False),
        swa_start=training_config.get("swa_start", 60),
        save_checkpoints=config["results"].get("save_checkpoints", True),
    )

    try:
        results = trainer.train()

        if config["results"]["save_history"]:
            history_file = results_dir / f"{model_name}_{dataset_name}_history.json"
            trainer.save_training_history(str(history_file))

        benchmark_results = {
            "model": model_name,
            "dataset": dataset_name,
            "dataset_info": dataset_info,
            "best_val_accuracy": results["best_val_accuracy"],
            "total_epochs": results["total_epochs"],
            "final_metrics": results["final_metrics"],
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "config_overrides": dataset_overrides,
            "evaluation_method": "train/test split",
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print(f"   Dataset: {dataset_name} (seq_len={dataset_info['sequence_length']})")
            print(f"   Model: {model_name} with batch_size={batch_size}")
            print(f"   Error: {str(e)[:100]}...\n")
            benchmark_results = {
                "model": model_name,
                "dataset": dataset_name,
                "error": f"CUDA OOM",
                "sequence_length": dataset_info["sequence_length"],
                "batch_size": batch_size,
            }
        else:
            raise e
    finally:
        del model
        del optimizer
        del trainer
        del train_loader
        del test_loader

    return benchmark_results


def benchmark_model_with_cross_validation(
    model_name: str,
    dataset_name: str,
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    training_config: dict[str, Any],
    dataset_overrides: dict[str, Any],
    k_folds: int = 3,
) -> dict[str, Any]:
    """
    Benchmark using k-fold cross-validation (for small datasets).
    More robust evaluation when training data is limited.
    """
    from experiments.cross_validate import cross_validate_dataset
    from src.data.loader import UCRDataLoader

    # Load full dataset for cross-validation
    loader = UCRDataLoader(
        dataset_name=dataset_name,
        normalize=training_config["normalize"],
        padding=training_config["padding"],
        max_length=dataset_overrides.get("max_length", training_config["max_length"]),
    )

    X_train, y_train, X_test, y_test = loader.load_data()

    # Combine train and test for full cross-validation
    X_full = np.concatenate([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    # Model and optimizer factory functions
    model_config = get_model_config(config, model_name)
    dataset_model_overrides = dataset_overrides.get("models", {}).get(model_name, {})

    if dataset_model_overrides:
        print(f"📋 Applying model-specific overrides for {model_name}:")
        for key, value in dataset_model_overrides.items():
            old_value = model_config.get(key, "not set")
            model_config[key] = value
            print(f"   {key}: {old_value} → {value}")
        print()

    def create_model_fn():
        return create_model(
            model_name=model_name,
            model_config=model_config,
            num_classes=dataset_info["n_classes"],
            input_length=dataset_info["sequence_length"],
            input_channels=dataset_info["n_channels"],
        )

    def create_optimizer_fn(model):
        return create_optimizer(
            model=model,
            optimizer_name=training_config["optimizer"],
            learning_rate=training_config["learning_rate"],
            **training_config.get("optimizer_params", {}),
        )

    # Loss function
    n_total = len(X_full)
    if n_total > 1000:
        label_smoothing = 0.1
    elif n_total > 200:
        label_smoothing = 0.05
    else:
        label_smoothing = 0.0

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Cross-validation parameters with small-dataset optimizations
    cv_epochs = dataset_overrides.get("epochs", training_config["epochs"])
    cv_patience = dataset_overrides.get("patience", training_config["patience"])
    cv_batch_size = dataset_overrides.get("batch_size", training_config["batch_size"])
    cv_num_workers = dataset_overrides.get("num_workers", training_config.get("num_workers", 0))

    # For very small datasets, reduce epochs and disable expensive augmentations
    n_train_per_fold = len(X_full) // k_folds
    use_tta_cv = training_config.get("use_tta", False)
    use_swa_cv = training_config.get("use_swa", False)

    if n_train_per_fold < 100:
        # Tiny dataset: reduce epochs significantly
        cv_epochs = min(cv_epochs, 50)
        cv_patience = min(cv_patience, 10)
        use_tta_cv = False  # Disable TTA for speed (too expensive for tiny datasets)
        use_swa_cv = False  # Disable SWA for tiny datasets
        print(f"⚡ Ultra-small fold size ({n_train_per_fold} samples/fold)")
        print(f"   Reducing epochs: {training_config['epochs']} → {cv_epochs}")
        print(f"   Disabling TTA and SWA for speed\n")
    elif n_train_per_fold < 200:
        # Small dataset: reduce epochs moderately
        cv_epochs = min(cv_epochs, 75)
        cv_patience = min(cv_patience, 15)
        use_tta_cv = False  # Still disable TTA
        use_swa_cv = False
        print(f"⚡ Small fold size ({n_train_per_fold} samples/fold)")
        print(f"   Reducing epochs: {training_config['epochs']} → {cv_epochs}")
        print(f"   Disabling TTA for speed\n")

    try:
        cv_results = cross_validate_dataset(
            model_fn=create_model_fn,
            X=X_full,
            y=y_full,
            optimizer_fn=create_optimizer_fn,
            criterion=criterion,
            num_classes=dataset_info["n_classes"],
            k_folds=k_folds,
            batch_size=cv_batch_size,
            epochs=cv_epochs,
            patience=cv_patience,
            device=training_config["device"],
            use_scheduler=training_config["use_scheduler"],
            warmup_epochs=training_config["warmup_epochs"],
            use_amp=training_config.get("use_amp", True),
            use_augmentation=config["training"].get("use_augmentation", True),
            augmentation_params=config["training"].get("augmentation_params", {}),
            num_workers=cv_num_workers,
            use_tta=use_tta_cv,
            tta_augmentations=training_config.get("tta_augmentations", 5),
            use_swa=use_swa_cv,
            swa_start=training_config.get("swa_start", 60),
            use_compile=training_config.get("use_compile", True),
            compile_mode=config.get("hardware", {}).get("compile_mode", "default"),
            save_checkpoints=config["results"].get("save_checkpoints", True),
        )

        benchmark_results = {
            "model": model_name,
            "dataset": dataset_name,
            "dataset_info": dataset_info,
            "cv_mean_accuracy": cv_results["mean_accuracy"],
            "cv_std_accuracy": cv_results["std_accuracy"],
            "cv_mean_f1": cv_results["mean_f1"],
            "cv_std_f1": cv_results["std_f1"],
            "k_folds": k_folds,
            "timestamp": datetime.now().isoformat(),
            "batch_size": cv_batch_size,
            "config_overrides": dataset_overrides,
            "evaluation_method": f"{k_folds}-fold cross-validation",
        }

        print(f"\n✅ Cross-Validation Results for {model_name} on {dataset_name}:")
        print(f"   Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"   Mean F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}\n")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory during cross-validation!")
            print(f"   Dataset: {dataset_name}")
            print(f"   Model: {model_name}\n")
            benchmark_results = {
                "model": model_name,
                "dataset": dataset_name,
                "error": "CUDA OOM during cross-validation",
                "evaluation_method": "cross-validation",
            }
        else:
            raise e

    return benchmark_results


def run_benchmark(config_path: str = "configs/config.yaml") -> None:
    """
    Run full benchmark across all models and datasets.

    Args:
        config_path: Path to configuration file
    """
    # Enable performance features on CUDA
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Load and validate config
    config = load_config(config_path)
    validate_config(config)

    # Set random seed
    if "seed" in config:
        set_seed(config["seed"])
        print(f"Random seed set to: {config['seed']}\n")

    # Create results directory
    results_dir = Path(config["results"]["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_results_dir = results_dir / run_id
    run_results_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    config_snapshot_file = run_results_dir / "config.yaml"
    with open(config_snapshot_file, "w") as f:
        import yaml

        yaml.dump(config, f, default_flow_style=False)

    # Track all results
    all_results: list[dict[str, Any]] = []

    # Benchmark each model on each dataset
    models = list(config["models"].keys())
    datasets = config["datasets"]

    print(f"Starting benchmark run: {run_id}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Total experiments: {len(models) * len(datasets)}\n")

    for model_name in models:
        for dataset_name in datasets:
            result = benchmark_model_on_dataset(
                model_name=model_name,
                dataset_name=dataset_name,
                config=config,
                results_dir=run_results_dir,
            )

            all_results.append(result)

            # Save intermediate results
            results_file = run_results_dir / "benchmark_results.json"
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # Generate summary
    summary = generate_summary(all_results)

    summary_file = run_results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Benchmark Complete!")
    print(f"{'=' * 80}\n")
    print(f"Results saved to: {run_results_dir}")
    print_summary(summary)


def generate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary statistics from benchmark results."""
    summary: dict[str, Any] = {
        "total_experiments": len(results),
        "successful_runs": 0,
        "failed_runs": 0,
        "skipped_runs": 0,
        "by_model": {},
        "by_dataset": {},
        "failures": [],  # Track individual failures
    }

    # Aggregate by model
    for result in results:
        if "skipped" in result:
            summary["skipped_runs"] += 1
            continue

        if "error" in result:
            summary["failed_runs"] += 1
            # Record failure details
            summary["failures"].append(
                {
                    "model": result.get("model"),
                    "dataset": result.get("dataset"),
                    "error": result.get("error"),
                }
            )
            continue

        summary["successful_runs"] += 1
        model = result["model"]
        dataset = result["dataset"]

        # Handle both train/test and cross-validation results
        if "best_val_accuracy" in result:
            accuracy = result["best_val_accuracy"]
        elif "cv_mean_accuracy" in result:
            accuracy = result["cv_mean_accuracy"]
        else:
            continue

        # By model
        if model not in summary["by_model"]:
            summary["by_model"][model] = {
                "accuracies": [],
                "datasets": [],
            }
        summary["by_model"][model]["accuracies"].append(accuracy)
        summary["by_model"][model]["datasets"].append(dataset)

        # By dataset
        if dataset not in summary["by_dataset"]:
            summary["by_dataset"][dataset] = {}
        summary["by_dataset"][dataset][model] = accuracy

    # Compute statistics
    for model in summary["by_model"]:
        accuracies = summary["by_model"][model]["accuracies"]
        summary["by_model"][model]["mean_accuracy"] = float(np.mean(accuracies))
        summary["by_model"][model]["std_accuracy"] = float(np.std(accuracies))

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print benchmark summary to console."""
    print("\nBenchmark Summary")
    print("-" * 80)

    print(f"\nTotal Experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_runs']}")
    print(f"Skipped: {summary['skipped_runs']}")
    print(f"Failed: {summary['failed_runs']}")

    print("\nResults by Model:")
    for model, stats in summary["by_model"].items():
        print(f"\n  {model.upper()}:")
        print(f"    Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"    Datasets: {len(stats['datasets'])}")

    print("\n" + "-" * 80)


if __name__ == "__main__":
    import torch

    torch.backends.cudnn.benchmark = True

    run_benchmark()
