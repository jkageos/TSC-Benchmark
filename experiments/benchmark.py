"""
Main benchmarking orchestration for TSC models.
"""

import gc
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from experiments.cross_validate import cross_validate_dataset
from src.data.loader import UCRDataLoader, load_ucr_dataset
from src.training.trainer import Trainer
from src.utils.config import create_model, create_optimizer, load_config
from src.utils.system import get_system_resources

logger = logging.getLogger(__name__)


def clear_cuda_memory() -> None:
    """Aggressively clear CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


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
    clear_cuda_memory()

    # Print system resources at start
    resources = get_system_resources()
    print(f"\n📊 System Resources:")
    print(f"   CPU: {resources['cpu_count']} cores @ {resources['cpu_percent']:.1f}% utilization")
    print(f"   Memory: {resources['memory_available_gb']:.1f}/{resources['memory_total_gb']:.1f} GB available")
    if "gpu_name" in resources:
        print(
            f"   GPU: {resources['gpu_name']} ({resources['gpu_memory_total_gb']:.1f}GB VRAM, "
            f"{resources.get('gpu_memory_free_gb', 0):.2f}GB free)\n"
        )

    # Extract configs
    hardware_config = config.get("hardware", {})
    training_config = config.get("training", {})
    dataset_overrides = config.get("dataset_overrides", {}).get(dataset_name, {})

    print(f"\n{'=' * 80}")
    print(f"Benchmarking {model_name.upper()} on {dataset_name}")
    print(f"{'=' * 80}\n")

    # Merge dataset-specific overrides
    batch_size = dataset_overrides.get("batch_size", training_config["batch_size"])
    epochs = dataset_overrides.get("epochs", training_config["epochs"])
    patience = dataset_overrides.get("patience", training_config["patience"])
    max_length = dataset_overrides.get("max_length", training_config.get("max_length"))

    # Get model-specific overrides from dataset config
    model_overrides = dataset_overrides.get("models", {}).get(model_name, {})

    # Load dataset
    try:
        train_loader, test_loader, dataset_info = load_ucr_dataset(
            dataset_name=dataset_name,
            batch_size=batch_size,
            normalize=training_config["normalize"],
            padding=training_config["padding"],
            max_length=max_length,
            num_workers=training_config.get("num_workers", 0),
            max_cpu_load=hardware_config.get("max_cpu_load", 0.5),
            auto_workers=hardware_config.get("auto_workers", False),
            max_workers_override=hardware_config.get("max_workers_override"),
        )
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        clear_cuda_memory()
        return {"error": str(e)}

    # Print dataset info
    print(f"Dataset: {dataset_name}")
    print(f"  Train samples: {dataset_info['n_train']}")
    print(f"  Test samples: {dataset_info['n_test']}")
    print(f"  Classes: {dataset_info['n_classes']}")
    print(f"  Sequence length: {dataset_info['sequence_length']}\n")

    # Small dataset → Cross-validation
    if dataset_info["n_train"] < 300:
        print(f"⚠️  Small dataset ({dataset_info['n_train']} samples) → Using cross-validation\n")
        return benchmark_with_cross_validation(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            dataset_info=dataset_info,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            model_overrides=model_overrides,
            max_length=max_length,
            k_folds=training_config.get("cv_folds", 5),
        )

    # Merge base model config with dataset-specific overrides
    base_model_config = config["models"][model_name].copy()
    base_model_config.update(model_overrides)

    # Standard train/test split with auto-capacity reduction
    try:
        model = create_model(
            model_name=model_name,
            model_config=base_model_config,
            num_classes=dataset_info["n_classes"],
            input_length=dataset_info["sequence_length"],
            input_channels=1,
        )
    except RuntimeError as e:
        print(f"❌ Model creation failed: {e}")
        clear_cuda_memory()
        return {"error": f"Model creation failed: {e}"}

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model=model,
        optimizer_name=training_config["optimizer"],
        learning_rate=training_config["learning_rate"],
        **training_config.get("optimizer_params", {}),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=dataset_info["n_classes"],
        epochs=epochs,
        device=hardware_config.get("device", "cuda"),
        patience=patience,
        use_scheduler=training_config.get("use_scheduler", True),
        warmup_epochs=training_config.get("warmup_epochs", 5),
        use_amp=hardware_config.get("use_amp", True),
        use_compile=hardware_config.get("use_compile", True),
        compile_mode=hardware_config.get("compile_mode", "default"),
        use_augmentation=training_config.get("use_augmentation", True),
        augmentation_params=training_config.get("augmentation_params", {}),
        use_tta=training_config.get("use_tta", False),
        tta_augmentations=training_config.get("tta_augmentations", 5),
        use_swa=training_config.get("use_swa", False),
        swa_start=training_config.get("swa_start", 60),
        checkpoint_dir=str(results_dir / "checkpoints" / model_name / dataset_name),
    )

    start_time = time.time()
    try:
        history = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ Out of memory during training: {e}")
            clear_cuda_memory()
            return {"error": f"OOM during training: {e}"}
        raise

    training_time = time.time() - start_time

    best_metrics = history["best_metrics"]

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": best_metrics["accuracy"],
        "f1_macro": best_metrics["f1_macro"],
        "f1_weighted": best_metrics["f1_weighted"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "training_time": training_time,
        "best_epoch": history["best_epoch"],
        "dataset_info": dataset_info,
    }

    clear_cuda_memory()
    return results


def benchmark_with_cross_validation(
    model_name: str,
    dataset_name: str,
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    batch_size: int,
    epochs: int,
    patience: int,
    model_overrides: dict[str, Any],
    max_length: int | None = None,
    k_folds: int = 5,
) -> dict[str, Any]:
    """Benchmark with k-fold cross-validation (for small datasets)."""
    loader = UCRDataLoader(
        dataset_name=dataset_name,
        normalize=config["training"]["normalize"],
        padding=config["training"]["padding"],
        max_length=max_length,
    )

    # Load combined dataset with guaranteed consistent sequence length
    X_full, y_full = loader.load_data_for_cross_validation()

    # **CRITICAL**: Get actual sequence length from loaded data, not dataset_info
    # dataset_info was computed from original train/test split before truncation/padding
    actual_sequence_length = X_full.shape[1]

    # Update dataset_info with actual sequence length for model instantiation
    dataset_info = {
        **dataset_info,
        "sequence_length": actual_sequence_length,
        "n_train": X_full.shape[0],  # Combined dataset size for logging
    }

    # Verify dataset info matches combined data
    print(f"\nCombined dataset for CV:")
    print(f"  Total samples: {X_full.shape[0]}")
    print(f"  Sequence length: {actual_sequence_length}")
    print(f"  Classes: {len(np.unique(y_full))}\n")

    # Merge base model config with dataset-specific overrides
    base_model_config = config["models"][model_name].copy()
    base_model_config.update(model_overrides)

    def create_model_fn() -> nn.Module:
        return create_model(
            model_name=model_name,
            model_config=base_model_config,
            num_classes=dataset_info["n_classes"],
            input_length=actual_sequence_length,  # Use actual length, not original
            input_channels=1,
        )

    def create_optimizer_fn(model: nn.Module) -> torch.optim.Optimizer:
        return create_optimizer(
            model=model,
            optimizer_name=config["training"]["optimizer"],
            learning_rate=config["training"]["learning_rate"],
            **config["training"].get("optimizer_params", {}),
        )

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    cv_results = cross_validate_dataset(
        model_fn=create_model_fn,
        X=X_full,
        y=y_full,
        optimizer_fn=create_optimizer_fn,
        criterion=criterion,
        num_classes=dataset_info["n_classes"],
        k_folds=k_folds,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        hardware_config=config.get("hardware", {}),
        training_config=config.get("training", {}),
        save_checkpoints=config.get("results", {}).get("save_checkpoints", True),
    )
    training_time = time.time() - start_time

    return {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": cv_results["mean_accuracy"],
        "accuracy_std": cv_results["std_accuracy"],
        "f1_macro": cv_results["mean_f1_macro"],
        "f1_std": cv_results["std_f1_macro"],
        "training_time": training_time,
        "cv_folds": k_folds,
        "dataset_info": dataset_info,
        "fold_accuracies": cv_results["fold_accuracies"],
        "fold_f1_scores": cv_results["fold_f1_scores"],
    }


def run_benchmark(config_path: str = "configs/config.yaml") -> None:
    """Run full benchmark suite."""
    config = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["results"]["save_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for dataset_name in config["datasets"]:
        for model_name in config["models"].keys():
            result = benchmark_model_on_dataset(
                model_name=model_name,
                dataset_name=dataset_name,
                config=config,
                results_dir=results_dir,
            )
            all_results.append(result)

            # Save incremental results
            with open(results_dir / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Save final config snapshot
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Benchmark complete! Results saved to {results_dir}")


def run_model_test(
    model_name: str,
    dataset_name: str = "Beef",
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """
    Quick test of a single model on smallest dataset.

    Args:
        model_name: Model to test ('fcn', 'cnn', 'transformer', etc.)
        dataset_name: Dataset to use (default: Beef with 30 samples)
        config_path: Config path

    Returns:
        Test results with timing stats
    """
    print(f"\n{'=' * 80}")
    print(f"QUICK MODEL TEST: {model_name.upper()} on {dataset_name}")
    print(f"{'=' * 80}\n")

    try:
        config = load_config(config_path)
        results_dir = Path(config["results"]["save_dir"]) / "test_run"
        results_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Run benchmark directly
        result = benchmark_model_on_dataset(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            results_dir=results_dir,
        )

        elapsed = time.time() - start_time

        # Add timing info
        result["total_elapsed"] = elapsed

        print(f"\n✅ Test complete: {elapsed:.1f}s")

        return result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "model": model_name,
            "dataset": dataset_name,
            "success": False,
            "error": str(e),
        }


def monitor_gpu_memory() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9
    return 0.0


if __name__ == "__main__":
    run_benchmark()
