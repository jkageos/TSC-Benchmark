"""
Main benchmarking orchestration for TSC models.

Focuses purely on model evaluation pipeline:
- Loading datasets
- Instantiating models
- Running training/evaluation
- Tracking results

All hardware/system management delegated to src/utils/system.py
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from experiments.cross_validate import cross_validate_dataset
from src.data.loader import UCRDataLoader, load_ucr_dataset
from src.training.trainer import Trainer
from src.utils.config import create_model, create_optimizer, load_config
from src.utils.system import (
    adjust_hyperparameters_for_memory,
    aggressive_memory_cleanup,
    clear_cuda_memory,
    get_system_resources,
)

logger = logging.getLogger(__name__)


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

    # Extract configs
    hardware_config = config.get("hardware", {})
    training_config = config.get("training", {})
    dataset_overrides = config.get("dataset_overrides", {}).get(dataset_name, {})

    # Extract training hyperparameters with dataset overrides
    epochs = dataset_overrides.get("epochs", training_config.get("epochs", 100))
    patience = dataset_overrides.get("patience", training_config.get("patience", 15))
    batch_size = dataset_overrides.get("batch_size", training_config["batch_size"])
    max_length = dataset_overrides.get("max_length", training_config.get("max_length"))

    # Load dataset first to get sequence info (DISABLE auto_workers on first load)
    try:
        train_loader, test_loader, dataset_info = load_ucr_dataset(
            dataset_name=dataset_name,
            batch_size=batch_size,  # Use config default initially
            normalize=training_config["normalize"],
            padding=training_config["padding"],
            max_length=max_length,
            num_workers=training_config.get("num_workers", 0),
            max_cpu_load=hardware_config.get("max_cpu_load", 0.5),
            auto_workers=False,  # Disable to avoid messages
            max_workers_override=hardware_config.get("max_workers_override"),
        )
    except Exception as e:
        return {"error": f"Dataset loading failed: {e}"}

    # AGGRESSIVE MEMORY ADJUSTMENT (from system.py)
    resources = get_system_resources()
    gpu_memory_gb = resources.get("gpu_memory_total_gb", 8.0)

    # Adjust hyperparameters based on GPU memory and dataset characteristics
    base_model_config = config["models"][model_name].copy()
    model_overrides = dataset_overrides.get("models", {}).get(model_name, {})
    base_model_config.update(model_overrides)

    adjusted_batch_size, adjusted_max_length, adjusted_model_config = adjust_hyperparameters_for_memory(
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        batch_size=batch_size,
        model_name=model_name,
        model_config=base_model_config,
        gpu_memory_gb=gpu_memory_gb,
    )

    # Reload dataloaders with adjusted batch size if needed
    if adjusted_batch_size != batch_size or adjusted_max_length != (max_length or dataset_info["sequence_length"]):
        clear_cuda_memory()
        try:
            train_loader, test_loader, dataset_info = load_ucr_dataset(
                dataset_name=dataset_name,
                batch_size=adjusted_batch_size,
                normalize=training_config["normalize"],
                padding=training_config["padding"],
                max_length=adjusted_max_length,
                num_workers=training_config.get("num_workers", 0),
                max_cpu_load=hardware_config.get("max_cpu_load", 0.5),
                auto_workers=False,
                max_workers_override=hardware_config.get("max_workers_override"),
            )
        except Exception as e:
            return {"error": f"Dataset reload failed: {e}"}

    # Small dataset → Cross-validation (silent)
    if dataset_info["n_train"] < 300:
        return benchmark_with_cross_validation(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            dataset_info=dataset_info,
            batch_size=adjusted_batch_size,
            epochs=epochs,
            patience=patience,
            model_overrides=adjusted_model_config,
            max_length=adjusted_max_length,
            k_folds=training_config.get("cv_folds", 5),
            results_dir=results_dir,
        )

    # Standard train/test split with auto-capacity reduction
    try:
        aggressive_memory_cleanup()

        # Reset memory stats to track peak usage for this specific model
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = create_model(
            model_name=model_name,
            model_config=adjusted_model_config,
            num_classes=dataset_info["n_classes"],
            input_length=dataset_info["sequence_length"],
            input_channels=1,
        )

        # Calculate parameter count
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    except RuntimeError as e:
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
        use_augmentation=training_config.get("use_augmentation", False),
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
            clear_cuda_memory()
            return {"error": f"OOM during training: {e}"}
        raise

    training_time = time.time() - start_time

    # Capture peak memory
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

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
        "num_params": num_params,
        "peak_memory_mb": peak_memory_mb,
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
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark with k-fold cross-validation (for small datasets)."""
    loader = UCRDataLoader(
        dataset_name=dataset_name,
        normalize=config["training"]["normalize"],
        padding=config["training"]["padding"],
        max_length=max_length,
    )

    # Load combined dataset (silent - no prints)
    X_full, y_full = loader.load_data_for_cross_validation()

    actual_sequence_length = X_full.shape[1]

    # Update dataset_info with actual sequence length
    dataset_info = {
        **dataset_info,
        "sequence_length": actual_sequence_length,
        "n_train": X_full.shape[0],
    }

    # Merge base model config with dataset-specific overrides
    base_model_config = config["models"][model_name].copy()
    base_model_config.update(model_overrides)

    def create_model_fn() -> nn.Module:
        aggressive_memory_cleanup()
        return create_model(
            model_name=model_name,
            model_config=base_model_config,
            num_classes=dataset_info["n_classes"],
            input_length=actual_sequence_length,
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

    checkpoint_base_dir: str | None = None
    if config.get("results", {}).get("save_checkpoints", True) and results_dir is not None:
        checkpoint_path = results_dir / "checkpoints" / "cross_validation" / model_name / dataset_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_base_dir = str(checkpoint_path)

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
        checkpoint_base_dir=checkpoint_base_dir,
    )
    training_time = time.time() - start_time

    return {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": cv_results["mean_accuracy"],
        "f1_macro": cv_results["mean_f1_macro"],
        "f1_weighted": cv_results["mean_f1_macro"],  # Use macro F1 as proxy for weighted
        "precision": cv_results["mean_accuracy"],  # CV doesn't track precision separately
        "recall": cv_results["mean_accuracy"],  # CV doesn't track recall separately
        "training_time": training_time,
        "best_epoch": 0,  # Not tracked in CV
        "dataset_info": dataset_info,
        "cv_folds": k_folds,
        "cv_std_accuracy": cv_results.get("std_accuracy", 0),
    }


def run_benchmark(config_path: str = "configs/config.yaml") -> None:
    """Run full benchmark suite."""
    config = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["results"]["save_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    # Calculate total combinations for progress bar
    total_combinations = len(config["datasets"]) * len(config["models"])

    # Print header ONCE before progress bar starts
    print(f"\n{'=' * 80}")
    print("🚀 BENCHMARK SUITE")
    print(f"{'=' * 80}")
    print(f"📊 Datasets: {len(config['datasets'])}")
    print(f"🧠 Models: {len(config['models'])}")
    print(f"🔢 Total combinations: {total_combinations}")
    print(f"💾 Results: {results_dir}")
    print(f"{'=' * 80}\n")

    # Track successful/failed runs
    successful_runs = 0
    failed_runs = 0

    # Create progress bar with enhanced formatting
    with tqdm(
        total=total_combinations,
        desc="Overall Progress",
        unit="run",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ncols=100,
        colour="green",
        position=0,
        leave=True,
    ) as pbar:
        for dataset_idx, dataset_name in enumerate(config["datasets"], 1):
            for model_idx, model_name in enumerate(config["models"].keys(), 1):
                # Compact status message
                current_combo = (dataset_idx - 1) * len(config["models"]) + model_idx
                status = f"{model_name:12s} on {dataset_name:20s}"

                # Update description without creating new line
                pbar.set_description(f"[{current_combo}/{total_combinations}] {status}")

                # Run benchmark (all prints will appear above progress bar)
                result = benchmark_model_on_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    config=config,
                    results_dir=results_dir,
                )
                all_results.append(result)

                # Track success/failure and print summary line
                if "error" in result:
                    failed_runs += 1
                    pbar.write(f"❌ FAILED | {status} | {result.get('error', 'Unknown error')}")
                else:
                    successful_runs += 1
                    acc = result.get("accuracy", 0.0)
                    f1 = result.get("f1_macro", 0.0)
                    time_taken = result.get("training_time", 0.0)
                    pbar.write(f"✅ SUCCESS | {status} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {time_taken:.1f}s")

                pbar.update(1)

    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save config snapshot for reproducibility
    config_file = results_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(f"✅ Successful: {successful_runs}/{total_combinations}")
    print(f"❌ Failed: {failed_runs}/{total_combinations}")
    print(f"📊 Results saved: {results_file}")
    print(f"📋 Config saved: {config_file}")
    print(f"{'=' * 80}\n")


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


if __name__ == "__main__":
    run_benchmark()
