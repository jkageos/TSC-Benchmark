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

import torch
import torch.nn as nn
from tqdm import tqdm

from experiments.cross_validate import cross_validate_dataset
from src.data.loader import UCRDataLoader, load_ucr_dataset
from src.training.trainer import Trainer
from src.utils.config import create_model, create_optimizer, load_config
from src.utils.system import (
    get_system_resources,
    recommend_batch_size,  # ADD THIS IMPORT
)

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

    # REMOVED: System resources print (too verbose)
    # Only print critical information

    # Extract configs
    hardware_config = config.get("hardware", {})
    training_config = config.get("training", {})
    dataset_overrides = config.get("dataset_overrides", {}).get(dataset_name, {})

    # Extract training hyperparameters with dataset overrides
    epochs = dataset_overrides.get("epochs", training_config.get("epochs", 100))
    patience = dataset_overrides.get("patience", training_config.get("patience", 15))
    batch_size = dataset_overrides.get("batch_size", training_config["batch_size"])
    max_length = dataset_overrides.get("max_length", training_config.get("max_length"))

    # Extract model-specific overrides
    model_overrides = dataset_overrides.get("models", {}).get(model_name, {})

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

    # **SINGLE optimization point - silent**
    resources = get_system_resources()
    gpu_memory_gb = resources.get("gpu_memory_total_gb", 6.0)
    recommended_batch_size = recommend_batch_size(
        dataset_size=dataset_info["n_train"],
        sequence_length=dataset_info["sequence_length"],
        gpu_memory_gb=gpu_memory_gb,
    )

    # Use recommendation if significantly better (silent optimization)
    if recommended_batch_size > batch_size * 1.25:
        batch_size = recommended_batch_size

        # Reload dataloaders with optimized batch size (silent)
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

    # REMOVED: Dataset info print (too verbose)

    # Small dataset → Cross-validation (silent)
    if dataset_info["n_train"] < 300:
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
            results_dir=results_dir,
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

    # REMOVED: CV dataset info print (too verbose)

    # Merge base model config with dataset-specific overrides
    base_model_config = config["models"][model_name].copy()
    base_model_config.update(model_overrides)

    def create_model_fn() -> nn.Module:
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
                    pbar.write(f"✅ SUCCESS | {status} | Acc: {acc:6.4f} | F1: {f1:6.4f} | Time: {time_taken:5.1f}s")

                # Update progress bar
                pbar.update(1)

                # Save incremental results
                with open(results_dir / "results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    # Final summary (after progress bar completes)
    print(f"\n{'=' * 80}")
    print("📈 BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(f"✅ Successful runs: {successful_runs}/{total_combinations}")
    print(f"❌ Failed runs: {failed_runs}/{total_combinations}")
    print(f"📁 Results saved to: {results_dir}")
    print(f"{'=' * 80}\n")

    # Save final config snapshot
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Benchmark complete! Results saved to {results_dir}")


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
