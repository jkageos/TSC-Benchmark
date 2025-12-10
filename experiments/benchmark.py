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


def benchmark_model_on_dataset(
    model_name: str,
    dataset_name: str,
    config: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    """
    Benchmark a single model on a single dataset.

    Args:
        model_name: Name of the model to benchmark
        dataset_name: Name of the dataset
        config: Configuration dictionary
        results_dir: Directory to save results

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 80}")
    print(f"Benchmarking {model_name.upper()} on {dataset_name}")
    print(f"{'=' * 80}\n")

    # Load dataset
    training_config = get_training_config(config)

    try:
        train_loader, test_loader, dataset_info = load_ucr_dataset(
            dataset_name=dataset_name,
            batch_size=training_config["batch_size"],
            normalize=training_config["normalize"],
            padding=training_config["padding"],
            max_length=training_config["max_length"],
        )
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return {"error": str(e)}

    print(f"Dataset Info:")
    print(f"  Classes: {dataset_info['n_classes']}")
    print(f"  Channels: {dataset_info['n_channels']}")
    print(f"  Sequence Length: {dataset_info['sequence_length']}")
    print(f"  Train Samples: {dataset_info['n_train']}")
    print(f"  Test Samples: {dataset_info['n_test']}\n")

    # Create model
    model_config = get_model_config(config, model_name)
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

    # Loss function
    criterion = nn.CrossEntropyLoss()

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
        epochs=training_config["epochs"],
        patience=training_config["patience"],
        device=training_config["device"],
        checkpoint_dir=str(checkpoint_dir),
        use_scheduler=training_config["use_scheduler"],
    )

    # Train model
    results = trainer.train()

    # Save training history
    if config["results"]["save_history"]:
        history_file = results_dir / f"{model_name}_{dataset_name}_history.json"
        trainer.save_training_history(str(history_file))

    # Prepare results
    benchmark_results = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_info": dataset_info,
        "best_val_accuracy": results["best_val_accuracy"],
        "total_epochs": results["total_epochs"],
        "final_metrics": results["final_metrics"],
        "timestamp": datetime.now().isoformat(),
    }

    return benchmark_results


def run_benchmark(config_path: str = "configs/config.yaml") -> None:
    """
    Run full benchmark across all models and datasets.

    Args:
        config_path: Path to configuration file
    """
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
    """
    Generate summary statistics from benchmark results.

    Args:
        results: List of benchmark results

    Returns:
        Summary dictionary
    """
    summary: dict[str, Any] = {
        "total_experiments": len(results),
        "by_model": {},
        "by_dataset": {},
    }

    # Aggregate by model
    for result in results:
        if "error" in result:
            continue

        model = result["model"]
        dataset = result["dataset"]
        accuracy = result["best_val_accuracy"]

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
    """
    Print benchmark summary to console.

    Args:
        summary: Summary dictionary
    """
    print("\nBenchmark Summary")
    print("-" * 80)

    print("\nResults by Model:")
    for model, stats in summary["by_model"].items():
        print(f"\n  {model.upper()}:")
        print(
            f"    Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}"
        )
        print(f"    Datasets: {len(stats['datasets'])}")

    print("\n" + "-" * 80)


if __name__ == "__main__":
    run_benchmark()
