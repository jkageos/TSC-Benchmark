"""
Unified entry point for TSC-Benchmark.

Supports multiple modes via configuration:
- benchmark: Full suite across all datasets
- test: Quick validation on smallest dataset
- tune: Auto-tuning with iterative improvement
- single: Single model-dataset combo for debugging

All behavior controlled via configs/config.yaml
"""

import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import setup_logging


def main() -> None:
    """Main entry point - mode selected from config."""
    config_path = Path("configs/config.yaml")
    config = load_config(str(config_path))

    # Get execution mode from config
    execution = config.get("execution", {})
    mode = execution.get("mode", "benchmark")
    verbose = execution.get("verbose", True)

    # Setup logging
    setup_logging(verbose=verbose)

    print("\n" + "=" * 80)
    print(f"TSC-Benchmark | Mode: {mode.upper()}")
    print("=" * 80 + "\n")

    # Route to appropriate handler
    if mode == "benchmark":
        from experiments.benchmark import run_benchmark

        run_benchmark(str(config_path))

    elif mode == "test":
        from experiments.benchmark import benchmark_model_on_dataset

        test_model = execution.get("test_model", "fcn")
        test_dataset = execution.get("test_dataset", "Beef")
        results_dir = Path(config["results"]["save_dir"]) / "test_run"
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Testing {test_model.upper()} on {test_dataset}...\n")

        # Run single model-dataset combo directly (no subprocess)
        result = benchmark_model_on_dataset(
            model_name=test_model,
            dataset_name=test_dataset,
            config=config,
            results_dir=results_dir,
        )

        print(f"\n{'=' * 80}")
        print("TEST RESULTS")
        print("=" * 80)
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Model: {result['model']}")
            print(f"   Dataset: {result['dataset']}")
            print(f"   Accuracy: {result.get('accuracy', 'N/A'):.4f}")
            print(f"   F1 (macro): {result.get('f1_macro', 'N/A'):.4f}")
            print(f"   Training time: {result.get('training_time', 0):.1f}s")
            print(f"   Best epoch: {result.get('best_epoch', 'N/A')}")

    elif mode == "tune":
        from experiments.auto_tune import auto_tune_and_run

        rounds = execution.get("tune_rounds", 2)
        auto_tune_and_run(rounds=rounds)

    elif mode == "single":
        from experiments.benchmark import benchmark_model_on_dataset

        model_name = execution.get("single_model", "fcn")
        dataset_name = execution.get("single_dataset", "Beef")
        results_dir = Path(config["results"]["save_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        result = benchmark_model_on_dataset(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            results_dir=results_dir,
        )
        print(f"\nResult: {result}")

    else:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: benchmark, test, tune, single")
        sys.exit(1)

    print("\n✅ Execution complete!")


if __name__ == "__main__":
    main()
