import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from experiments.benchmark import run_benchmark

RESULTS_DIR = Path("results")
CONFIG_PATH = Path("configs/config.yaml")
BACKUP_PATH = CONFIG_PATH.with_suffix(".backup.yaml")


def find_latest_run_dir(results_dir: Path) -> Path | None:
    runs = [p for p in results_dir.iterdir() if p.is_dir() and p.name[0].isdigit()]
    return sorted(runs)[-1] if runs else None


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        return None
    with open(summary_file, "r") as f:
        return json.load(f)


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(path: Path, cfg: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def backup_config() -> None:
    if not BACKUP_PATH.exists():
        shutil.copy2(CONFIG_PATH, BACKUP_PATH)


def bump_capacity(m: dict[str, Any], d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float) -> None:
    m["d_model"] = max(d_model, int(m.get("d_model", d_model)))
    m["num_layers"] = max(num_layers, int(m.get("num_layers", num_layers)))
    m["num_heads"] = max(num_heads, int(m.get("num_heads", num_heads)))
    m["d_ff"] = max(d_ff, int(m.get("d_ff", d_ff)))
    m["dropout"] = min(dropout, float(m.get("dropout", dropout)))


def extract_failed_datasets(summary: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Extract datasets that failed with OOM or other errors.

    Returns:
        List of (dataset_name, model_name) tuples that failed
    """
    failed: list[tuple[str, str]] = []

    # Check by_dataset for missing model results (indicates failure)
    by_dataset = summary.get("by_dataset", {})

    for ds, model_results in by_dataset.items():
        # Expected models
        expected_models = ["transformer", "cats", "autoformer", "patchtst", "cnn", "fcn"]

        for model in expected_models:
            if model not in model_results:
                # This model failed on this dataset
                failed.append((ds, model))

    return failed


def apply_failure_recovery(cfg: dict[str, Any], failed: list[tuple[str, str]]) -> dict[str, Any]:
    """
    Apply aggressive recovery strategies for failed dataset/model combinations.
    """
    dataset_overrides = cfg.setdefault("dataset_overrides", {})
    training = cfg.setdefault("training", {})

    # Group failures by dataset
    failures_by_dataset: dict[str, list[str]] = {}
    for ds, model in failed:
        failures_by_dataset.setdefault(ds, []).append(model)

    for ds, failed_models in failures_by_dataset.items():
        print(f"⚠️  Recovery for {ds}: {len(failed_models)} models failed")

        ds_override = dataset_overrides.setdefault(ds, {})
        ds_models = ds_override.setdefault("models", {})

        # Aggressive batch size reduction
        current_batch = ds_override.get("batch_size", training.get("batch_size", 64))
        ds_override["batch_size"] = max(4, current_batch // 2)
        print(f"   Reducing batch_size: {current_batch} → {ds_override['batch_size']}")

        # Reduce CV folds for small datasets
        if ds_override.get("cv_folds", training.get("cv_folds", 3)) > 3:
            ds_override["cv_folds"] = 3
            print(f"   Reducing cv_folds to 3")

        # For each failed model, apply model-specific fixes
        for model in failed_models:
            if model in ["transformer", "cats", "autoformer", "patchtst"]:
                # Attention models: reduce capacity drastically
                model_cfg = ds_models.setdefault(model, {})

                # Reduce model size
                model_cfg["d_model"] = min(model_cfg.get("d_model", 192), 128)
                model_cfg["num_layers"] = min(model_cfg.get("num_layers", 3), 2)
                model_cfg["num_heads"] = min(model_cfg.get("num_heads", 4), 4)
                model_cfg["d_ff"] = min(model_cfg.get("d_ff", 768), 512)
                model_cfg["dropout"] = max(model_cfg.get("dropout", 0.1), 0.15)

                print(f"   {model}: d_model={model_cfg['d_model']}, layers={model_cfg['num_layers']}")

        # Aggressive sequence truncation for long sequences
        if "max_length" not in ds_override:
            # Estimate if we need truncation
            if ds in ("ChlorineConcentration", "KeplerLightCurves"):
                ds_override["max_length"] = 256  # Very aggressive
                print(f"   Setting max_length to 256 (aggressive truncation)")
            elif ds in ("Adiac", "FiftyWords"):
                ds_override["max_length"] = 512
                print(f"   Setting max_length to 512")
        else:
            # If already set, reduce further
            current_max = ds_override["max_length"]
            ds_override["max_length"] = max(128, current_max // 2)
            print(f"   Reducing max_length: {current_max} → {ds_override['max_length']}")

        # Reduce epochs for faster iteration
        ds_override["epochs"] = min(ds_override.get("epochs", 100), 60)
        ds_override["patience"] = min(ds_override.get("patience", 15), 8)

        print()

    return cfg


def tweak_from_summary(cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Enhanced tweaking with failure recovery."""
    training = cfg.setdefault("training", {})
    models_cfg = cfg.setdefault("models", {})
    dataset_overrides = cfg.setdefault("dataset_overrides", {})

    # Ensure speed settings are honored via config
    training["cv_folds"] = training.get("cv_folds", 3)
    training["num_workers"] = training.get("num_workers", 2)
    training["use_tta"] = False
    training["use_swa"] = False
    training["use_compile"] = training.get("use_compile", True)

    # Check for failures first
    failed = extract_failed_datasets(summary)
    if failed:
        print(f"\n🔧 Detected {len(failed)} dataset/model failures")
        cfg = apply_failure_recovery(cfg, failed)

    # Mild global capacity bump for transformer-family
    bump_capacity(models_cfg.setdefault("transformer", {}), 192, 3, 4, 768, 0.1)
    bump_capacity(models_cfg.setdefault("cats", {}), 192, 3, 4, 768, 0.1)
    bump_capacity(models_cfg.setdefault("autoformer", {}), 192, 3, 4, 768, 0.1)
    bump_capacity(models_cfg.setdefault("patchtst", {}), 160, 3, 4, 640, 0.1)

    # Detect datasets where CNN outperforms transformer-family by ≥5%
    by_dataset = summary.get("by_dataset", {})
    weak_datasets: list[str] = []
    for ds, metrics in by_dataset.items():
        cnn_acc = metrics.get("cnn")
        tx_scores = [metrics[m] for m in ("transformer", "cats", "autoformer", "patchtst") if m in metrics]
        if cnn_acc is not None and tx_scores:
            tx_mean = sum(tx_scores) / len(tx_scores)
            if cnn_acc - tx_mean >= 0.05:
                weak_datasets.append(ds)

    for ds in weak_datasets:
        ds_override = dataset_overrides.setdefault(ds, {})
        ds_models = ds_override.setdefault("models", {})

        # Smaller batches for stability on tiny sets
        if "batch_size" not in ds_override:
            ds_override["batch_size"] = max(8, training.get("batch_size", 64) // 2)

        # Cap epochs/patience to keep CV fast
        ds_override["epochs"] = min(ds_override.get("epochs", training.get("epochs", 100)), 80)
        ds_override["patience"] = min(ds_override.get("patience", training.get("patience", 15)), 12)

        # Long sequences: truncate to help attention
        if ds in ("ChlorineConcentration", "Adiac", "KeplerLightCurves"):
            ds_override["max_length"] = ds_override.get("max_length", 512 if ds == "KeplerLightCurves" else 1024)

        # Stronger transformer-family per-dataset
        bump_capacity(ds_models.setdefault("transformer", {}), 256, 3, 4, 1024, 0.08)
        bump_capacity(ds_models.setdefault("cats", {}), 256, 3, 4, 1024, 0.08)
        bump_capacity(ds_models.setdefault("autoformer", {}), 256, 3, 4, 1024, 0.10)
        bump_capacity(ds_models.setdefault("patchtst", {}), 192, 3, 4, 768, 0.10)

    return cfg


def auto_tune_and_run(rounds: int = 2) -> None:
    latest = find_latest_run_dir(RESULTS_DIR)
    if latest is None:
        print("No previous results found. Running initial benchmark...")
        run_benchmark()
        latest = find_latest_run_dir(RESULTS_DIR)
        if latest is None:
            print("Failed to produce an initial run; aborting.")
            return

    # latest is guaranteed to be a Path here
    summary = load_summary(latest)
    if summary is None:
        print(f"No summary.json in {latest}, running benchmark to create baseline...")
        run_benchmark()
        latest = find_latest_run_dir(RESULTS_DIR)
        if latest is None:
            print("Failed to produce an initial run; aborting.")
            return
        summary = load_summary(latest)
        if summary is None:
            print("Failed to load summary after baseline; aborting.")
            return

    backup_config()

    for r in range(1, rounds + 1):
        print(f"\n=== Auto-tune round {r} ===")
        cfg = load_config(CONFIG_PATH)
        cfg = tweak_from_summary(cfg, summary)
        save_config(CONFIG_PATH, cfg)
        print("✓ Config updated to boost transformer-family.")

        print(f"→ Starting benchmark round {r}...")
        run_benchmark(str(CONFIG_PATH))

        latest = find_latest_run_dir(RESULTS_DIR)
        if latest:
            new_summary = load_summary(latest)
            if new_summary:
                summary = new_summary
                print("✓ Loaded updated summary for next round.")
            else:
                print("⚠️ Could not load updated summary; reusing previous summary.")


if __name__ == "__main__":
    auto_tune_and_run(rounds=2)
