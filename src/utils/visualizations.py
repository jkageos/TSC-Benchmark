"""
Visualization utilities for TSC-Benchmark results.

Generates:
1. Dataset information table
2. Maximum validation accuracy heatmap
3. Average validation accuracy heatmap
4. Efficiency and performance analysis plots
"""

import json
import sys
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Dataset metadata for UCR datasets
DATASET_METADATA = {
    "Adiac": {
        "train_size": 390,
        "test_size": 391,
        "length": 176,
        "n_classes": 37,
        "type": "Multivariate",
        "source": "UCR Archive",
    },
    "ArrowHead": {
        "train_size": 36,
        "test_size": 175,
        "length": 251,
        "n_classes": 3,
        "type": "Univariate",
        "source": "UCR Archive",
    },
    "Beef": {
        "train_size": 30,
        "test_size": 30,
        "length": 470,
        "n_classes": 5,
        "type": "Univariate",
        "source": "UCR Archive",
    },
    "Car": {
        "train_size": 60,
        "test_size": 60,
        "length": 577,
        "n_classes": 4,
        "type": "Univariate",
        "source": "UCR Archive",
    },
    "ChlorineConcentration": {
        "train_size": 467,
        "test_size": 3840,
        "length": 166,
        "n_classes": 3,
        "type": "Univariate",
        "source": "UCR Archive",
    },
    "CinCECGTorso": {
        "train_size": 40,
        "test_size": 1380,
        "length": 1638,
        "n_classes": 4,
        "type": "Univariate",
        "source": "UCR Archive",
    },
    "FiftyWords": {
        "train_size": 450,
        "test_size": 455,
        "length": 270,
        "n_classes": 50,
        "type": "Univariate",
        "source": "UCR Archive",
    },
    "HouseTwenty": {
        "train_size": 40,
        "test_size": 40,
        "length": 2000,
        "n_classes": 2,
        "type": "Univariate",
        "source": "UCR Archive",
    },
    "KeplerLightCurves": {
        "train_size": 1024,
        "test_size": 3236,
        "length": 512,
        "n_classes": 2,
        "type": "Univariate",
        "source": "UCR Archive",
    },
}


def _filter_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out error rows from results DataFrame.

    Args:
        df: Results dataframe potentially containing 'error' column

    Returns:
        Filtered dataframe with only successful runs
    """
    if "error" not in df.columns:
        return df.copy()

    # Filter rows where error column is NaN or None (successful runs)
    return df[df["error"].isna()].copy()


def create_dataset_table(output_dir: Path) -> None:
    """
    Create a formatted dataset information table.

    Displays:
    - Dataset name
    - Training size
    - Test size
    - Sequence length
    - Number of classes
    - Type (univariate/multivariate)
    - Source
    """
    data: list[dict[str, Any]] = []
    for dataset_name, metadata in DATASET_METADATA.items():
        data.append(
            {
                "Dataset": dataset_name,
                "Train Size": metadata["train_size"],
                "Test Size": metadata["test_size"],
                "Length": metadata["length"],
                "Classes": metadata["n_classes"],
                "Type": metadata["type"],
                "Source": metadata["source"],
            }
        )

    df = pd.DataFrame(data)

    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")

    # Create table with proper type casting
    table = ax.table(
        cellText=df.values.astype(str).tolist(),
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Header styling
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#4CAF50")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#f0f0f0")
            else:
                cell.set_facecolor("white")

    plt.title("UCR Time Series Classification Datasets", fontsize=16, weight="bold", pad=20)

    # Save figure
    output_file = output_dir / "01_dataset_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Dataset table saved: {output_file}")
    plt.close()


def load_results(results_dir: Path | str) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """
    Load benchmark results from specified or latest run.

    Args:
        results_dir: Path to results directory

    Returns:
        Tuple of (results dataframe, summary dict, run directory path)

    Raises:
        FileNotFoundError: If no results found
    """
    # Convert string to Path if needed
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    # If results_dir is a timestamped run directory, use it directly
    results_file = results_dir / "results.json"
    summary_file = results_dir / "summary.json"

    if results_file.exists():
        # Direct run directory provided
        with open(results_file, "r") as f:
            results = json.load(f)

        summary: dict[str, Any] = {}
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)

        df = pd.DataFrame(results)
        return df, summary, results_dir

    # Otherwise, find latest timestamped directory
    parent_dir = results_dir
    run_dirs = sorted([d for d in parent_dir.iterdir() if d.is_dir() and len(d.name) > 0 and d.name[0].isdigit()])

    if not run_dirs:
        raise FileNotFoundError(f"No benchmark results found in {parent_dir}")

    latest_run = run_dirs[-1]
    results_file = latest_run / "results.json"
    summary_file = latest_run / "summary.json"

    if not results_file.exists():
        raise FileNotFoundError(f"No results.json in {latest_run}")

    with open(results_file, "r") as f:
        results = json.load(f)

    summary = {}
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)

    # Convert to dataframe
    df = pd.DataFrame(results)
    return df, summary, latest_run


def load_multiple_runs(results_dir: Path) -> tuple[pd.DataFrame, Path]:
    """
    Load and combine results from all timestamped run directories.

    Args:
        results_dir: Parent results directory containing timestamped subdirectories

    Returns:
        Tuple of (combined dataframe, parent directory path)

    Raises:
        FileNotFoundError: If no valid runs found
    """
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and len(d.name) > 0 and d.name[0].isdigit()])

    if not run_dirs:
        raise FileNotFoundError(f"No timestamped run directories found in {results_dir}")

    all_dataframes: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        results_file = run_dir / "results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)
            df = pd.DataFrame(results)
            # Add run_id to track which run each result came from
            df["run_id"] = run_dir.name
            all_dataframes.append(df)

    if not all_dataframes:
        raise FileNotFoundError(f"No valid results.json files found in {results_dir}")

    # Combine all runs
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df, results_dir


def create_accuracy_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "accuracy",
    agg_func: Literal["max", "mean"] = "max",
) -> None:
    """
    Create accuracy heatmap (max or mean across runs).

    Args:
        df: Results dataframe with columns: model, dataset, accuracy, f1_macro, etc.
        output_dir: Directory to save plot
        metric: Metric column to visualize (accuracy, f1_macro, etc.)
        agg_func: Aggregation function ('max' or 'mean')
    """
    # Filter valid data
    valid_df = _filter_valid_data(df)

    if valid_df.empty:
        print(f"⚠️ No valid data for {agg_func} {metric} heatmap")
        return

    # Handle both standard and CV results
    if "accuracy_std" in valid_df.columns and agg_func == "mean":
        # Cross-validation results: use accuracy directly (it's already mean)
        pivot_df = valid_df.pivot_table(index="dataset", columns="model", values="accuracy")
    else:
        # Pass string directly - pandas accepts it at runtime despite type stubs
        # Type stubs are overly restrictive; strings work and avoid FutureWarning
        pivot_df = valid_df.pivot_table(
            index="dataset",
            columns="model",
            values=metric,
            aggfunc=agg_func,  # type: ignore[arg-type]
        )

    if pivot_df.empty:
        print(f"⚠️ No data for {agg_func} {metric} heatmap after pivot")
        return

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": metric.replace("_", " ").title()},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    title_prefix = "Maximum" if agg_func == "max" else "Average"
    title = f"{title_prefix} {metric.replace('_', ' ').title()} Across Runs"
    filename = f"02_heatmap_{'max' if agg_func == 'max' else 'avg'}_accuracy.png"

    plt.title(title, fontsize=14, weight="bold", pad=20)
    plt.xlabel("Model", fontsize=12, weight="bold")
    plt.ylabel("Dataset", fontsize=12, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ {title} heatmap saved: {output_file}")
    plt.close()


def create_efficiency_plots(
    df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Create comprehensive efficiency and performance analysis plots.

    Includes:
    1. Model Performance (Box Plot of Accuracy)
    2. Parameter Efficiency (Accuracy vs Parameters)
    3. Memory Efficiency (Accuracy vs Peak Memory)
    4. Running Time Efficiency (Accuracy vs Training Time)
    """
    # Filter valid data
    valid_df = _filter_valid_data(df)

    if valid_df.empty:
        print("⚠️ No valid data for efficiency plots")
        return

    fig = plt.figure(figsize=(20, 16))

    # Define common style for scatter plots
    models = sorted(valid_df["model"].unique())
    cmap_tab = plt.get_cmap("tab10")
    colors_map = cmap_tab(np.linspace(0, 1, len(models)))

    def plot_scatter(ax, x_col, y_col, x_label, y_label, title, log_x=False):
        if x_col in valid_df.columns and y_col in valid_df.columns:
            for idx, model in enumerate(models):
                model_data = valid_df[valid_df["model"] == model]
                ax.scatter(
                    model_data[x_col],
                    model_data[y_col],
                    label=model,
                    s=100,
                    alpha=0.6,
                    color=colors_map[idx],
                    edgecolors="w",
                    linewidth=0.5,
                )

            ax.set_xlabel(x_label, fontsize=12, weight="bold")
            ax.set_ylabel(y_label, fontsize=12, weight="bold")
            ax.set_title(title, fontsize=14, weight="bold", pad=10)
            ax.grid(True, alpha=0.3, linestyle="--")

            if log_x:
                ax.set_xscale("log")

                # Use LogLocator to force ticks at 1, 2, 5 intervals (e.g., 100, 200, 500)
                # Use ScalarFormatter to show plain numbers (100) instead of scientific (10^2)
                from matplotlib.ticker import LogLocator, ScalarFormatter

                ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))

                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                formatter.set_useOffset(False)
                ax.xaxis.set_major_formatter(formatter)
        else:
            ax.text(
                0.5,
                0.5,
                f"Missing data for {title}\nReq: {x_col}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # 1. Classification Performance (Box Plot)
    ax1 = plt.subplot(2, 2, 1)
    if "accuracy" in valid_df.columns:
        valid_df.boxplot(
            column="accuracy", by="model", ax=ax1, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.5)
        )
        ax1.set_title("Model Performance", fontsize=14, weight="bold", pad=10)
        ax1.set_xlabel("Model", fontsize=12, weight="bold")
        ax1.set_ylabel("Accuracy", fontsize=12, weight="bold")
        # Remove automatic title from boxplot
        fig.suptitle("")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")

    # 2. Parameter Efficiency
    ax2 = plt.subplot(2, 2, 2)
    # Convert params to thousands for readability
    if "num_params" in valid_df.columns:
        valid_df["params_k"] = valid_df["num_params"] / 1e3
        plot_scatter(
            ax2, "params_k", "accuracy", "Parameters (Thousands)", "Accuracy", "Parameter Efficiency", log_x=True
        )
        # Add legend only directly to this plot (shared across others by color)
        ax2.legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "Missing 'num_params' data", ha="center", va="center", transform=ax2.transAxes)

    # 3. Memory Efficiency
    ax3 = plt.subplot(2, 2, 3)
    plot_scatter(ax3, "peak_memory_mb", "accuracy", "Peak GPU Memory (MB)", "Accuracy", "Memory Efficiency")

    # 4. Running Time Efficiency
    ax4 = plt.subplot(2, 2, 4)
    plot_scatter(ax4, "training_time", "accuracy", "Training Time (s)", "Accuracy", "Running Time Efficiency")

    plt.tight_layout(pad=3.0)
    output_file = output_dir / "04_efficiency_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Efficiency analysis plot saved: {output_file}")
    plt.close()


def create_model_comparison_table(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a detailed model comparison table with aggregated metrics.

    Includes:
    - Mean accuracy and std dev
    - Mean F1 score
    - Mean training time
    - Number of successful runs
    """
    # Filter valid data
    valid_df = _filter_valid_data(df)

    if valid_df.empty:
        print("⚠️ No valid data for model comparison table")
        return

    summary_stats: list[dict[str, Any]] = []

    for model in sorted(valid_df["model"].unique()):
        model_data = valid_df[valid_df["model"] == model]

        if len(model_data) > 0:
            mean_acc = model_data["accuracy"].mean() if "accuracy" in model_data.columns else 0.0
            std_acc = model_data["accuracy"].std() if "accuracy" in model_data.columns else 0.0
            mean_f1 = model_data["f1_macro"].mean() if "f1_macro" in model_data.columns else 0.0
            mean_time = model_data["training_time"].mean() if "training_time" in model_data.columns else 0.0

            summary_stats.append(
                {
                    "Model": model,
                    "Mean Accuracy": f"{mean_acc:.4f}",
                    "Std Dev": f"{std_acc:.4f}",
                    "Mean F1": f"{mean_f1:.4f}",
                    "Mean Time (s)": f"{mean_time:.1f}",
                    "Runs": len(model_data),
                }
            )

    if not summary_stats:
        print("⚠️ No valid data for model comparison table")
        return

    summary_df = pd.DataFrame(summary_stats)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values.astype(str).tolist(),
        colLabels=list(summary_df.columns),
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Header styling
    for i in range(len(summary_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#2196F3")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#e3f2fd")
            else:
                cell.set_facecolor("white")

    plt.title("Model Performance Summary", fontsize=14, weight="bold", pad=20)

    output_file = output_dir / "05_model_comparison_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Model comparison table saved: {output_file}")
    plt.close()


def create_dataset_performance_table(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a table showing best model per dataset.

    Shows:
    - Best performing model
    - Best accuracy achieved
    - Mean accuracy across all models
    - Number of successful runs
    """
    # Filter valid data
    valid_df = _filter_valid_data(df)

    if valid_df.empty:
        print("⚠️ No valid data for dataset performance table")
        return

    summary_stats: list[dict[str, Any]] = []

    for dataset in sorted(valid_df["dataset"].unique()):
        dataset_data = valid_df[valid_df["dataset"] == dataset]

        if len(dataset_data) > 0:
            best_idx = dataset_data["accuracy"].idxmax() if "accuracy" in dataset_data.columns else None
            best_row = dataset_data.loc[best_idx] if best_idx is not None else None

            if best_row is not None:
                best_acc = best_row["accuracy"] if "accuracy" in best_row else 0.0
                mean_acc = dataset_data["accuracy"].mean() if "accuracy" in dataset_data.columns else 0.0

                summary_stats.append(
                    {
                        "Dataset": dataset,
                        "Best Model": best_row["model"],
                        "Best Accuracy": f"{best_acc:.4f}",
                        "Mean Accuracy": f"{mean_acc:.4f}",
                        "Runs": len(dataset_data),
                    }
                )

    if not summary_stats:
        print("⚠️ No valid data for dataset performance table")
        return

    summary_df = pd.DataFrame(summary_stats)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values.astype(str).tolist(),
        colLabels=list(summary_df.columns),
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Header styling
    for i in range(len(summary_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#FF9800")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#ffe0b2")
            else:
                cell.set_facecolor("white")

    plt.title("Dataset-wise Performance Summary", fontsize=14, weight="bold", pad=20)

    output_file = output_dir / "06_dataset_performance_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Dataset performance table saved: {output_file}")
    plt.close()


def generate_all_visualizations(results_path: str | Path = "results") -> None:
    """
    Generate all visualization plots from benchmark results.

    Args:
        results_path: Path to results directory or specific timestamped run
    """
    results_dir = Path(results_path) if isinstance(results_path, str) else results_path

    print("\n" + "=" * 80)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    try:
        # Check if this is a single run or parent directory
        if (results_dir / "results.json").exists():
            # Single run directory
            df, summary, run_dir = load_results(results_dir)
            viz_dir = run_dir / "visualizations"
            print(f"✅ Loaded single run from: {run_dir}")
            print("⚠️  Note: Average heatmap will be same as max (only one run)")
        else:
            # Parent directory with multiple runs - load all
            df, run_dir = load_multiple_runs(results_dir)
            viz_dir = run_dir / "visualizations_combined"
            print(f"✅ Loaded {len(df['run_id'].unique())} runs from: {run_dir}")
            print(f"   Total records: {len(df)}")

        # Create output directory
        viz_dir.mkdir(exist_ok=True)
        print(f"✅ Output directory: {viz_dir}\n")

        # Generate all plots
        print("Generating plots...")
        create_dataset_table(viz_dir)
        create_accuracy_heatmap(df, viz_dir, metric="accuracy", agg_func="max")
        create_accuracy_heatmap(df, viz_dir, metric="accuracy", agg_func="mean")
        create_efficiency_plots(df, viz_dir)
        create_model_comparison_table(df, viz_dir)
        create_dataset_performance_table(df, viz_dir)

        print(f"\n{'=' * 80}")
        print(f"✅ All visualizations saved to: {viz_dir}")
        print(f"{'=' * 80}\n")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure benchmark has been run and results exist.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during visualization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
