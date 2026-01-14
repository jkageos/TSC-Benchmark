# TSC-Benchmark Visualizations

This guide explains how to generate publication-quality visualizations from benchmark results.

## Quick Start

### Generate visualizations from latest results:

```bash
uv run python main.py -p
```

### Generate visualizations from specific run:

```bash
# From a specific timestamped directory
uv run python main.py -p results/20251212_001253

# Or from results root (auto-finds latest)
uv run python main.py --plot results
```

### Full command with all options:

```bash
# Long form
uv run python main.py --plot results/20251212_001253

# Short form
uv run python main.py -p results/20251212_001253

# Use latest automatically
uv run python main.py -p
```

## Generated Plots

All visualizations are saved to `{results_directory}/visualizations/`

### 1. Dataset Information Table (`01_dataset_table.png`)

A comprehensive table showing UCR dataset characteristics:

- **Train Size**: Number of training samples
- **Test Size**: Number of test samples
- **Length**: Sequence length (time steps)
- **Classes**: Number of classification classes
- **Type**: Univariate or multivariate
- **Source**: Dataset source (UCR Archive)

**Usage**: Include in methodology section to describe benchmarking datasets.

### 2. Maximum Accuracy Heatmap (`02_heatmap_max_accuracy.png`)

Shows the best (maximum) validation accuracy achieved for each model-dataset combination.

- **Rows**: Datasets (Adiac, ArrowHead, etc.)
- **Columns**: Models (FCN, CNN, Transformer, CATS, Autoformer, PatchTST)
- **Cell Values**: Best accuracy across all runs (0-1 scale)
- **Color Scheme**: Red-Yellow-Green (RYG) for intuitive interpretation

**Interpretation**: Identifies which models perform best on specific datasets. Darker green = higher accuracy.

**Usage**: Primary results figure showing raw performance.

### 3. Average Accuracy Heatmap (`03_heatmap_avg_accuracy.png`)

Shows the average validation accuracy across runs for each model-dataset combination.

- Same structure as max heatmap
- Cell values: Mean accuracy

**Interpretation**: Shows consistent (stable) performance vs. best-case performance. Compare with max heatmap to assess variance.

**Usage**: Supplementary figure showing robustness and consistency.

### 4. Efficiency Analysis (`04_efficiency_analysis.png`)

Four-panel plot analyzing model efficiency:

#### Panel 1: Classification Performance by Model

- Box plot of accuracy across all datasets per model
- Shows distribution, median, and outliers

#### Panel 2: Accuracy Distribution by Model

- Violin plot showing full probability density
- Reveals bimodal or skewed distributions

#### Panel 3: Training Time Efficiency

- Bar chart with error bars (mean ± std)
- Color-coded by performance (red = slow, green = fast)

#### Panel 4: Performance vs. Training Time

- Scatter plot with each point = one model-dataset run
- X-axis: Training time (seconds)
- Y-axis: Accuracy
- Color-coded by model
- Identifies Pareto-optimal models (fast AND accurate)

**Usage**: Efficiency section to discuss computational cost vs. performance tradeoff.

### 5. Model Comparison Table (`05_model_comparison_table.png`)

Summary statistics per model:

- **Mean Accuracy**: Average across all datasets
- **Std Dev**: Standard deviation (stability)
- **Mean F1**: Macro F1-score
- **Mean Time**: Average training time per dataset
- **Runs**: Number of successful benchmark runs

**Interpretation**: Quick summary of which models are most reliable and efficient.

**Usage**: Results section as a reference table.

### 6. Dataset Performance Table (`06_dataset_performance_table.png`)

Per-dataset summary showing:

- **Best Model**: Which model achieved highest accuracy
- **Best Accuracy**: Maximum accuracy on that dataset
- **Mean Accuracy**: Average across all models
- **Runs**: Number of successful runs

**Interpretation**: Shows which datasets favor which models, and identifies easy vs. hard datasets (by mean accuracy).

**Usage**: Supplementary results to highlight dataset-specific insights.

## Customization

### Modifying Visualization Parameters

Edit `src/utils/visualizations.py`:

```python
# Change heatmap colormap
sns.heatmap(..., cmap="viridis")  # instead of "RdYlGn"

# Adjust figure sizes
fig = plt.figure(figsize=(16, 12))  # (width, height) in inches

# Change output DPI (quality)
plt.savefig(output_file, dpi=600)  # 600 DPI for publication
```

### Adding New Datasets

Update `DATASET_METADATA` dictionary:

```python
DATASET_METADATA = {
    "YourDataset": {
        "train_size": 1000,
        "test_size": 500,
        "length": 300,
        "n_classes": 10,
        "type": "Univariate",
        "source": "Your Source",
    },
    # ... existing datasets
}
```

### Custom Color Schemes

```python
# For heatmaps
cmap = "coolwarm"    # Blue-red diverging
cmap = "viridis"     # Perceptually uniform
cmap = "RdYlGn"      # Red-Yellow-Green (default)
cmap = "Blues"       # Sequential
```

## Integration with LaTeX/Papers

### Embedding Plots in Overleaf

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.95\textwidth]{02_heatmap_max_accuracy.png}
    \caption{Maximum validation accuracy heatmap across 6 models and 9 UCR datasets.}
    \label{fig:max_accuracy_heatmap}
\end{figure}
```

### Recommended Figure Order

1. Dataset table (Fig 1)
2. Max accuracy heatmap (Fig 2)
3. Model comparison table (Table 1)
4. Efficiency analysis (Fig 3)
5. Average accuracy heatmap (Appendix Fig A)
6. Dataset performance table (Appendix Table A)

## Troubleshooting

### "No benchmark results found"

```bash
# Ensure benchmark was run first
uv run python main.py
# Then generate visualizations
uv run python main.py -p
```

### "No results.json found"

Check that your results directory structure is correct:

```
results/
└── YYYYMMDD_HHMMSS/
    ├── results.json        # Required
    ├── summary.json        # Optional but recommended
    └── config.json
```

### "No valid data for X table"

This occurs when all runs had errors. Check:

1. Benchmark completed successfully
2. `results.json` exists in the run directory
3. At least one model-dataset combination succeeded

### High-resolution PNG not rendering

Use PDF format instead (publication-quality):

```python
plt.savefig(output_file.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
```

## Performance Metrics Explained

### Accuracy

Classification accuracy on test set: $\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$

### F1 Macro

Unweighted average F1 across all classes. Useful for imbalanced datasets.

### Training Time

Wall-clock time for full training (including data loading, validation, checkpointing).

### Standard Deviation

Stability across different datasets or random seeds. Lower = more consistent model.

## Citation

If using these visualizations in publications:

```bibtex
@software{tsc_benchmark_2026,
  title={TSC-Benchmark: Comprehensive Neural Network Benchmarking for Time Series Classification},
  year={2026},
  note={Visualizations generated using src/utils/visualizations.py}
}
```
