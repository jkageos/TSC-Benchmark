# Time Series Classification Benchmark

A comprehensive neural network benchmarking framework for time series classification (TSC) on UCR datasets. Compares multiple state-of-the-art architectures with support for k-fold cross-validation, test-time augmentation, and automated hyperparameter tuning.

## Architectures

- **Fully Connected Neural Network (FCN)** - Dense layers with batch normalization and dropout
- **Convolutional Neural Network (CNN)** - Temporal convolutions with progressive pooling
- **Transformer** - Multi-head self-attention with positional encoding and feed-forward networks
- **CATS** - Cross-Attention and Temporal Self-Attention for multi-view time series
- **Autoformer** - Auto-correlation attention mechanism with decomposition transformers
- **PatchTST** - Patch-based tokenization with transformer blocks and class token aggregation

## Datasets

Benchmarked on 9 UCR time series classification datasets:

- Adiac
- ArrowHead
- Beef
- Car
- ChlorineConcentration
- CinCECGTorso
- FiftyWords
- HouseTwenty
- KeplerLightCurves

## Features

- **Multiple Training Strategies**

  - Standard train/test split for large datasets (n_train ≥ 300)
  - k-fold cross-validation for small datasets (n_train < 300)
  - Stratified sampling to preserve class distribution

- **Advanced Training Techniques**

  - Mixed precision training (AMP) for faster convergence
  - Learning rate warmup + cosine annealing scheduling
  - Early stopping with configurable patience
  - Model checkpointing to preserve best validation accuracy
  - Stochastic Weight Averaging (SWA) for improved generalization
  - Test-Time Augmentation (TTA) for ensemble-like inference

- **Data Augmentation**

  - Jittering (Gaussian noise injection)
  - Random scaling
  - Magnitude warping
  - Time warping
  - Configurable probability per augmentation type

- **Memory & Performance Optimization**

  - Automatic GPU memory monitoring and capacity reduction
  - `torch.compile` support for inference speedup
  - Flash Attention integration for transformer models (when available)
  - Safe DataLoader worker management based on system resources
  - CPU/GPU load balancing with configurable constraints

- **Hyperparameter Management**
  - YAML-based configuration for all models and training parameters
  - Dataset-specific overrides for fine-tuning per dataset
  - Model-specific capacity reduction for constrained environments
  - Auto-tuning with iterative improvement across multiple rounds

## Setup

```bash
# Install dependencies via uv
uv sync

# Verify CUDA setup (optional)
uv run python test_cuda.py
```

### Requirements

- Python 3.13+
- PyTorch 2.0+ (with CUDA support)
- aeon (for UCR dataset management)
- PyYAML (for configuration)
- scikit-learn (for metrics and cross-validation)
- psutil (for system resource monitoring)

## Running Benchmarks

### Quick Test

Test a single model on the smallest dataset (Beef with 30 samples):

```bash
uv run python main.py
# Set execution.mode = "test" in configs/config.yaml
```

### Full Benchmark Suite

Run all models on all datasets:

```bash
uv run python main.py
# Set execution.mode = "benchmark" in configs/config.yaml
```

### Multiple Benchmark Sessions

Run benchmark repeatedly until interrupted (useful for statistical analysis across runs):

```bash
uv run python main.py -m
# or
uv run python main.py --multiple
```

Press `Ctrl+C` to stop the loop. Each run creates a new timestamped results directory.

### Single Model-Dataset Combination

Debug or profile a specific model/dataset pair:

```bash
uv run python main.py
# Set execution.mode = "single" in configs/config.yaml
# Configure execution.single_model and execution.single_dataset
```

### Auto-Tuning

Iteratively improve hyperparameters based on benchmark results:

```bash
uv run python main.py
# Set execution.mode = "tune" in configs/config.yaml
# Configure execution.tune_rounds
```

## Project Structure

```
TSC-Benchmark/
├── main.py                          # Unified entry point (mode-driven via config)
├── test_cuda.py                     # CUDA availability verification
├── pyproject.toml                   # uv project configuration
│
├── configs/
│   ├── config.yaml                  # Main configuration (hyperparameters, datasets, models)
│
├── src/
│   ├── models/                      # Model implementations
│   │   ├── base.py                  # Abstract base class for all models
│   │   ├── fcn.py                   # Fully Connected Network
│   │   ├── cnn.py                   # Convolutional Neural Network
│   │   ├── transformer.py           # Transformer with self-attention
│   │   ├── cats.py                  # Cross-Attention and Temporal Self-Attention
│   │   ├── autoformer.py            # Auto-correlation Transformer
│   │   └── patchtst.py              # Patch-based Transformer
│   │
│   ├── data/                        # Data loading and preprocessing
│   │   ├── loader.py                # UCR dataset loading and normalization
│   │   └── augmentation.py          # Time series augmentation strategies
│   │
│   ├── training/                    # Training orchestration
│   │   ├── trainer.py               # Main training loop with early stopping
│   │   ├── metrics.py               # Classification metrics computation
│   │   └── tta.py                   # Test-time augmentation
│   │
│   └── utils/                       # Shared utilities
│       ├── config.py                # YAML config loading and model factory
│       ├── system.py                # Resource management and worker recommendations
│       └── logger.py                # Unified logging configuration
│
├── experiments/                     # Benchmarking scripts
│   ├── benchmark.py                 # Main benchmarking orchestrator
│   ├── cross_validate.py            # k-fold cross-validation implementation
│   ├── auto_tune.py                 # Automatic hyperparameter tuning
│   └── __init__.py
│
├── results/                         # Benchmark results and logs
│   ├── .gitkeep
│   ├── visualizations_combined/     # Combined visualizations from all runs (on-demand)
│   │   ├── 01_dataset_table.png
│   │   ├── 02_heatmap_max_accuracy.png
│   │   ├── 03_heatmap_avg_accuracy.png
│   │   ├── 04_efficiency_analysis.png
│   │   ├── 05_model_comparison_table.png
│   │   └── 06_dataset_performance_table.png
│   └── YYYYMMDD_HHMMSS/             # Timestamped result directories
│       ├── results.json             # Per-model accuracy and metrics
│       ├── summary.json             # Aggregated summary by dataset/model
│       ├── config.json              # Config snapshot for reproducibility
│       ├── visualizations/          # Generated plots (single run, on-demand)
│       │   ├── 01_dataset_table.png
│       │   ├── 02_heatmap_max_accuracy.png
│       │   ├── 03_heatmap_avg_accuracy.png
│       │   ├── 04_efficiency_analysis.png
│       │   ├── 05_model_comparison_table.png
│       │   └── 06_dataset_performance_table.png
│       └── checkpoints/             # Model weights
│           ├── {model_name}/        # Standard train/test checkpoints
│           │   └── {dataset_name}/
│           │       ├── best_model.pt
│           │       └── latest.pt
│           └── cross_validation/    # Cross-validation checkpoints
│               └── {model_name}/
│                   └── {dataset_name}/
│                       ├── fold_1/
│                       ├── fold_2/
│                       ├── fold_3/
│                       ├── fold_4/
│                       └── fold_5/
│
├── data/
│   └── cache/                       # Local UCR dataset cache
│
└── README.md                        # This file
```

## Configuration

All behavior is controlled via `configs/config.yaml`:

```yaml
# Execution mode: benchmark, test, tune, or single
execution:
  mode: benchmark
  verbose: true
  test_model: fcn
  test_dataset: Beef
  single_model: transformer
  single_dataset: Adiac
  tune_rounds: 2

# Global training settings
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  patience: 15
  normalize: true
  padding: zero
  max_length: null
  cv_folds: 5
  use_scheduler: true
  warmup_epochs: 5
  use_augmentation: true
  use_tta: false
  use_swa: false
  num_workers: 0

# Hardware configuration
hardware:
  device: cuda
  use_amp: true
  use_compile: true
  compile_mode: default
  max_cpu_load: 0.5
  auto_workers: false
  max_workers_override: null

# Model configurations
models:
  fcn:
    hidden_dims: [512, 256]
    dropout_rate: 0.5
    use_batch_norm: true

  cnn:
    kernel_size: 3
    num_filters: [64, 128, 256]
    dropout_rate: 0.5

  transformer:
    d_model: 256
    num_heads: 8
    num_layers: 4
    d_ff: 1024
    dropout: 0.1

  # ... (cats, autoformer, patchtst similarly configured)

# Dataset-specific overrides (optional)
dataset_overrides:
  Beef:
    batch_size: 16
    epochs: 50
    models:
      transformer:
        d_model: 128
        num_layers: 2

# Results storage
results:
  save_dir: results
  save_checkpoints: true

# Target datasets
datasets:
  - Adiac
  - ArrowHead
  - Beef
  - Car
  - ChlorineConcentration
  - CinCECGTorso
  - FiftyWords
  - HouseTwenty
  - KeplerLightCurves
```

## Key Workflows

### Adding a New Model

1. Create `src/models/your_model.py` inheriting from [`BaseModel`](src/models/base.py)
2. Implement `__init__()` with layer definitions
3. Implement `forward(x: torch.Tensor) -> torch.Tensor` with proper shape handling
4. Add entry in `configs/config.yaml` under `models`
5. Register in [`create_model()`](src/utils/config.py) factory function

Example:

```python
from src.models.base import BaseModel

class YourModel(BaseModel):
    def __init__(self, num_classes: int, input_length: int, input_channels: int = 1, **kwargs):
        super().__init__(num_classes)
        # Initialize layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to (B, T, C)
        # Process through layers
        # Return logits (B, num_classes)
```

### Understanding Data Flow

```
config.yaml
    ↓
load_config() [src/utils/config.py]
    ↓
benchmark.py [experiments/benchmark.py]
    ├→ load_ucr_dataset() [src/data/loader.py]
    │   └→ Normalize + Pad/Truncate → DataLoader
    │
    ├→ create_model() [src/utils/config.py]
    │   └→ Instantiate from registry
    │
    └→ Trainer.train() [src/training/trainer.py]
        ├→ train_epoch() with augmentation
        ├→ validate_epoch() with metrics
        ├→ Early stopping + checkpointing
        └→ Save results
```

### Cross-Validation Workflow

For datasets with < 300 training samples:

1. Load full combined data (train + test)
2. Apply z-score normalization on full dataset
3. Split into k folds with stratified sampling
4. For each fold:
   - Create fresh model
   - Train on k-1 folds
   - Validate on held-out fold
   - Save checkpoint
5. Aggregate metrics across folds

See [`cross_validate.py`](experiments/cross_validate.py) for implementation.

### Auto-Tuning Pipeline

1. **Baseline**: Run initial benchmark with default config
2. **Analysis**: Load summary, identify failures and weak spots
3. **Adjustment**: Apply targeted fixes per dataset/model
   - Reduce capacity for OOM failures
   - Increase capacity for underperforming transformers
   - Truncate sequences aggressively where needed
4. **Re-run**: Execute updated benchmark
5. **Iterate**: Repeat for specified rounds

See [`auto_tune.py`](experiments/auto_tune.py) for details.

## Metrics & Results

### Tracked Metrics

Per epoch:

- Training loss and accuracy
- Validation loss and accuracy
- Validation F1 (macro and weighted)
- Precision and recall per class
- Confusion matrix

Per run:

- Best validation accuracy
- Best validation F1
- Training time
- Dataset information (class distribution, sequence length)
- Model configuration snapshot

### Results Format

```json
{
  "model": "transformer",
  "dataset": "Adiac",
  "accuracy": 0.8234,
  "f1_macro": 0.8101,
  "f1_weighted": 0.8256,
  "precision": 0.8345,
  "recall": 0.8156,
  "training_time": 234.5,
  "best_epoch": 45,
  "dataset_info": {
    "n_train": 390,
    "n_test": 391,
    "n_classes": 37,
    "sequence_length": 176
  }
}
```

Results saved to `results/YYYYMMDD_HHMMSS/results.json`

## System Resource Management

The framework automatically:

- Detects GPU memory and reduces model capacity if needed
- Recommends safe number of DataLoader workers
- Monitors CPU load and adjusts parallelization
- Handles OOM gracefully with capacity reduction and retry logic
- Respects `max_cpu_load` configuration for system stability

See [`system.py`](src/utils/system.py) for resource monitoring utilities.

## Performance Considerations

- **GPU Memory**: Use `use_amp=true` for mixed precision (faster, lower memory)
- **Inference Speed**: Enable `use_compile=true` for torch.compile speedup on NVIDIA GPUs
- **Data Loading**: Set `num_workers` based on dataset size (auto-recommended if `auto_workers=true`)
- **Attention Efficiency**: Flash Attention automatically used for transformer models when CUDA permits
- **Augmentation Overhead**: Disable `use_augmentation` for faster training if convergence is stable

## Known Limitations

- **CUDA Required**: Current implementation requires NVIDIA GPU with CUDA
- **Single GPU**: No distributed training support (single device only)
- **Sequence Handling**: Standardized padding/truncation; variable-length handling via masking not yet implemented
- **No Ensemble Methods**: Individual models only; no voting/stacking

## References

This implementation is informed by:

1. **CATS** - https://github.com/dongbeank/CATS
2. **Autoformer** - https://github.com/thuml/Autoformer
3. **Time-Series-Library** - https://github.com/thuml/Time-Series-Library
4. **PatchTST** - https://github.com/yuqinie98/PatchTST
5. **NN Standard Architectures HPS** - https://github.com/carmelyr/nn_standard_architecture_hps

## Development

### Running Tests

Run all tests in the tests folder (requires CUDA):

```bash
uv run pytest -v tests/
```

Run a single test:

```bash
uv run pytest tests/test_cuda.py::test_cuda_device -q
```

In Visual Studio Code, use the Test Explorer (pytest) to discover and run tests in [tests/test_cuda.py](tests/test_cuda.py).

If CUDA is not available, tests in [tests/test_cuda.py](tests/test_cuda.py) will fail. Verify CUDA first:

```bash
uv run python tests/test_cuda.py
```

### Code Quality

```bash
# Format code
black src/ experiments/

# Lint
ruff check src/ experiments/
```

### Troubleshooting

**CUDA Not Available**

```bash
uv run python test_cuda.py
# Install CUDA-enabled PyTorch if needed
```

**Out of Memory (OOM)**

- Reduce `batch_size` in config
- Set `use_amp: true` for mixed precision
- Enable auto-capacity reduction (automatic on low memory)
- Reduce `max_length` for sequence truncation

**Poor Model Performance**

- Increase `epochs` or reduce `patience`
- Tune `learning_rate` and `warmup_epochs`
- Enable `use_augmentation: true`
- Increase model capacity (`d_model`, `num_layers`, etc.)

## Generate Visualizations

See [VISUALIZATIONS.md](VISUALIZATIONS.md) for detailed visualization guide.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
