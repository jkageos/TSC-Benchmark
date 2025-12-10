# Time Series Classification Benchmark

A comprehensive benchmark of neural network architectures for time series classification using UCR datasets.

## Architectures

- **Fully Connected Neural Network (FCN)** - Dense layers for feature extraction
- **Convolutional Neural Network (CNN)** - Convolutional layers for temporal patterns
- **Transformer** - Self-attention and cross-attention mechanisms

## Datasets

Benchmarked on UCR time series classification datasets:
- Adiac, ArrowHead, Beef, Car, ChlorineConcentration
- CinCECGTorso, FiftyWords, HouseTwenty, KeplerLightCurves

## Setup

```bash
uv sync
```

## Running Benchmarks

```bash
uv run python experiments/benchmark.py
```

## Project Structure

```
src/
├── data/          # Dataset loading
├── models/        # Model implementations
├── training/      # Training loops and metrics
└── utils/         # Configuration and utilities
experiments/
├── benchmark.py   # Main benchmarking script
configs/
├── config.yaml    # Hyperparameter configurations
results/           # Benchmark results and logs
```