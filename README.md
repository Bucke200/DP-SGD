# Empirical Evaluation of Differential Privacy in Machine Learning under Membership Inference Attacks

Final-year undergraduate research project evaluating the empirical relationship between **Differential Privacy (DP-SGD)**, **model utility**, and **vulnerability to Membership Inference Attacks (MIA)** on MNIST using PyTorch and Opacus.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Automated Testing](#2-automated-testing)
  - [3. Objective 1: Baseline & DP-SGD Pipelines](#3-objective-1-baseline--dp-sgd-pipelines)
  - [4. Objective 2: Multi-ε Sweep & Verification](#4-objective-2-multi-ε-sweep--verification)
- [Configuration](#configuration)
- [Experimental Results](#experimental-results)
  - [Multi-ε Sweep Summary](#multi-ε-sweep-summary)
  - [Privacy–Utility Tradeoff Curve](#privacyutility-tradeoff-curve)
  - [Checkpoint Format & MIA Readiness](#checkpoint-format--mia-readiness)
- [Research Objectives & Roadmap](#research-objectives--roadmap)

---

## Project Overview

This project investigates how privacy-preserving training changes model utility and resistance against adversarial membership inference. 

Key technical components:
- **Framework**: PyTorch + Opacus
- **Dataset**: MNIST (60,000 train, 10,000 test; normalized with mean 0.1307, std 0.3081)
- **Model Architecture**: `SampleCNN` (Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Linear → ReLU → Linear; BatchNorm-free for efficient per-sample gradient computation and full Opacus compatibility)
- **Privacy Engine**: DP-SGD with Rényi Differential Privacy (RDP) accountant, per-sample gradient clipping ($C=1.0$), and calibrated Gaussian noise injection.
- **Decoupled Inference**: Checkpoints unwrap Opacus `GradSampleModule` (`model._module.state_dict()`) into clean, standalone PyTorch models for downstream MIA evaluation without Opacus runtime dependencies.

---

## Repository Structure

```text
dp-sgd/
├── README.md                           # Project documentation & usage guide
├── PROJECT_CONTEXT.md                  # Persistent research context & evidence boundaries
├── config.py                           # Central configuration (hyperparameters & DP settings)
├── requirements.txt                    # Python dependencies
├── src/
│   ├── __init__.py
│   ├── dataset.py                      # MNIST data loaders & transforms
│   ├── model.py                        # SampleCNN architecture & Opacus validation
│   ├── evaluate.py                     # Standard evaluation routines (loss, accuracy)
│   ├── utils.py                        # Seed management & device selection
│   ├── train_baseline.py               # Objective 1: Standard SGD baseline
│   ├── train_dp.py                     # Objective 1: Single-run DP-SGD pipeline
│   ├── sweep_epsilon.py                # Objective 2: Multi-noise-multiplier sweep & checkpointing
│   ├── verify_checkpoints.py           # Objective 2: Standalone checkpoint loading & validation
│   └── plot_curves.py                  # Objective 2: Privacy–utility curve visualization
├── experiments/
│   ├── checkpoints/                    # Saved model checkpoints (.pt)
│   │   ├── baseline_seed42.pt
│   │   ├── dp_nm_0.30_seed42.pt
│   │   ├── dp_nm_0.50_seed42.pt
│   │   ├── dp_nm_0.70_seed42.pt
│   │   ├── dp_nm_0.90_seed42.pt
│   │   ├── dp_nm_1.10_seed42.pt
│   │   ├── dp_nm_1.50_seed42.pt
│   │   ├── dp_nm_2.00_seed42.pt
│   │   ├── dp_nm_3.00_seed42.pt
│   │   └── dp_nm_5.00_seed42.pt
│   └── results/
│       ├── epsilon_sweep.json          # Experimental metadata manifest
│       ├── objective_1_verification.md # Objective 1 verification report
│       └── privacy_utility_curve.png   # Generated privacy–utility tradeoff plot
└── tests/
    └── test_pipeline.py                # Automated pytest unit test suite
```

---

## Quick Start

### 1. Environment Setup

Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Automated Testing

Run the full pytest suite to verify dataset loading, model forward passes, Opacus compatibility, and checkpoint serialization:

```powershell
python -m pytest tests/test_pipeline.py -v
```

### 3. Objective 1: Baseline & DP-SGD Pipelines

Run a standalone non-private SGD baseline:
```powershell
python -m src.train_baseline
```

Run a standalone DP-SGD training run with the default noise multiplier ($\sigma=1.1$):
```powershell
python -m src.train_dp
```

### 4. Objective 2: Multi-ε Sweep & Verification

#### Execute the Full Multi-ε Sweep
Trains 10 models (1 baseline + 9 DP-SGD across noise multipliers `[0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0]`):

```powershell
# Single seed (default: config.SEED = 42)
python -m src.sweep_epsilon

# Multi-seed for aggregated statistics
python -m src.sweep_epsilon --seeds 42 123 256
```

#### Verify Checkpoints
Validates that all checkpoints load into clean `SampleCNN` instances without Opacus dependencies and match recorded test accuracy:

```powershell
python -m src.verify_checkpoints
```

#### Plot Privacy–Utility Tradeoff
Generates the privacy–utility tradeoff curve with logarithmic $\epsilon$ scaling:

```powershell
# Generate and save PNG to experiments/results/privacy_utility_curve.png
python -m src.plot_curves

# Optionally display interactive window
python -m src.plot_curves --show
```

---

## Configuration

All central parameters are defined in [`config.py`](config.py):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `SEED` | `42` | Global random seed for reproducibility |
| `DATA_DIR` | `"./data"` | Root path for MNIST dataset storage |
| `BATCH_SIZE` | `64` | Training batch size |
| `TEST_BATCH_SIZE` | `1000` | Evaluation batch size |
| `EPOCHS` | `5` | Training epochs per model |
| `LEARNING_RATE` (`LR`) | `0.05` | SGD learning rate |
| `MOMENTUM` | `0.0` | SGD momentum (0.0 for clean DP clipping baseline) |
| `MAX_GRAD_NORM` | `1.0` | Per-sample gradient clipping bound ($C$) |
| `NOISE_MULTIPLIER` | `1.1` | Default Gaussian noise multiplier ($\sigma$) |
| `DELTA` | `1e-5` | Target DP failure probability ($\delta$) |

---

## Experimental Results

### Multi-ε Sweep Summary

Trained under 5 epochs, batch size 64, learning rate 0.05, clipping norm 1.0, $\delta = 10^{-5}$ (Seed 42):

| Model Type | Noise Multiplier ($\sigma$) | Privacy Guarantee ($\epsilon, \delta=10^{-5}$) | Test Accuracy | Test Loss | Training Time | Checkpoint File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | 0.0 | $\infty$ (No DP) | **99.05%** | 0.0309 | 72.8s | `baseline_seed42.pt` |
| **DP-SGD** | 0.3 | $\epsilon = 31.3806$ | **91.30%** | 0.4661 | 107.5s | `dp_nm_0.30_seed42.pt` |
| **DP-SGD** | 0.5 | $\epsilon = 4.5028$ | **91.28%** | 0.4638 | 107.8s | `dp_nm_0.50_seed42.pt` |
| **DP-SGD** | 0.7 | $\epsilon = 1.0337$ | **91.09%** | 0.4685 | 106.9s | `dp_nm_0.70_seed42.pt` |
| **DP-SGD** | 0.9 | $\epsilon = 0.4402$ | **90.76%** | 0.4847 | 107.6s | `dp_nm_0.90_seed42.pt` |
| **DP-SGD** | 1.1 | $\epsilon = 0.3000$ | **90.31%** | 0.5226 | 107.3s | `dp_nm_1.10_seed42.pt` |
| **DP-SGD** | 1.5 | $\epsilon = 0.1905$ | **89.11%** | 0.6351 | 107.6s | `dp_nm_1.50_seed42.pt` |
| **DP-SGD** | 2.0 | $\epsilon = 0.1336$ | **86.89%** | 0.8765 | 107.3s | `dp_nm_2.00_seed42.pt` |
| **DP-SGD** | 3.0 | $\epsilon = 0.0858$ | **80.90%** | 1.7463 | 113.6s | `dp_nm_3.00_seed42.pt` |
| **DP-SGD** | 5.0 | $\epsilon = 0.0522$ | **68.45%** | 5.3000 | 113.0s | `dp_nm_5.00_seed42.pt` |

### Privacy–Utility Tradeoff Curve

The generated curve ([`experiments/results/privacy_utility_curve.png`](experiments/results/privacy_utility_curve.png))
![Privacy-Utility Curve](./experiments/results/privacy_utility_curve.png)
highlights:
- **Strong Privacy Regime ($\epsilon < 1.0$)**: Maintains ~89–91% accuracy down to $\epsilon \approx 0.19$, demonstrating strong utility retention for MNIST under DP-SGD.
- **Moderate Privacy Regime ($1.0 \le \epsilon \le 10.0$)**: Accuracy plateaus at ~91.3% (within ~7.8 percentage points of the non-private baseline).
- **Extreme Noise Regime ($\sigma \ge 3.0, \epsilon < 0.1$)**: Utility degrades noticeably (80.9% at $\sigma=3.0$, 68.5% at $\sigma=5.0$) due to heavy gradient perturbation.

### Checkpoint Format & MIA Readiness

Every checkpoint in `experiments/checkpoints/` contains:
```python
{
    "model_state_dict": ...,       # Clean PyTorch weights (loads directly into SampleCNN)
    "model_type": "dp-sgd",        # "baseline" or "dp-sgd"
    "epsilon": 0.3000,             # Derived from accountant ("inf" for baseline)
    "delta": 1e-05,                # Target delta
    "noise_multiplier": 1.1,       # Noise scale parameter
    "max_grad_norm": 1.0,          # Gradient clipping threshold
    "epochs": 5,                   # Training epochs
    "batch_size": 64,              # Batch size
    "learning_rate": 0.05,         # Learning rate
    "seed": 42,                    # Random seed
    "test_accuracy": 90.31,        # Evaluation accuracy
    "test_loss": 0.5226,           # Evaluation loss
    "training_time_sec": 107.28,   # Elapsed training time
}
```

All 10 checkpoints pass `verify_checkpoints.py` with 100% accuracy verification match and full metadata validation.

---

## Research Objectives & Roadmap

- [x] **Objective 1: DP-SGD Baseline Implementation & Opacus Verification**
  - Standard SGD vs DP-SGD training comparison.
  - Automated unit tests for per-sample gradients and Opacus privacy accounting.
- [x] **Objective 2: Multi-ε Privacy Sweep & Model Checkpointing**
  - Sweep over noise multipliers ($\sigma \in [0.3, 5.0]$) producing models from $\epsilon \approx 0.05$ to $\epsilon \approx 31.38$.
  - Standalone checkpoint serialization and verification pipeline.
  - Automated privacy–utility tradeoff visualization.
- [ ] **Objective 3: Membership Inference Attack (MIA) Framework**
  - Target model evaluation on member vs non-member splits.
  - Shadow model training, feature extraction, and attack classifiers.
  - Empirical MIA metrics (Attack Accuracy, ROC-AUC) across privacy budgets.
- [ ] **Objective 4: Privacy–Utility–Security Tradeoff Analysis**
  - Joint correlation analysis connecting formal $\epsilon$ guarantees, model utility, and empirical MIA resistance.
