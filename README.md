# Empirical Evaluation of Differential Privacy in Machine Learning under Membership Inference Attacks

Undergraduate research project evaluating the empirical relationship between **Differential Privacy (DP-SGD)**, **model utility**, and **vulnerability to Membership Inference Attacks (MIA)** across MNIST and CIFAR-10 using PyTorch and Opacus.

MNIST serves as a control case (the model generalizes too well for MIA to find signal). CIFAR-10 is the primary evaluation dataset, where a subsampled training set and extended training produce a natural memorization gap that enables meaningful MIA results.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Experimental Results](#experimental-results)
  - [MNIST (Control Case)](#mnist-control-case)
  - [CIFAR-10 (Primary Evaluation)](#cifar-10-primary-evaluation)
  - [Ablation: Clipping vs Noise](#ablation-clipping-vs-noise)
- [Checkpoint Format](#checkpoint-format)
- [Research Objectives & Roadmap](#research-objectives--roadmap)

---

## Repository Structure

```text
dp-sgd/
├── README.md
├── config.py                           # Central configuration (hyperparameters, DP settings, paths)
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py                      # Original MNIST loaders (legacy, kept for compatibility)
│   ├── data_split.py                   # Dispatcher: routes to MNIST or CIFAR-10 subsampled splits, saves member indices
│   ├── model.py                        # SampleCNN (MNIST), CifarCNN (CIFAR-10), get_model() dispatcher
│   ├── evaluate.py                     # Model evaluation (loss & accuracy)
│   ├── utils.py                        # Seeding & device detection
│   ├── train_baseline.py               # Standard SGD baseline training
│   ├── train_dp.py                     # DP-SGD training with Opacus PrivacyEngine
│   ├── sweep_epsilon.py                # Multi-noise-multiplier sweep with checkpointing
│   ├── verify_checkpoints.py           # Checkpoint loading verification
│   ├── threshold_attack.py             # Loss-threshold MIA evaluation
│   ├── run_ablation.py                 # Clipping vs noise ablation study
│   ├── plot_curves.py                  # Privacy-utility curve plotting
│   └── plot_mia_curves.py              # MIA AUC and ROC curve plotting
├── experiments/
│   ├── mnist/
│   │   ├── checkpoints/
│   │   ├── splits/
│   │   └── results/
│   └── cifar10/
│       ├── checkpoints/
│       ├── splits/
│       └── results/
└── tests/
    └── test_pipeline.py                # 17 passing tests
```

Output paths are dataset-scoped: all checkpoints, splits, and results go under `experiments/{dataset}/`. Checkpoint naming convention: `baseline_seed{N}.pt` for baselines, `dp_nm_{sigma}_seed{N}.pt` for DP-SGD models.

---

## Quick Start

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

**Dependencies**: torch, torchvision, opacus, numpy, matplotlib, scikit-learn, pytest

### Run Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

### Full Pipeline

Set `DATASET` in `config.py` to `"mnist"` or `"cifar10"`, then:

```bash
python -m src.sweep_epsilon                     # Train baseline + 9 DP-SGD models
python -m src.sweep_epsilon --seeds 42 123 256  # Multi-seed variant
python -m src.verify_checkpoints                # Verify checkpoint integrity
python -m src.threshold_attack                  # Run MIA evaluation
python -m src.plot_curves                       # Privacy-utility curve
python -m src.plot_mia_curves                   # MIA AUC plots
```

---

## Configuration

All parameters are defined in [`config.py`](config.py):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `DATASET` | `"cifar10"` | Active dataset (`"mnist"` or `"cifar10"`) |
| `SEED` | `42` | Global random seed |
| `N_TRAIN` | `5000` | Subsampled training set size |
| `EPOCHS` | `50` | Training epochs per model |
| `BATCH_SIZE` | `64` | Training batch size |
| `TEST_BATCH_SIZE` | `1000` | Evaluation batch size |
| `LR` | `0.05` | SGD learning rate |
| `MOMENTUM` | `0.0` | SGD momentum (0.0 for clean DP clipping baseline) |
| `MAX_GRAD_NORM` | `1.0` | Per-sample gradient clipping bound (C) |
| `NOISE_MULTIPLIER` | `1.1` | Default Gaussian noise multiplier (sigma) |
| `DELTA` | `1e-5` | Target DP failure probability (delta) |
| `MNIST_MEAN / STD` | `0.1307 / 0.3081` | MNIST channel normalization |
| `CIFAR10_MEAN` | `[0.4914, 0.4822, 0.4465]` | CIFAR-10 per-channel mean |
| `CIFAR10_STD` | `[0.2470, 0.2435, 0.2616]` | CIFAR-10 per-channel std |

Noise multipliers swept: `[0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0]`

---

## Model Architectures

Both architectures are BatchNorm-free and pass the Opacus `ModuleValidator` for per-sample gradient computation.

**SampleCNN** (MNIST, 28x28x1):
`Conv2d(1,16,5) -> ReLU -> MaxPool2d(2) -> Conv2d(16,32,5) -> ReLU -> MaxPool2d(2) -> Linear(512,128) -> ReLU -> Linear(128,10)`

**CifarCNN** (CIFAR-10, 32x32x3):
`Conv2d(3,32,3,pad=1) -> ReLU -> Conv2d(32,32,3,pad=1) -> ReLU -> MaxPool2d(2) -> Conv2d(32,64,3,pad=1) -> ReLU -> Conv2d(64,64,3,pad=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(64*8*8,256) -> ReLU -> Linear(256,10)`

The `get_model()` dispatcher in `src/model.py` selects the architecture based on the active dataset. Checkpoints unwrap the Opacus `GradSampleModule` wrapper (`model._module.state_dict()`) so saved weights load directly into clean PyTorch models without Opacus runtime dependencies.

---

## Experimental Results

### MNIST (Control Case)

Trained on 60k samples, 5 epochs, batch size 64, LR 0.05, clipping norm 1.0, delta = 1e-5 (seed 42):

| Model | sigma | epsilon | Test Accuracy | Test Loss | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | 0.0 | inf | **99.05%** | 0.0309 | 72.8s |
| DP-SGD | 0.3 | 31.3806 | 91.30% | 0.4661 | 107.5s |
| DP-SGD | 0.5 | 4.5028 | 91.28% | 0.4638 | 107.8s |
| DP-SGD | 0.7 | 1.0337 | 91.09% | 0.4685 | 106.9s |
| DP-SGD | 0.9 | 0.4402 | 90.76% | 0.4847 | 107.6s |
| DP-SGD | 1.1 | 0.3000 | 90.31% | 0.5226 | 107.3s |
| DP-SGD | 1.5 | 0.1905 | 89.11% | 0.6351 | 107.6s |
| DP-SGD | 2.0 | 0.1336 | 86.89% | 0.8765 | 107.3s |
| DP-SGD | 3.0 | 0.0858 | 80.90% | 1.7463 | 113.6s |
| DP-SGD | 5.0 | 0.0522 | 68.45% | 5.3000 | 113.0s |

MNIST maintains 89-91% accuracy down to epsilon ~ 0.19. Threshold MIA AUC ~ 0.49 across all epsilon values, indistinguishable from random guessing. The model generalizes too well on MNIST for membership inference to find any signal, which is why it serves as a control rather than the primary evaluation dataset.

The generated curve ([`experiments/mnist/results/privacy_utility_curve.png`](experiments/mnist/results/privacy_utility_curve.png))
![Privacy-Utility Curve](./experiments/mnist/results/privacy_utility_curve.png)
highlights:
- **Strong Privacy Regime ($\epsilon < 1.0$)**: Maintains ~89–91% accuracy down to $\epsilon \approx 0.19$, demonstrating strong utility retention for MNIST under DP-SGD.
- **Moderate Privacy Regime ($1.0 \le \epsilon \le 10.0$)**: Accuracy plateaus at ~91.3% (within ~7.8 percentage points of the non-private baseline).
- **Extreme Noise Regime ($\sigma \ge 3.0, \epsilon < 0.1$)**: Utility degrades noticeably (80.9% at $\sigma=3.0$, 68.5% at $\sigma=5.0$) due to heavy gradient perturbation.

### CIFAR-10 (Primary Evaluation)

Trained on a 5k subsampled training set for 50 epochs to induce controlled overfitting. The smaller training set and longer training create a natural memorization gap (+49.30%), giving the threshold MIA a realistic signal to exploit.

| Model | Noise Mult ($\sigma$) | $\epsilon$ ($\delta=10^{-5}$) | Test Accuracy | Member Acc | Non-Member Acc | Gen Gap | Loss Attack AUC | Attack Accuracy | TPR @ 1% FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.00 | $\infty$ | **50.96%** | 100.00% | 50.70% | **+49.30%** | **0.8535** | **85.07%** | 0.0287 |
| **DP-SGD** | 0.30 | 193.50 | 37.42% | 39.48% | 38.00% | +1.48% | **0.5151** | 51.63% | 0.0102 |
| **DP-SGD** | 0.50 | 32.53 | 37.26% | 39.22% | 37.72% | +1.50% | **0.5146** | 51.60% | 0.0092 |
| **DP-SGD** | 0.70 | 11.29 | 37.28% | 38.86% | 37.48% | +1.38% | **0.5129** | 51.55% | 0.0090 |
| **DP-SGD** | 0.90 | 6.05 | 37.20% | 38.36% | 37.46% | +0.90% | **0.5124** | 51.26% | 0.0092 |
| **DP-SGD** | 1.10 | 4.09 | 37.14% | 39.14% | 37.50% | +1.64% | **0.5109** | 51.23% | 0.0102 |
| **DP-SGD** | 1.50 | 2.51 | 36.05% | 38.74% | 36.30% | +2.44% | **0.5151** | 51.64% | 0.0076 |
| **DP-SGD** | 2.00 | 1.71 | 31.15% | 34.40% | 31.00% | +3.40% | **0.5167** | 51.80% | 0.0120 |
| **DP-SGD** | 3.00 | 1.05 | 19.97% | 20.80% | 20.64% | +0.16% | **0.5023** | 50.64% | 0.0112 |
| **DP-SGD** | 5.00 | 0.59 | 15.47% | 16.88% | 15.46% | +1.42% | **0.5048** | 50.91% | 0.0107 |

#### Visualizations (CIFAR-10)

- **Privacy-Utility-Attack Tradeoff:**
  ![Privacy-Utility-Attack Tradeoff](./experiments/cifar10/results/privacy_utility_attack.png)
- **MIA ROC Curves (Linear & Log-Log):**
  ![MIA ROC Curves](./experiments/cifar10/results/mia_roc_curves.png)
- **Attack AUC vs Privacy Budget ($\epsilon$):**
  ![MIA AUC vs Epsilon](./experiments/cifar10/results/mia_auc_vs_epsilon.png)

### Ablation: Clipping vs Noise

To isolate the individual contribution of **per-sample gradient clipping** from **Gaussian noise addition**, an ablation experiment was conducted with clipping norm $C = 1.0$ and noise multiplier $\sigma = 0.0$:

| Condition | Clipping ($C$) | Noise ($\sigma$) | Formal Privacy ($\epsilon$) | Test Accuracy | Train Loss | Generalization Gap | MIA Attack AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Baseline** | None | 0.0 | $\infty$ | **50.96%** | **0.0002** | **+49.30%** | **0.8535** |
| **Ablation (Clip Only)** | **1.0** | **0.0** | $\infty$ | **36.10%** | **1.8997** | **+2.20%** | **0.5175** |
| **Full DP-SGD** | **1.0** | **1.1** | **4.09** | **37.14%** | **1.9494** | **+1.64%** | **0.5109** |

#### Key Takeaways:
1. **Clipping suppresses memorization:** Restricting the per-sample gradient norm alone prevents individual training instances from dominating parameter updates, reducing the generalization gap from `+49.30%` to `+2.20%` and lowering threshold MIA AUC from `0.8535` to `0.5175`.
2. **Noise provides formal mathematical privacy:** While clipping acts as an implicit regularizer that destroys simple threshold MIA signal, it provides $\epsilon = \infty$ (zero formal differential privacy). Calibrated noise injection ($\sigma = 1.1$) guarantees formal $(\epsilon=4.09, \delta=10^{-5})$-DP with minimal additional utility loss beyond clipping.

---

## Checkpoint Format

Every checkpoint stores a self-contained metadata dict for downstream MIA evaluation:

```python
{
    "model_state_dict": ...,
    "model_type": "baseline" | "dp-sgd",
    "epsilon": float | "inf",
    "delta": 1e-5 | None,
    "noise_multiplier": float,
    "max_grad_norm": float,
    "epochs": int,
    "batch_size": int,
    "learning_rate": float,
    "seed": int,
    "test_accuracy": float,
    "test_loss": float,
    "training_time_sec": float,
}
```

Weights load directly into the appropriate model class (`SampleCNN` or `CifarCNN`) without Opacus dependencies. Use `src/verify_checkpoints.py` to validate checkpoint integrity and accuracy match.

---

## Research Objectives & Roadmap

- [x] **Objective 1**: DP-SGD baseline implementation and Opacus verification (MNIST + CIFAR-10)
- [x] **Objective 2**: Multi-epsilon privacy sweep with model checkpointing (both datasets)
- [x] **Objective 3**: Threshold Membership Inference Attack (MIA) framework and empirical evaluation (MNIST + CIFAR-10)
- [x] **Objective 4**: Privacy–Utility–Security tradeoff analysis and gradient clipping ablation
- [ ] **Objective 5 (Next Steps)**: Shadow model MIA and Likelihood Ratio Attack (LiRA) framework to probe subtle clipping vs noise representations

### Planned Extensions
- **Clipping-norm sweep**: $C \in \{0.5, 1, 5, 10, 50\}$ at $\sigma=0$ to isolate clipping dynamics
- **Multi-seed validation**: 3–5 seeds across core configurations for error bars
- **Shadow model attack**: Train shadow models to learn non-linear decision boundaries
- **LiRA**: Likelihood Ratio Attack across per-sample out-of-bag models
