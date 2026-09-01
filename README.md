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

Trained on a 5k subsampled training set for 50 epochs to induce controlled overfitting. The smaller training set and longer training create a train-test accuracy gap that gives the MIA a realistic signal to exploit.

The baseline is expected to show a significant memorization gap. The full DP-SGD sweep and threshold MIA evaluation are pending completion. Results will quantify how increasing noise (decreasing epsilon) affects both the privacy-utility tradeoff and the MIA attack surface on a dataset where membership inference is actually feasible.

### Ablation: Clipping vs Noise

At C=1.0 with sigma=0 (clip-only, no noise), the CIFAR-10 model achieves ~36% test accuracy versus ~37% at sigma=1.1. The ~14-point utility drop from baseline is almost entirely attributable to gradient clipping, not noise injection. Clipping acts as a severe optimization constraint (effective learning-rate cut) rather than a regularizer. A planned clipping-norm sweep (C in {0.5, 1, 5, 10, 50} at sigma=0) will further characterize this effect.

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
- [ ] **Objective 3** (partial): Threshold MIA implemented and evaluated on MNIST; CIFAR-10 evaluation in progress
- [ ] **Objective 3** (remaining): Shadow model MIA framework
- [ ] **Objective 4**: Three-way privacy-utility-MIA tradeoff analysis

### Planned Work

- **Clipping-norm sweep**: C in {0.5, 1, 5, 10, 50} at sigma=0 to isolate the clipping contribution to utility loss
- **Multi-seed runs**: 3-5 seeds on key configurations for statistical rigor
- **Shadow model MIA**: Train shadow models to build attack classifiers beyond the threshold baseline
- **Stretch**: LiRA (Likelihood Ratio Attack) to test whether a stronger attack can distinguish clip-only from clip+noise regimes
