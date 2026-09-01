# Progress Update: CIFAR-10 Integration, Full Pipeline Execution, and Clipping Ablation

**Date:** August 31, 2026  
**Project:** Empirical Evaluation of Differential Privacy in Machine Learning under Membership Inference Attacks  
**Framework:** PyTorch, Opacus, Torchvision  

---

## 1. Overview & Objectives Accomplished

This session upgraded the existing MNIST-only DP-SGD and Membership Inference Attack (MIA) evaluation project into a dual-dataset research framework supporting both **MNIST** and **CIFAR-10**. All training, multi-$\epsilon$ privacy sweeps, standalone verification, threshold MIA evaluations, and plotting routines were generalized, executed end-to-end, and empirically validated.

---

## 2. Detailed Summary of Actions & Changes

### Step 1: Central Configuration (`config.py`)
- Added `DATASET = "cifar10"` toggle (defaulting to `"cifar10"`, supports `"mnist"`).
- Added standard CIFAR-10 normalization constants:
  - `CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]`
  - `CIFAR10_STD = [0.2470, 0.2435, 0.2616]`
- Preserved MNIST normalization constants and shared training hyperparameters (`N_TRAIN = 5000`, `EPOCHS = 50`, `BATCH_SIZE = 64`, `LR = 0.05`, `MAX_GRAD_NORM = 1.0`, `NOISE_MULTIPLIER = 1.1`, `DELTA = 1e-5`).
- Configured dynamic dataset-scoped output directory paths:
  ```python
  OUTPUT_DIR = f"experiments/{DATASET}"
  CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
  SPLITS_DIR = f"{OUTPUT_DIR}/splits"
  RESULTS_DIR = f"{OUTPUT_DIR}/results"
  ```
- Documented migration instructions for preexisting MNIST experimental artifacts.

### Step 2: Model Architecture (`src/model.py`)
- Preserved `SampleCNN` for MNIST classification.
- Added `CifarCNN` for CIFAR-10 ($3 \times 32 \times 32$ RGB images):
  - **Architecture:** `Conv2d(3, 32, 3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(32, 32, 3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` $\rightarrow$ `Conv2d(32, 64, 3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(64, 64, 3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(4096, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(256, 10)`.
  - **Opacus Compatibility:** Completely BatchNorm-free and Dropout-free, enabling per-sample gradient computation and controlled training memorization.
- Added `get_model(dataset=None)` factory function reading from `config.DATASET`.
- Verified `CifarCNN` with `ModuleValidator.validate()`, confirming `is_valid == True` and 0 layer errors.

### Step 3: Dataset Splitting & Dispatcher (`src/data_split.py`)
- Preserved `get_split_loaders()` for MNIST.
- Added `get_cifar10_split_loaders()`:
  - Automates CIFAR-10 download via `torchvision.datasets.CIFAR10`.
  - Normalizes with `config.CIFAR10_MEAN` and `config.CIFAR10_STD`.
  - Subsamples 5,000 training examples with seeded RNG (`seed=42`) to create a measurable generalization gap.
  - Persists member indices to `config.SPLITS_DIR/member_indices.npy` and `member_indices_n5000_seed42.json`.
- Implemented `get_data_loaders(dataset=None, ...)` dispatcher function.

### Step 4: Pipeline Scripts Update
- **`src/train_baseline.py`**: Replaced specific loaders and model construction with `get_data_loaders()` and `get_model()`.
- **`src/train_dp.py`**: Replaced with generic loaders/models, and added Windows stdout encoding safety for console metrics.
- **`src/sweep_epsilon.py`**: Generalized training routines across all noise multipliers, checkpointing unwrapped model weights and metadata into dataset-scoped directories.
- **`src/verify_checkpoints.py`**: Updated to resolve checkpoints and manifests dynamically from `config.RESULTS_DIR` and `config.CHECKPOINT_DIR`.
- **`src/threshold_attack.py`**: Updated to load models via `get_model()`, evaluate against dataset-specific member/non-member splits, and write score matrices to `config.RESULTS_DIR/mia_scores/`.
- **`src/plot_curves.py` & `src/plot_mia_curves.py`**: Updated to read dataset-scoped results and generate plots with dynamic figure titles.

### Step 5: Test Suite & Dependencies
- **`tests/test_pipeline.py`**: Added unit and integration tests for:
  - Dataset-scoped config paths consistency
  - `CifarCNN` forward pass shape `(8, 10)`
  - Opacus `ModuleValidator` check on `CifarCNN`
  - Per-sample gradient computation and DP-SGD step on 3-channel CIFAR inputs
  - Model and loader dispatcher routing
  - All 17 tests passing cleanly (`pytest -v`).
- **`requirements.txt`**: Added `scikit-learn` and ensured `matplotlib` is present.

### Step 6: MNIST Data Migration
- Successfully moved existing MNIST experiment artifacts:
  - `experiments/checkpoints` $\rightarrow$ `experiments/mnist/checkpoints`
  - `experiments/splits` $\rightarrow$ `experiments/mnist/splits`
  - `experiments/results` $\rightarrow$ `experiments/mnist/results`
- Verified all 10 MNIST checkpoints: **10/10 passed** verification with exact test accuracy reproduction.

---

## 3. CIFAR-10 Full Experimental Results

### Multi-$\epsilon$ Privacy Sweep & MIA Attack Metrics

| Model | Noise Mult ($\sigma$) | $\epsilon$ ($\delta=10^{-5}$) | Test Accuracy | Generalization Gap | MIA Loss AUC | Attack Accuracy | TPR @ 1% FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Non-DP)** | $0.00$ | $\infty$ | **50.96%** | **+49.30%** | **0.8535** | **85.07%** | **0.0287** |
| **DP-SGD** | $0.30$ | 193.50 | 37.42% | +1.48% | 0.5151 | 51.63% | 0.0102 |
| **DP-SGD** | $0.50$ | 32.53 | 37.26% | +1.50% | 0.5146 | 51.60% | 0.0092 |
| **DP-SGD** | $0.70$ | 11.29 | 37.28% | +1.38% | 0.5129 | 51.55% | 0.0090 |
| **DP-SGD** | $0.90$ | 6.05 | 37.20% | +0.90% | 0.5124 | 51.26% | 0.0092 |
| **DP-SGD** | $1.10$ | 4.09 | 37.14% | +1.64% | 0.5109 | 51.23% | 0.0102 |
| **DP-SGD** | $1.50$ | 2.51 | 36.05% | +2.44% | 0.5151 | 51.64% | 0.0076 |
| **DP-SGD** | $2.00$ | 1.71 | 31.15% | +3.40% | 0.5167 | 51.80% | 0.0120 |
| **DP-SGD** | $3.00$ | 1.05 | 19.97% | +0.16% | 0.5023 | 50.64% | 0.0112 |
| **DP-SGD** | $5.00$ | 0.59 | 15.47% | +1.42% | 0.5048 | 50.91% | 0.0107 |

### Generated Visualizations (CIFAR-10)
- **Privacy–Utility Curve:** `experiments/cifar10/results/privacy_utility_curve.png`
- **MIA ROC Curves (Linear & Log-Log):** `experiments/cifar10/results/mia_roc_curves.png`
- **Attack AUC vs $\epsilon$:** `experiments/cifar10/results/mia_auc_vs_epsilon.png`
- **Privacy–Utility–Attack Tradeoff:** `experiments/cifar10/results/privacy_utility_attack.png`

---

## 4. Ablation Study: Per-Sample Clipping Only (`C=1.0, \sigma=0.0`)

To isolate the individual contribution of **gradient clipping** from **Gaussian noise**, an ablation was executed with `max_grad_norm = 1.0` and `noise_multiplier = 0.0`:

| Condition | Clipping ($C$) | Noise ($\sigma$) | Formal Privacy ($\epsilon$) | Test Accuracy | Train Loss | Generalization Gap | MIA Attack AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Baseline** | None | $0.0$ | $\infty$ | **50.96%** | **0.0002** | **+49.30%** | **0.8535** |
| **Ablation (Clip Only)** | **1.0** | **0.0** | **$\infty$** | **36.10%** | **1.8997** | **+2.20%** | **0.5175** |
| **Full DP-SGD** | **1.0** | **1.1** | **$\mathbf{4.09}$** | **37.14%** | **1.9494** | **+1.64%** | **0.5109** |

### Key Takeaways:
1. **Clipping suppresses memorization:** Restricting per-sample gradient norm alone prevents individual training samples from dominating parameter updates, reducing the generalization gap from `+49.30%` to `+2.20%` and lowering threshold MIA AUC from `0.8535` to `0.5175`.
2. **Noise provides formal mathematical differential privacy:** While clipping provides empirical regularization, it offers zero mathematical guarantee against worst-case adversaries ($\epsilon = \infty$). Injecting calibrated Gaussian noise ($\sigma = 1.1$) proves the formal $(\epsilon, \delta)$-DP bound ($\epsilon = 4.09$) with minimal utility penalty relative to clipping alone.

---

## 5. Artifact & Codebase Status

- **Codebase Health:** Clean, fully modular, and dataset-agnostic.
- **Automated Tests:** 17/17 passed (`pytest tests/test_pipeline.py`).
- **Data Scoping:** Fully separated between `experiments/mnist/` and `experiments/cifar10/`.
