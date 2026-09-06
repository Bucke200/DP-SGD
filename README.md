# Empirical Evaluation of Differential Privacy in Machine Learning under Membership Inference Attacks

Undergraduate research project evaluating the empirical relationship between **Differential Privacy (DP-SGD)**, **model utility**, and **vulnerability to Membership Inference Attacks (MIA)** across MNIST and CIFAR-10 using PyTorch and Opacus.

> [!NOTE]
> **Research Manuscript**: The complete paper is available in [`PAPER.md`](PAPER.md): *Deconstructing DP-SGD: Empirical Membership Privacy Under Threshold Attacks Stems from Gradient Clipping, Not Noise Injection*.

MNIST serves as a control case (the model generalizes too well for MIA to find signal). CIFAR-10 is the primary evaluation dataset, where a subsampled training set and extended training produce a natural memorization gap that enables meaningful MIA results.

---

## Table of Contents

- [Research Paper](#research-paper)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Experimental Results](#experimental-results)
  - [MNIST (control: 60k train, 5 epochs — no memorization)](#mnist-control-60k-train-5-epochs--no-memorization)
  - [CIFAR-10 (primary: 5k train, 50 epochs — induced memorization)](#cifar-10-primary-5k-train-50-epochs--induced-memorization)
  - [Ablation: Clipping vs Noise](#ablation-clipping-vs-noise)
  - [Synthesis: Privacy–Utility–Attack Decomposition](#synthesis-privacyutilityattack-decomposition)
- [Limitations & Future Work](#limitations--future-work)
- [Final Thoughts & Research Takeaways](#final-thoughts--research-takeaways)
- [Checkpoint Format](#checkpoint-format)
- [Research Objectives & Roadmap](#research-objectives--roadmap)

---

## Research Paper

The full research paper is compiled at [`PAPER.md`](PAPER.md). It details:
- The theoretical motivation and background on DP-SGD and Membership Inference Attacks.
- Multi-seed CIFAR-10 $\epsilon$-sweep across 9 noise multipliers ($N=30$ runs, seeds $42, 43, 44$) with $10{,}000$-resample bootstrap confidence intervals.
- The 15-run gradient clipping sweep ($C \in \{0.5, 1.0, 5.0, 10.0, 50.0\}$ at $\sigma=0.0$) isolating clipping dynamics from Gaussian noise.
- The two-mechanism empirical decomposition and the paper's central thesis (an empirical finding scoped specifically to one architecture, one dataset, and one attack, with no formal proof claimed).
- Formal discussion of limitations (threshold attack as a lower-bound baseline) and roadmap for state-dependent shadow/LiRA attacks.

---

## Repository Structure

```text
dp-sgd/
├── README.md
├── PAPER.md                            # Full research paper manuscript
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
│   ├── sweep_clipping.py               # Multi-seed gradient clipping sweep (sigma=0.0) with binding diagnostics
│   ├── verify_checkpoints.py           # Checkpoint loading verification
│   ├── threshold_attack.py             # Loss-threshold MIA evaluation
│   ├── run_ablation.py                 # Clipping vs noise ablation study
│   ├── plot_curves.py                  # Privacy-utility curve plotting
│   ├── plot_mia_curves.py              # MIA AUC and ROC curve plotting
│   └── plot_clipping.py                # Dual-axis utility vs MIA attack AUC plot for clipping sweep
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
    └── test_pipeline.py                # 26 passing tests
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

Set `DATASET` in `config.py` to `"mnist"` or `"cifar10"`, then execute the pipeline in order:

```bash
# 1. Epsilon Sweep: Multi-seed training + MIA evaluation + bootstrap CI aggregation
python -m src.sweep_epsilon --seeds 42 43 44    # Multi-seed sweep across 3 seeds (re-uses existing checkpoints)
# (For a quick single-seed run with seed 42 default: python -m src.sweep_epsilon)

# 2. Checkpoint Verification: Validate weight integrity and accuracy match
python -m src.verify_checkpoints                # Validates checkpoints against the sweep manifest

# 3. Privacy-Utility & Attack Visualizations
python -m src.plot_curves                       # Privacy-utility curve (with multi-seed error bars)
python -m src.plot_mia_curves                   # MIA AUC (bootstrap CI band), ROC curves, & 3-way tradeoff

# 4. Gradient Clipping Ablation (Pure clipping at noise sigma = 0.0)
python -m src.sweep_clipping --seeds 42 43 44   # 15-run clipping sweep (C in {0.5, 1, 5, 10, 50}) + MIA evaluation
python -m src.plot_clipping                     # Dual-axis utility vs. attack AUC plot

# Optional: Standalone loss-threshold MIA against a single manifest
# python -m src.threshold_attack
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

### MNIST (control: 60k train, 5 epochs — no memorization)

**Configuration**: Dataset = MNIST | N_TRAIN = 60,000 | Epochs = 5 | Purpose = Negative control (verifies MIA baseline fails without memorization)

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

*Note: MNIST threshold attack AUC ~0.50 across all epsilon is the expected null result when the model does not memorize, not a failure of the attack implementation.*

MNIST maintains 89-91% accuracy down to epsilon ~ 0.19. Threshold MIA AUC ~ 0.49 across all epsilon values, indistinguishable from random guessing. The model generalizes too well on MNIST for membership inference to find any signal, which is why it serves as a control rather than the primary evaluation dataset.

The generated curve ([`experiments/mnist/results/privacy_utility_curve.png`](experiments/mnist/results/privacy_utility_curve.png))
![Privacy-Utility Curve](./experiments/mnist/results/privacy_utility_curve.png)
highlights:
- **Strong Privacy Regime ($\epsilon < 1.0$)**: Maintains ~89–91% accuracy down to $\epsilon \approx 0.19$, demonstrating strong utility retention for MNIST under DP-SGD.
- **Moderate Privacy Regime ($1.0 \le \epsilon \le 10.0$)**: Accuracy plateaus at ~91.3% (within ~7.8 percentage points of the non-private baseline).
- **Extreme Noise Regime ($\sigma \ge 3.0, \epsilon < 0.1$)**: Utility degrades noticeably (80.9% at $\sigma=3.0$, 68.5% at $\sigma=5.0$) due to heavy gradient perturbation.

### CIFAR-10 (primary: 5k train, 50 epochs — induced memorization)

Trained on a 5k subsampled training set for 50 epochs to induce controlled overfitting across 3 random seeds ($42, 43, 44$). The smaller training set and extended training create a strong natural memorization gap ($+49.11\% \pm 0.75\%$), giving the loss-threshold MIA a strong signal to exploit in the non-private baseline (Attack AUC = $0.8578$, Attack Accuracy = $85.02\%$).

All results below report multi-seed aggregations across 3 random seeds ($42, 43, 44$) with $10{,}000$-resample 95% bootstrap confidence intervals for Loss Attack AUC:

**Configuration**: Dataset = CIFAR-10 | N_TRAIN = 5,000 | Epochs = 50 | Purpose = Primary evaluation (induced memorization gap creates signal for MIA benchmarking)

| Model | Noise Mult ($\sigma$) | $\epsilon$ ($\delta=10^{-5}$) | Test Acc (mean ± std) | Gen Gap (mean ± std) | Loss Attack AUC (95% Bootstrap CI) | Attack Acc (mean ± std) | TPR @ 1% FPR (mean ± std) | Distinguishable from 0.50? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.00 | $\infty$ | **51.13% ± 0.30%** | **+0.4911 ± 0.0075** | **0.8578** [0.8530, 0.8623] | **85.02% ± 0.18%** | 0.0311 ± 0.0019 | **Yes** |
| **DP-SGD** | 0.30 | 193.50 | 38.76% ± 0.98% | +0.0249 ± 0.0091 | **0.5216** [0.5149, 0.5276] | 52.08% ± 0.52% | 0.0095 ± 0.0008 | Yes |
| **DP-SGD** | 0.50 | 32.53 | 38.54% ± 0.95% | +0.0253 ± 0.0090 | **0.5207** [0.5143, 0.5272] | 52.02% ± 0.44% | 0.0100 ± 0.0006 | Yes |
| **DP-SGD** | 0.70 | 11.29 | 38.53% ± 0.99% | +0.0236 ± 0.0081 | **0.5198** [0.5137, 0.5260] | 51.89% ± 0.37% | 0.0096 ± 0.0015 | Yes |
| **DP-SGD** | 0.90 | 6.05 | 38.40% ± 1.02% | +0.0192 ± 0.0091 | **0.5199** [0.5131, 0.5263] | 51.80% ± 0.43% | 0.0105 ± 0.0019 | Yes |
| **DP-SGD** | 1.10 | 4.09 | 38.31% ± 1.08% | +0.0234 ± 0.0095 | **0.5188** [0.5127, 0.5254] | 51.72% ± 0.38% | 0.0121 ± 0.0032 | Yes |
| **DP-SGD** | 1.50 | 2.51 | 36.93% ± 0.64% | +0.0283 ± 0.0042 | **0.5218** [0.5156, 0.5282] | 51.99% ± 0.26% | 0.0111 ± 0.0032 | Yes |
| **DP-SGD** | 2.00 | 1.71 | 31.40% ± 0.18% | +0.0343 ± 0.0024 | **0.5235** [0.5173, 0.5301] | 52.01% ± 0.15% | 0.0113 ± 0.0007 | Yes |
| **DP-SGD** | 3.00 | 1.05 | 20.24% ± 0.85% | +0.0087 ± 0.0078 | **0.5090** [0.5025, 0.5152] | 50.99% ± 0.31% | 0.0110 ± 0.0007 | Yes |
| **DP-SGD** | 5.00 | 0.59 | 13.75% ± 1.52% | +0.0038 ± 0.0074 | **0.4995** [0.4938, 0.5061] | 50.57% ± 0.24% | 0.0113 ± 0.0016 | **No** (spans 0.50) |

> [!NOTE]
> **Statistical vs. Practical Significance**: AUC values of ~0.52 clear the bootstrap CI test (statistically distinguishable from 0.50) due to statistical power from 10,000 evaluation pairs, but give an attacker almost nothing in practice—roughly 52% correct pairwise ranking versus 50% for a random coin flip. For assessing operational privacy leakage, TPR @ 1% FPR (~1%, equal to random guessing) is the far more decision-relevant number. The full methodological treatment belongs in [`PAPER.md`](PAPER.md).

#### Visualizations (CIFAR-10)

- **Privacy-Utility-Attack Tradeoff (Dual-Axis with Multi-Seed Error Bars):**
  ![Privacy-Utility-Attack Tradeoff](./experiments/cifar10/results/privacy_utility_attack.png)
- **Privacy-Utility Accuracy Curve (Error Bars on All Points):**
  ![Privacy-Utility Curve](./experiments/cifar10/results/privacy_utility_curve.png)
- **Attack AUC vs Privacy Budget ($\epsilon$) (with 95% Bootstrap Confidence Band):**
  ![MIA AUC vs Epsilon](./experiments/cifar10/results/mia_auc_vs_epsilon.png)
- **MIA ROC Curves (Linear & Log-Log):**
  ![MIA ROC Curves](./experiments/cifar10/results/mia_roc_curves.png)

#### Key Empirical Findings (Epsilon Sweep):
1. **Utility Plateau vs. Steep Privacy Cliff:**
   Model accuracy is remarkably resilient between $\epsilon \approx 193.5$ ($\sigma=0.3$) and $\epsilon \approx 4.09$ ($\sigma=1.1$, Opacus default), remaining virtually flat around $38.3\% - 38.8\%$ (a penalty of ~12.5 percentage points relative to unconstrained baseline $51.13\%$). However, utility falls off a cliff past $\sigma = 1.5$ ($\epsilon < 2.5$), reaching $20.24\%$ at $\epsilon = 1.05$ and collapsing to $13.75\%$ near random guess at $\epsilon = 0.59$.
2. **DP-SGD Crushes Memorization and MIA Vulnerability:**
   Even minimal DP noise ($\sigma=0.3, \epsilon=193.5$) suppresses the baseline generalization gap from $+49.11\%$ down to $+2.49\%$, collapsing Attack AUC from $0.8578$ to $0.5216$. For all practical DP regimes ($\sigma \ge 0.3$), MIA AUC remains bounded between $0.4995$ and $0.5235$.
3. **Statistical Random Guessing at Extreme Noise:**
   At $\sigma = 5.0$ ($\epsilon = 0.59$), the 95% bootstrap confidence interval for Attack AUC ($[0.4938, 0.5061]$) crosses $0.50$, rendering membership inference statistically indistinguishable from random chance.
4. **Optimal Tradeoff Regime:**
   $\sigma \in [0.9, 1.1]$ ($\epsilon \in [4.09, 6.05]$) provides the optimal empirical balance: provable single-digit differential privacy ($\epsilon \le 6.0$), maximum achievable DP accuracy (~$38.4\%$), and complete empirical suppression of threshold MIA (AUC $< 0.52$).

### Ablation: Clipping vs Noise

To isolate the individual contribution of **per-sample gradient clipping** from **Gaussian noise addition**, an extensive ablation sweep was conducted across five clipping thresholds ($C \in \{0.5, 1.0, 5.0, 10.0, 50.0\}$) at noise multiplier $\sigma = 0.0$ across three random seeds ($42, 43, 44$), logging per-sample gradient norm diagnostics at epoch 1 and epoch 50:

**Configuration**: Dataset = CIFAR-10 | N_TRAIN = 5,000 | Epochs = 50 | Purpose = Gradient clipping ablation ($\sigma=0.0$, varying $C \in \{0.5, 1.0, 5.0, 10.0, 50.0\}$ across seeds 42–44)

| Condition | Clip Norm ($C$) | Test Acc (mean ± std) | Gen Gap (mean ± std) | Loss Attack AUC (95% Bootstrap CI) | Distinguishable from 0.50? | Epoch 1 Clipped | Epoch 50 Clipped |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Clip** | **0.5** | $35.40\% \pm 0.73\%$ | $+0.0191 \pm 0.0100$ | **$0.5138$** $[0.5071, 0.5201]$ | Marginal | $100.0\%$ | $99.7\%$ |
| **Pure Clip** | **1.0** | $38.85\% \pm 0.85\%$ | $+0.0337 \pm 0.0074$ | **$0.5230$** $[0.5161, 0.5291]$ | Yes ($>0.50$) | $100.0\%$ | $98.1\%$ |
| **Pure Clip** | **5.0** | $45.34\% \pm 1.28\%$ | $+0.1891 \pm 0.0295$ | **$0.6198$** $[0.6131, 0.6258]$ | **Yes** | $3.1\%$ | $72.6\%$ |
| **Pure Clip** | **10.0** | $46.72\% \pm 0.82\%$ | $+0.4611 \pm 0.0077$ | **$0.7780$** $[0.7730, 0.7833]$ | **Yes** | $0.3\%$ | $24.3\%$ |
| **Pure Clip** | **50.0** | $49.94\% \pm 0.61\%$ | $+0.5014 \pm 0.0102$ | **$0.8421$** $[0.8369, 0.8468]$ | **Yes** | $0.0\%$ | $0.0\%$ |
| *Baseline (no DP)* | *None* | *$51.13\% \pm 0.30\%$* | *$+0.4911 \pm 0.0075$* | **$0.8578$** $[0.8530, 0.8623]$ | *Yes* | *0.0%* | *0.0%* |
| *Full DP-SGD* | *1.0 ($\sigma=1.1$)* | *$38.31\% \pm 1.08\%$* | *$+0.0234 \pm 0.0095$* | **$0.5188$** $[0.5127, 0.5254]$ | *Marginal* | *—* | *—* |

> [!NOTE]
> **Statistical vs. Practical Significance**: While pure clipping with $C \le 1.0$ yields AUC values (~0.51–0.52) that are statistically distinguishable from 0.50 under the bootstrap CI test, this confers virtually no practical advantage to an adversary (~52% pairwise ranking vs. 50% for a coin flip). Readers should look to TPR @ 1% FPR as the decision-relevant indicator of meaningful operational attack success. See [`PAPER.md`](PAPER.md) for full discussion.

#### Visualizations (Clipping Sweep)

- **Gradient Clipping Sweep (Utility vs. MIA Attack AUC):**
  ![Clipping Sweep](./experiments/cifar10/results/clipping_sweep.png)

#### Key Empirical Findings:
1. **The accuracy penalty is an optimization constraint that relaxes with $C$:**
   At $C = 1.0$, test accuracy is suppressed by ~12–14 percentage points compared to the baseline. As $C$ increases to $50.0$, test accuracy reaches $49.94\%$ (peaking at $50.52\%$ on seed 44), effectively recovering the unclipped baseline accuracy ($51.13\% \pm 0.30\%$).
2. **No sweet spot exists where clipping alone suppresses attacks without hurting accuracy:**
   Utility and memorization are tightly coupled under gradient clipping. As soon as $C$ is relaxed to $5.0$ to regain accuracy ($45.34\%$), the generalization gap surges to $+18.91\%$ and the MIA Attack AUC climbs to $0.6198$ (statistically distinguishable from $0.50$). At $C=10.0$, clipping is largely unbinding, and Attack AUC escalates to $0.7780$, reaching $0.8421$ at $C=50.0$.
3. **Clipping suppresses memorization via underfitting, not formal mathematical privacy:**
   While small clip norms ($C \le 1.0$) destroy simple threshold MIA signal by preventing sample memorization (the network never adequately fits individual training samples), they provide $\epsilon = \infty$ (zero formal differential privacy). Calibrated Gaussian noise addition is essential for provable, worst-case differential privacy guarantees.

### Synthesis: Privacy–Utility–Attack Decomposition

The multi-seed $\epsilon$ sweep ($N=30$ runs, seeds $42, 43, 44$) perfectly corroborates the gradient clipping ablation, providing empirical closure on how Differential Privacy interacts with model utility and membership vulnerability in deep learning.

#### 1. Empirical Alignment Across Sweeps

A direct comparative analysis of the empirical attack surface reveals an unmistakable dichotomy:
- **Attack Invariance Under Noise Scaling ($\sigma \in [0.3, 5.0]$):**
  Across all nine DP-SGD configurations, Loss Attack AUC remains strictly compressed within a narrow $0.4995 - 0.5235$ band, contrasted against the unconstrained baseline of $0.8578$. This constitutes an immediate **$0.34$-point AUC gap** between the baseline and any DP model, but **essentially zero variance across privacy budgets**. Scaling Gaussian noise by $>16\times$ shifts the formal mathematical bound by over **$300\times$** ($\epsilon = 193.50 \to 0.59$), yet changes empirical Attack AUC by less than $0.024$ (within the 95% bootstrap confidence margin).
- **Attack Sensitivity Under Gradient Clipping ($C \in [1.0, 50.0]$ at $\sigma=0.0$):**
  Conversely, varying the clipping threshold $C$ in the absence of any noise injection drives a massive **$0.32$-point swing in Attack AUC** ($0.5230 \to 0.8421$). At $C=1.0$, pure clipping alone restricts Attack AUC to $0.5230$, matching full DP-SGD ($0.5188$ at $\sigma=1.1$). As $C$ is relaxed to $50.0$, the baseline attack surface ($0.8578$) is virtually recovered ($0.8421$).

| Mechanism | Parameter | Primary Operational Role | Governs | Empirical MIA Impact | Formal Privacy Impact |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **Gradient Clipping** | $C$ | Optimization capacity throttling | Utility vs. Empirical Vulnerability | **Decisive**: $0.32$ AUC swing ($0.52 \to 0.84$) | None ($\epsilon = \infty$) |
| **Gaussian Noise** | $\sigma$ | Stochastic gradient perturbation | Theoretical Worst-Case Bounds | **Negligible**: $\le 0.02$ AUC variation across $300\times \epsilon$ | **Decisive**: $\epsilon \in [0.59, 193.50]$ |

#### 2. Mechanistic Decomposition: Two Independent Axes

The privacy–utility–attack relationship empirically decomposes into two independent mechanisms (an empirical finding scoped specifically to our evaluated setup: CifarCNN on CIFAR-10 under threshold MIA, with no formal proof claimed):
1. **Clipping ($C$) dictates the empirical utility–vulnerability Pareto frontier:**
   Per-sample gradient clipping throttles the network's optimization capacity. By bounding $\|\mathbf{g}_i\|_2 \le C$, the optimizer cannot fit atypical, high-gradient samples, which directly suppresses sample-level memorization and crushes the generalization gap ($\Delta_{\text{gen}} \le +3.43\%$ for all DP runs vs $+49.11\%$ in baseline). However, this optimization bottleneck causes severe underfitting rather than benign regularization, incurring an unavoidable utility penalty (~$12.5$ percentage points) because the model never adequately fits the training set.
2. **Noise ($\sigma$) dictates the formal $(\epsilon, \delta)$-DP guarantee:**
   Gaussian noise injection establishes the mathematical privacy bound via differential privacy accounting. However, when $C=1.0$, the model is already so optimizationally constrained that there is virtually no residual memorization signal remaining for noise perturbation to mask. Hence, noise provides worst-case provable bounds without visibly moving the empirical threshold attack surface.

#### 3. Tripartite Paper Evidence (Core Figures)

Three core figures tell the complete empirical story for publication:
1. **The Privacy–Utility Curve ([`experiments/cifar10/results/privacy_utility_curve.png`](experiments/cifar10/results/privacy_utility_curve.png)):**
   Plots test accuracy across $\epsilon$, showing an initial plateau from $\epsilon \approx 193.5$ to $\epsilon \approx 4.09$ (~$38.3\% - 38.8\%$), followed by a sharp utility cliff past $\epsilon \approx 2.51$ ($\sigma = 1.50$), collapsing to near-random accuracy ($13.75\%$) at $\epsilon = 0.59$.
2. **The Empirical Attack Invariance Curve ([`experiments/cifar10/results/mia_auc_vs_epsilon.png`](experiments/cifar10/results/mia_auc_vs_epsilon.png)):**
   Plots Loss Attack AUC across the formal privacy budget with $95\%$ bootstrap confidence bands. The attack curve is flat across all DP budgets; the entire mitigation occurs in the discrete transition from the unclipped baseline ($0.8578$) to any clipped model ($0.5188 - 0.5235$).
3. **The Coupled Clipping Dynamics Curve ([`experiments/cifar10/results/clipping_sweep.png`](experiments/cifar10/results/clipping_sweep.png)):**
   Visualizes the pure clipping ablation at $\sigma=0.0$, demonstrating that accuracy and Attack AUC are monotonically coupled across $C \in [0.5, 50.0]$. No operating point exists where clipping alone provides privacy without an accuracy penalty.

#### 4. Central Paper Thesis

> **Central Claim**: DP-SGD's empirical defense against threshold-based Membership Inference Attacks (MIA) on CIFAR-10 is primarily attributable to **per-sample gradient clipping** rather than **Gaussian noise injection**. Clipping acts as an optimization constraint that prevents sample memorization—and therefore destroys the generalization gap that MIA exploits—at a proportional cost to model utility. Gaussian noise injection provides the formal, worst-case $(\epsilon, \delta)$-DP guarantee required for mathematical certification, but contributes no measurable empirical defense beyond what clipping already establishes, because clipping at $C=1.0$ eliminates the memorization signal before additive noise has anything to mask.

---

## Limitations & Future Work

While our empirical findings across 45 verified checkpoints are conclusive for the standard threshold attack paradigm, we explicitly identify the boundary conditions of this study and frame the path forward:

### 1. The Threshold MIA as a Conservative Baseline
- **Nature of the Attack**: The loss/confidence-threshold membership inference attack (Yeom et al., 2018) is the foundational benchmark in empirical privacy research. It tests whether an adversary can separate members from non-members based on first-order prediction loss distributions.
- **Why It Matters Here**: The fact that per-sample gradient clipping at $C=1.0$ drops threshold Attack AUC to $0.5188 - 0.5235$ (near-random guessing) demonstrates that clipping suppresses the macroscopic generalization gap that threshold attacks rely on. 
- **Limitation**: The threshold attack is nonetheless the weakest MIA in the literature. It computes global decision thresholds without modeling point-specific difficulty or variance.

### 2. Stronger State-Dependent Attacks (LiRA & Shadow Models)
- **The Core Scientific Hypothesis**: Advanced membership inference attacks—most notably the **Likelihood Ratio Attack (LiRA; Carlini et al., 2022)** and **shadow model ensembles (Shokri et al., 2017)**—train parametric Gaussian models on query outputs across multiple out-of-bag shadow networks. These attacks probe higher-order, per-sample statistical representations rather than aggregate loss margins.
- **The Open Question**: Does pure gradient clipping ($\sigma=0.0$) leave subtle, sample-specific parameter representations that a parametric attack like LiRA could extract, which calibrated Gaussian noise ($\sigma > 0$) successfully destroys?
- **Future Direction**: We frame this not as a caveat that weakens our findings, but as an exciting open frontier. Our clean baseline establishes that threshold attacks are completely blind to Gaussian noise injection once clipping is enforced. Benchmarking LiRA across our multi-seed clipping vs. noise matrix will isolate the exact point at which additive noise transitions from theoretically certified to empirically indispensable.

### 3. Architectural and Benchmark Scope
- **Current Setup**: Evaluated on subsampled CIFAR-10 ($N=5{,}000$, 50 epochs, induced memorization) and full MNIST ($N=60{,}000$, negative control) using 4-layer and 2-layer BatchNorm-free CNNs.
- **Future Extensions**: Scaling the multi-seed decomposition to modern architectures (Vision Transformers, pre-trained transfer learning with DP-SGD fine-tuning) to assess whether representation learning in larger parameter spaces alters the clipping-to-noise ratio.

---

## Final Thoughts & Research Takeaways

1. **Decoupling Optimization from Formal Privacy Accounting**:
   In the existing literature, DP-SGD is predominantly treated as a monolithic black box, with Gaussian noise injection credited for both formal mathematical bounds and observed empirical privacy gains. Our multi-seed empirical finding demonstrates that this conflates two independent mechanisms (scoped specifically to one architecture, one dataset, and one attack):
   - **Clipping ($C$)** is an optimization bottleneck. The mechanism is underfitting: clipping caps the effective per-sample gradient contribution, the model never fits the training set (train loss ~1.9), and test accuracy falls 12.5 percentage points. A regularizer would close the train-test gap while holding test accuracy; clipping merely prevents learning. It eliminates the memorization signal that drives first-order MIA vulnerability directly through optimization failure.
   - **Noise ($\sigma$)** is a theoretical privacy mechanism. It establishes the worst-case $(\epsilon, \delta)$ guarantee required for cryptographic and legal certification, but adds virtually zero empirical defense against standard threshold attacks once $C=1.0$ is in place.

2. **Practical Guidance for Private ML Practitioners**:
   - **The $C=1.0$ Default**: Setting $C=1.0$ provides strong empirical resistance to first-order threshold attacks out of the box, but enforces an unavoidable ~12.5 percentage point drop on CIFAR-10 utility.
   - **The Operational Sweet Spot**: Operating at $\sigma \in [0.9, 1.1]$ ($\epsilon \in [4.09, 6.05]$) provides the best real-world balance: certified single-digit $(\epsilon, \delta)$-DP at effectively zero marginal accuracy penalty compared to the clipped baseline.
   - **The Diminishing Returns of Extreme Noise**: Pushing noise to $\sigma \ge 2.0$ ($\epsilon \le 1.7$) triggers a severe utility cliff (collapsing to $13.75\%$ at $\sigma=5.0$) while delivering virtually no empirical gain over $\sigma=1.1$.

3. **Methodological Standard for Empirical Privacy Research**:
   To avoid misattributing underfitting side-effects of gradient clipping to differential privacy noise, empirical evaluations of DP-SGD should routinely include **pure clipping baselines ($\sigma=0.0$)** as standard controls alongside full DP-SGD runs.

---

## Checkpoint Format

Every checkpoint stores a self-contained metadata dict for downstream MIA evaluation:

```python
{
    "model_state_dict": ...,
    "model_type": "baseline" | "dp-sgd",
    "epsilon": float | "inf" | None,
    "epsilon_note": str | None,
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
    "clipping_diagnostics": dict | None,
}
```

Weights load directly into the appropriate model class (`SampleCNN` or `CifarCNN`) without Opacus dependencies. Use `src/verify_checkpoints.py` to validate checkpoint integrity and accuracy match.

---

## Research Objectives & Roadmap

- [x] **Objective 1**: DP-SGD baseline implementation and Opacus verification (MNIST + CIFAR-10)
- [x] **Objective 2**: Multi-epsilon privacy sweep with model checkpointing (MNIST + multi-seed CIFAR-10 across seeds 42–44 with bootstrap CIs)
- [x] **Objective 3**: Threshold Membership Inference Attack (MIA) framework and empirical evaluation (MNIST + CIFAR-10)
- [x] **Objective 4**: Privacy–Utility–Security tradeoff analysis and multi-seed gradient clipping ablation ($C \in \{0.5, 1, 5, 10, 50\}$, seeds 42–44)
- [ ] **Objective 5 (Next Steps)**: Shadow model MIA and Likelihood Ratio Attack (LiRA) framework to probe subtle clipping vs noise representations

### Planned Extensions
- [x] **Clipping-norm sweep**: $C \in \{0.5, 1, 5, 10, 50\}$ at $\sigma=0$ to isolate clipping dynamics
- [x] **Multi-seed validation**: 3 seeds across both epsilon sweep (10 conditions, 30 checkpoints) and clipping configurations for statistical confidence intervals
- [ ] **Shadow model attack**: Train shadow models to learn non-linear decision boundaries
- [ ] **LiRA**: Likelihood Ratio Attack across per-sample out-of-bag models
