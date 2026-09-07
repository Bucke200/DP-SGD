# Empirical Deconstruction of DP-SGD: Clipping-Induced Underfitting as the Primary Driver of Threshold MIA Defense

**Authors**: Srinjay Panja et al.
**Code & Data Repository**: `https://github.com/Bucke200/DP-SGD` (PyTorch + Opacus)  
**Primary Artifacts**: 45 Evaluated Checkpoints across MNIST & Subsampled CIFAR-10 (Seeds 42, 43, 44)

---

## Abstract

Differentially Private Stochastic Gradient Descent (DP-SGD) provides provable, worst-case $(\epsilon, \delta)$-privacy guarantees by clipping per-sample gradient norms to a threshold $C$ and injecting calibrated Gaussian noise scaled by $\sigma$. Standard evaluations treat DP-SGD as a unified algorithm and evaluate empirical privacy against Membership Inference Attacks (MIA) solely as a function of the privacy budget $\epsilon$. In this work, we systematically decompose the twin mechanisms of DP-SGD—per-sample gradient clipping versus Gaussian noise injection—to examine their individual contributions to model utility and empirical membership privacy under global-threshold attacks across three membership signals (loss, confidence, modified entropy).

Across a rigorous multi-seed sweep ($N=30$ runs, seeds $42, 43, 44$) on CIFAR-10, we observe a striking phenomenon: scaling the noise multiplier across practical operating regimes ($\sigma \in [0.30, 2.00]$) tightens the formal privacy budget by over $110\times$ (from $\epsilon = 193.50$ down to $\epsilon = 1.71$), yet empirical Attack AUC across all three signals remains completely flat within a narrow $0.5186 - 0.5239$ band (compared to $0.8578 - 0.8579$ for the non-private baseline); further reduction to exact statistical randomness at $\sigma=5.00$ ($\epsilon=0.59$, AUC $0.4980 - 0.5019$) is confounded by catastrophic model collapse to $13.75\%$ accuracy. Conversely, an extensive gradient clipping sweep at zero noise ($\sigma = 0.0$) across five clipping bounds ($C \in \{0.5, 1.0, 5.0, 10.0, 50.0\}$, $N=15$ runs) reveals a monotonic dose-response curve where varying $C$ produces a massive $0.32$-point swing in Attack AUC ($0.5138 \to 0.8421$), with $C=1.0$ uniformly suppressing all three signals (AUC $0.5230 - 0.5231$) to match full DP-SGD ($0.5186 - 0.5188$ at $C=1.0, \sigma=1.10$). Diagnostic per-sample gradient norms reveal that clipping binds from epoch 1 only at $C \le 1.0$ ($100.0\%$ clipped), whereas at $C \ge 5.0$ clipping is essentially inactive from the start ($3.1\%$ clipped at $C=5.0$, $0.3\%$ at $C=10.0$), allowing the network to freely fit training data; epoch-50 unbinding ($24.3\%$ clipped at $C=10.0$, $0.0\%$ at $C=50.0$) is a downstream signature of model convergence rather than its cause. While global Attack AUC rises strictly monotonically across all five clipping bounds, TPR @ 1% FPR remains flat at the random-guessing baseline for $C \le 1.0$ ($0.0095$ at $C=0.5$, $0.0093$ at $C=1.0$) before rising monotonically for $C \ge 1.0$; moreover, the apparent severity is metric-dependent: while global AUC surges by nearly $0.10$ points at $C=5.0$ ($0.5230 \to 0.6198$), TPR @ 1% FPR rises by only $0.0039$ ($0.0093 \to 0.0132$) while recovering $+6.49$ percentage points of test accuracy.

Furthermore, analyzing Attack AUC against the generalization gap across all 15 evaluated configurations reveals that the two mechanisms co-locate in the narrow low-gap overlap regime ($\Delta_{\text{gen}} \in [1.91\%, 3.43\%]$), where pure clipping ($C=0.50, 1.00$) and noise-varied models ($\sigma=0.90, 2.00$) exhibit matched Attack AUC ($0.514 - 0.523$) within overlapping bootstrap confidence intervals, but separate at high gap where residual clipping ($C=50.0$) costs the attacker ~0.016 AUC relative to the unclipped baseline. This reframes the fundamental privacy mechanism: neither gradient clipping per se nor Gaussian noise directly defends against global-threshold membership inference attacks; rather, empirical defense stems strictly from **clipping-induced underfitting**. Per-sample gradient clipping throttles optimization capacity, preventing the network from fitting atypical outlier samples and crushing the generalization gap ($+49.11\% \to \le +3.43\%$) at an unavoidable 12.5 percentage point utility penalty. Calibrated Gaussian noise provides formal worst-case $(\epsilon, \delta)$ guarantees, but contributes negligible empirical defense once clipping-induced underfitting is enforced. We evaluate multiple comparisons over 90 non-duplicate tests (loss and mentr) with Benjamini-Hochberg FDR control ($q=0.05$), confirm near-perfect loss–mentr rank collinearity (mean Spearman $\rho = 0.9787$), and discuss implications for state-dependent attacks (LiRA).

---

## 1. Introduction

Deep learning models are prone to memorizing individual training examples, exposing them to Membership Inference Attacks (MIA) where an adversary determines whether a given data point was part of the training set (Shokri et al., 2017; Yeom et al., 2018; Carlini et al., 2022). Differentially Private Stochastic Gradient Descent (DP-SGD; Abadi et al., 2016) has emerged as the principal algorithmic framework to prevent such leaks, providing rigorous, mathematically certified $(\epsilon, \delta)$-differential privacy guarantees.

To establish differential privacy, DP-SGD modifies standard mini-batch stochastic gradient descent through two distinct operations:
1. **Per-Sample Gradient Clipping**: Each individual sample's gradient $\mathbf{g}_i = \nabla_\theta \ell(\theta; x_i, y_i)$ is projected into an $L_2$ ball of radius $C$:
   $$\text{clip}_C(\mathbf{g}_i) = \mathbf{g}_i \cdot \min\left(1, \frac{C}{\|\mathbf{g}_i\|_2}\right)$$
2. **Gaussian Noise Perturbation**: The sum of clipped gradients is perturbed by isotropic Gaussian noise scaled by the noise multiplier $\sigma$:
   $$\tilde{\mathbf{g}} = \frac{1}{B} \left( \sum_{i=1}^B \text{clip}_C(\mathbf{g}_i) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right)$$

In theoretical literature, gradient clipping is treated as an operational prerequisite: its sole formal role is to bound the $L_2$ sensitivity of the gradient sum to $C$, enabling the calibrated Gaussian mechanism to inject the minimum variance noise required for $(\epsilon, \delta)$ accounting. Conventional wisdom consequently credits **additive Gaussian noise** as the primary protective agent against privacy attacks, viewing clipping merely as an unfortunate source of optimization bias.

In this paper, we challenge this premise. By designing controlled experiments that isolate the impact of gradient clipping from noise addition across multiple random seeds with bootstrap confidence intervals, we discover that:

> **Central Finding**: Under global-threshold membership inference attacks across three membership signals (loss, confidence, modified entropy), DP-SGD's empirical privacy protection is overwhelmingly driven by **clipping-induced underfitting**, not noise addition. Once gradients are clipped tightly enough to prevent fitting ($C \le 1.0$), sample memorization is suppressed to such an extent that global-threshold MIA AUC across all three signals drops to near-random guessing (clipping sweep: $0.5135 - 0.5231$ across $C \le 1.0$; noise sweep: $0.5186 - 0.5239$ across practical $\sigma \in [0.30, 2.00]$, with $\sigma=5.00$ reaching $0.4980 - 0.5019$ only via catastrophic model collapse), leaving effectively no empirical memorization signal for additive noise to obscure—even across an over $110\times$ variation in $\epsilon$ ($193.50 \to 1.71$). When $C$ is relaxed to $50.0$, clipping is inactive from the start and the model remains highly vulnerable ($0.8421$ AUC) despite clipping being active. Furthermore, comparing configurations across generalization gaps reveals that both mechanisms co-locate in the narrow low-gap overlap region ($\Delta_{\text{gen}} \in [1.91\%, 3.43\%]$) with overlapping bootstrap confidence intervals, but separate at high gap where residual clipping costs the attacker ~0.016 AUC beyond what gap alone predicts, demonstrating that empirical defense is governed by underfitting rather than additive noise.

### Summary of Contributions
1. **Multi-Seed Epsilon Sweep ($N=30$ runs)**: We train and verify 30 CIFAR-10 models across nine noise multipliers ($\sigma \in [0.3, 5.0]$) and non-private baselines under three random seeds ($42, 43, 44$), evaluating utility, generalization gap, and global-threshold MIA across three signals with 1,000-resample bootstrap confidence intervals and Benjamini-Hochberg FDR correction ($q=0.05$) across the 90 non-duplicate tests.
2. **Clip Norm as a Dose-Response Curve ($N=15$ runs)**: We sweep five clipping thresholds ($C \in \{0.5, 1.0, 5.0, 10.0, 50.0\}$) at fixed zero noise ($\sigma = 0.0$) across three seeds, logging per-sample gradient norm distributions and binding fractions at epoch 1 and 50. We establish that while utility and global Attack AUC rise strictly monotonically across all five clipping bounds, TPR @ 1% FPR remains at the random-guessing baseline for $C \le 1.0$ ($0.0095 \to 0.0093$, statistically indistinguishable) before rising monotonically for $C \ge 1.0$ ($0.0093 \to 0.0132 \to 0.0149 \to 0.0231$, reaching $0.0311$ in the unclipped baseline). Furthermore, the rate of increase differs dramatically: while global AUC surges by $0.0968$ from $C=1.0$ to $C=5.0$, TPR @ 1% FPR rises by only $0.0039$ ($0.0132 \pm 0.0025$) while recovering $+6.49$ percentage points of test accuracy ($45.34\%$), demonstrating that the perceived severity of empirical leakage is highly metric-dependent even though monotonic ordering for $C \ge 1.0$ is preserved. We demonstrate that epoch-1 binding ($100.0\%$ at $C \le 1.0$ vs. $\le 3.1\%$ at $C \ge 5.0$) drives the underfitting mechanism, whereas epoch-50 clipped fractions ($24.3\%$ at $C=10.0$, $0.0\%$ at $C=50.0$) reflect downstream convergence.
3. **Unifying Generalization Gap vs. Attack AUC Analysis**: We pool all 15 experimental conditions spanning baseline, noise-varied DP-SGD, and pure clipping. We show that the two mechanisms co-locate in the low-gap overlap regime ($\Delta_{\text{gen}} \in [1.91\%, 3.43\%]$) with overlapping bootstrap confidence intervals, and separate at high gap where residual clipping at $C=50.0$ still costs the attacker ~0.016 AUC relative to the unclipped baseline. This unifies the findings: neither mechanism defends directly; both govern attack vulnerability primarily by controlling the generalization gap.
4. **Methodological Transparency & Limitations**: We demonstrate that confidence is rank-equivalent to loss under a global threshold, identify near-perfect loss–mentr rank collinearity (mean Spearman $\rho = 0.9787$), and emphasize that our defense findings are contingent on operating in the tight-clipping underfitting regime under global-threshold attacks, motivating future evaluation under per-example calibrated attacks (LiRA).

---

## 2. Threat Model & Preliminaries

### 2.1 Differential Privacy Framework
A randomized mechanism $\mathcal{M}$ satisfies $(\epsilon, \delta)$-Differential Privacy (Dwork et al., 2006) if for all neighboring datasets $D, D'$ differing by at most one sample ($|D \Delta D'| = 1$) and all measurable sets of outcomes $\mathcal{S} \subseteq \text{Range}(\mathcal{M})$:
$$\mathbb{P}[\mathcal{M}(D) \in \mathcal{S}] \le e^\epsilon \, \mathbb{P}[\mathcal{M}(D') \in \mathcal{S}] + \delta$$

In our experiments, privacy accounting is executed using Rényi Differential Privacy (RDP; Mironov, 2017) tracked via Opacus's `PrivacyEngine` with target failure probability $\delta = 10^{-5}$ ($< 1/N$).

### 2.2 Membership Inference Attacks (MIA)
We adopt the standard membership inference formulation (Yeom et al., 2018). An adversary has black-box query access to a trained model $\theta$ and seeks to predict whether a target record $(x, y)$ was included in the training set $D_{\text{train}}$ ($\text{member}$, label $1$) or drawn from the same underlying distribution but withheld during training ($D_{\text{test}}$, label $0$).

We evaluate three standard signal metrics, all formulated as scoring functions $\phi(x, y; \theta)$ where higher values denote greater likelihood of membership:
1. **Loss Thresholding**:
   $$\phi_{\text{loss}}(x, y; \theta) = -\ell(\theta; x, y) = \log p_\theta(y \mid x)$$
   Members typically exhibit lower cross-entropy loss than non-members; taking the negative loss ensures that higher scores correspond to higher membership likelihood.
2. **Confidence Thresholding**:
   $$\phi_{\text{conf}}(x, y; \theta) = p_\theta(y \mid x)$$
   The softmax probability assigned directly to the ground-truth class $y$.  
   *Rank-Equivalence Consistency Check*: Because cross-entropy loss is defined as $\ell(\theta; x, y) = -\log p_\theta(y \mid x)$, confidence is an exact, strictly monotonic transformation of negative loss: $p_\theta(y \mid x) = \exp(-\ell(\theta; x, y))$. Under a global decision threshold, any threshold applied to $\phi_{\text{conf}}$ maps one-to-one to a threshold on $\phi_{\text{loss}}$. Consequently, confidence produces **identical pairwise rankings and mathematically identical ROC curves and AUC values to loss**. We explicitly retain confidence throughout all tables and artifacts as an empirical rank-equivalence consistency check rather than an independent test signal, alerting the reader that matching values across loss and confidence reflect mathematical consistency rather than a duplicate reporting error.
3. **Modified Prediction Entropy (Mentr; Song & Mittal, 2021)**:
   $$\text{Mentr}(p_\theta(x), y) = -(1 - p_\theta(y \mid x))\log(p_\theta(y \mid x)) - \sum_{k \ne y} p_\theta(k \mid x) \log(1 - p_\theta(k \mid x))$$
   $$\phi_{\text{mentr}}(x, y; \theta) = -\text{Mentr}(p_\theta(x), y)$$
   Mentr penalizes high entropy on incorrect classes and low entropy on the true class. Members exhibit lower modified entropy; negating it ensures that higher scores indicate greater membership likelihood. Unlike confidence, mentr incorporates probabilities across non-true classes, providing a structurally distinct probe of the output distribution.

**Threat Model Scoping**: All three attacks operate strictly under a **global-threshold threat model**: a single decision threshold is swept across the entire dataset to evaluate classification separation. This setup evaluates whether aggregate prediction margins distinguish members from non-members. It does *not* cover per-example calibrated attacks such as the Likelihood Ratio Attack (LiRA; Carlini et al., 2022), which calibrate sample-specific decision boundaries using shadow model ensembles.

### 2.3 Evaluation Metrics
- **Test Accuracy**: Multi-class top-1 classification accuracy on held-out test data.
- **Generalization Gap**: $\Delta_{\text{gen}} = \text{Acc}_{\text{member}} - \text{Acc}_{\text{nonmember}}$ (the top-1 accuracy difference between training members and held-out non-members). We standardize on this member–nonmember formulation because it reflects the exact distributional separation that membership inference attacks exploit.
- **Attack AUC**: Area Under the Receiver Operating Characteristic (ROC) curve across all classification thresholds. An AUC of $0.50$ denotes random guessing; $1.0$ indicates perfect vulnerability.
- **Attack Accuracy**: Balanced classification accuracy at the optimal threshold chosen on the receiver curve.
- **TPR at 1% FPR**: True Positive Rate achievable at a strict False Positive Rate of $0.01$, capturing high-confidence identification capability.

---

## 3. Experimental Setup & Methodology

### 3.1 Induced Overfitting on Subsampled CIFAR-10
To evaluate membership inference meaningfully, the target model must be susceptible to memorization. On standard benchmarks where models generalize exceptionally well (such as full MNIST), models exhibit near-zero generalization gaps, leaving threshold attacks powerless regardless of privacy mechanisms.

To create a realistic testbed, we train on a subsampled subset of CIFAR-10 ($N_{\text{train}} = 5{,}000$) for 50 epochs using batch size $B=64$ and learning rate $\eta = 0.05$. This induces a substantial natural memorization gap in the non-private baseline ($\Delta_{\text{gen}} = +49.11\% \pm 0.75\%$, member accuracy $100.0\%$, non-member accuracy $50.89\% \pm 0.75\%$, full test accuracy $51.13\% \pm 0.30\%$), establishing a high-signal environment for MIA (baseline Attack AUC $= 0.8578$).

As a negative control, we train on the full MNIST dataset ($N=60{,}000$, 5 epochs, $\eta=0.05$), where the baseline generalizes to $99.05\%$ with virtually zero memorization gap.

### 3.2 Model Architectures
Both models avoid Batch Normalization (which leaks per-sample statistics across batches and complicates per-sample gradient computation) and pass Opacus's `ModuleValidator`:
- **`SampleCNN` (MNIST)**: 2 convolutional layers (16 and 32 filters, $5\times 5$, ReLU, MaxPool2d) followed by 2 fully connected layers (128 and 10 units).
- **`CifarCNN` (CIFAR-10)**: 4 convolutional layers (two 32-channel, two 64-channel, $3\times 3$, padding 1, ReLU, MaxPool2d) followed by a 256-unit dense layer and a 10-unit linear classification head.

Checkpoints unwrap Opacus's `GradSampleModule` (`model._module.state_dict()`), enabling standalone loading without Opacus runtime dependencies.

### 3.3 Statistical Protocol, Bootstrap CIs & False Discovery Rate Control
To ensure rigorous statistical validation, all experiments adhere to the following protocol:
- **Seed-Independent Splits**: Member and non-member indices are partitioned deterministically per seed (`seed42`, `seed43`, `seed44`) and persisted to disk. Strict runtime assertions ensure that downstream MIA evaluators strictly match the member split corresponding to the checkpoint's seed.
- **Bootstrap Confidence Intervals**: For every checkpoint and scoring signal, 1,000 bootstrap resamples are computed over the concatenated member and non-member score distributions to generate $95\%$ percentile confidence intervals $[CI_{\text{lower}}, CI_{\text{upper}}]$.
- **Condition for Distinguishability**: An individual attack run is nominally distinguishable from random guessing if $CI_{\text{lower}} > 0.50$ and the corresponding two-sided Wilcoxon-Mann-Whitney $p$-value satisfies $p < 0.05$.
- **Multiple Testing & Corrected Benjamini-Hochberg FDR Family**: Because confidence is strictly rank-equivalent to cross-entropy loss under a global threshold, 45 of the 135 tests are exact mathematical duplicates. Controlling false discoveries across redundant tests artificially distorts the family size. We therefore establish the multiple testing family over the **90 non-duplicate tests** (45 checkpoints $\times$ 2 non-redundant signals: loss and mentr) against the null hypothesis $\mathcal{H}_0: \text{AUC} = 0.50$. At a standard uncorrected significance threshold of $\alpha = 0.05$, $90 \times 0.05 = 4.5$ false discoveries would be expected by chance alone. Applying the Benjamini-Hochberg procedure (BH-FDR; Benjamini & Hochberg, 1995) at $q = 0.05$ over the 90 non-duplicate tests confirms:
  - For two-sided tests ($\mathcal{H}_0: \text{AUC} = 0.50$): **77 tests** reach uncorrected $p < 0.05$, and **75 remain significant** under BH-FDR ($83.33\%$).
  - For one-sided tests ($\mathcal{H}_1: \text{AUC} > 0.50$): **81 tests** reach uncorrected $p < 0.05$, and **79 remain significant** under BH-FDR ($87.78\%$).
  - Recomputing over the 90-test family results in **exactly zero verdict changes** compared to the 135-test evaluation, demonstrating robust stability.
  - Crucially, even 90 overstates the number of statistically independent tests: computing the Spearman rank correlation between the loss and mentr score vectors across all 45 checkpoints yields a mean correlation of $\bar{\rho} = 0.9787$ (exceeding $\rho > 0.993$ for all trained models with test accuracy $> 20\%$, peaking at $\rho = 0.9995$ on the baseline). This demonstrates that mentr and loss are near-perfectly collinear in their sample rankings.
- **Hardware Non-Determinism & Error Bar Interpretation**: Per-sample gradient computation under Opacus is not bit-deterministic on GPU due to non-deterministic atomic operations, CUDA memory allocator state, and cuDNN heuristic convolution kernel selection across 50 epochs. An empirical re-run of the identical nominal configuration and seed ($C=1.0, \sigma=0.0$, Seed 42) yielded a test accuracy variation of up to ~1.5 percentage points ($36.10\%$ in an initial single-run ablation vs. $37.66\%$ in the multi-seed sweep, with maximum weight divergence of $0.0222$). Because run-to-run irreproducibility at fixed seed under GPU non-determinism (~1.5 pp) is larger than the inter-seed standard deviation reported for this condition ($\pm 0.85\%$), we explicitly caution that all reported $\pm$ standard deviations in both documents reflect **seed variance only** across our experimental runs, not an absolute lower bound on the variation that an independent replicator might observe without strict deterministic flags (`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Crucially, however, this variation affects only absolute accuracy levels; the core empirical findings—complete collapse of the generalization gap ($\le +3.43\%$) and near-random threshold Attack AUC ($0.5186 - 0.5239$)—are invariant across runs.

---

## 4. Experimental Results

### 4.1 MNIST: The Negative Control
On full MNIST, models generalize with high fidelity, preventing threshold MIA from finding signal:

| Model | $\sigma$ | $\epsilon$ ($\delta=10^{-5}$) | Test Accuracy | Test Loss | Loss Attack AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.0 | $\infty$ | **99.05%** | 0.0309 | **0.498** |
| **DP-SGD** | 0.3 | 31.38 | 91.30% | 0.4661 | 0.499 |
| **DP-SGD** | 0.7 | 1.03 | 91.09% | 0.4685 | 0.497 |
| **DP-SGD** | 1.1 | 0.30 | 90.31% | 0.5226 | 0.498 |
| **DP-SGD** | 3.0 | 0.09 | 80.90% | 1.7463 | 0.499 |
| **DP-SGD** | 5.0 | 0.05 | 68.45% | 5.3000 | 0.501 |

Across all $\epsilon$ values from $\infty$ down to $0.05$, Attack AUC sits at $0.498 \pm 0.002$. Because MNIST generalizes without overfitting, membership inference has zero purchase, validating our motivation to focus on CIFAR-10 for primary evaluation.

---

### 4.2 CIFAR-10 Epsilon Sweep (Seeds 42, 43, 44)

We evaluate nine Gaussian noise multipliers ($\sigma \in [0.30, 5.00]$) alongside the unconstrained baseline at fixed clipping norm $C = 1.0$, evaluating all three membership signals (loss, confidence, mentr) with Benjamini-Hochberg FDR correction ($q=0.05$):

| Model | Noise Mult ($\sigma$) | $\epsilon$ ($\delta=10^{-5}$) | Test Acc (mean ± std) | Gen Gap (mean ± std) | Loss AUC (95% CI) | Conf AUC (95% CI) | Mentr AUC (95% CI) | TPR @ 1% FPR (mean ± std) | Distinguishable (BH-FDR $q=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.00 | $\infty$ | **51.13% ± 0.30%** | **+49.11% ± 0.75%** | **0.8578** [0.8530, 0.8623] | **0.8578** [0.8530, 0.8623] | **0.8579** [0.8531, 0.8624] | 0.0311 ± 0.0019 | **Yes (3/3)** |
| **DP-SGD** | 0.30 | 193.50 | 38.76% ± 0.98% | +2.49% ± 0.91% | **0.5216** [0.5149, 0.5283] | **0.5216** [0.5149, 0.5283] | **0.5216** [0.5150, 0.5285] | 0.0095 ± 0.0008 | Yes (3/3) |
| **DP-SGD** | 0.50 | 32.53 | 38.54% ± 0.95% | +2.53% ± 0.90% | **0.5207** [0.5140, 0.5276] | **0.5207** [0.5140, 0.5276] | **0.5205** [0.5137, 0.5272] | 0.0100 ± 0.0006 | Yes (3/3) |
| **DP-SGD** | 0.70 | 11.29 | 38.53% ± 0.99% | +2.36% ± 0.81% | **0.5198** [0.5136, 0.5266] | **0.5198** [0.5136, 0.5266] | **0.5195** [0.5132, 0.5263] | 0.0096 ± 0.0015 | Yes (3/3) |
| **DP-SGD** | 0.90 | 6.05 | 38.40% ± 1.02% | +1.92% ± 0.91% | **0.5199** [0.5136, 0.5265] | **0.5199** [0.5136, 0.5265] | **0.5196** [0.5133, 0.5263] | 0.0105 ± 0.0019 | Yes (3/3) |
| **DP-SGD** | 1.10 | 4.09 | 38.31% ± 1.08% | +2.34% ± 0.95% | **0.5188** [0.5124, 0.5256] | **0.5188** [0.5124, 0.5256] | **0.5186** [0.5123, 0.5253] | 0.0121 ± 0.0032 | Partial (2/3) |
| **DP-SGD** | 1.50 | 2.51 | 36.93% ± 0.64% | +2.83% ± 0.42% | **0.5218** [0.5154, 0.5285] | **0.5218** [0.5154, 0.5285] | **0.5219** [0.5154, 0.5288] | 0.0111 ± 0.0032 | Yes (3/3) |
| **DP-SGD** | 2.00 | 1.71 | 31.40% ± 0.18% | +3.43% ± 0.24% | **0.5235** [0.5173, 0.5302] | **0.5235** [0.5173, 0.5302] | **0.5239** [0.5177, 0.5305] | 0.0113 ± 0.0007 | Yes (3/3) |
| **DP-SGD** | 3.00 | 1.05 | 20.24% ± 0.85% | +0.87% ± 0.78% | **0.5090** [0.5025, 0.5157] | **0.5090** [0.5025, 0.5157] | **0.5092** [0.5028, 0.5158] | 0.0110 ± 0.0007 | Partial (1/3) |
| **DP-SGD** | 5.00 | 0.59 | 13.75% ± 1.52% | +0.38% ± 0.74% | **0.4995** [0.4930, 0.5061] | **0.5019** [0.4963, 0.5073] | **0.4980** [0.4921, 0.5042] | 0.0113 ± 0.0016 | **No (0/3)** |

##### Key Analytical Observations:
1. **Multi-Signal Concordance & Clipping-Driven Mitigation**: Transitioning from the unconstrained baseline to any DP-SGD configuration instantly suppresses Attack AUC across all three signals by over $0.336$ points (dropping from $0.8578 - 0.8579$ to $0.5216$ at $\sigma=0.30$) and reduces the generalization gap from $+49.11\%$ to $+2.49\%$. Crucially, as established in the clipping ablation (§4.3), this massive suppression is driven entirely by the fixed $C = 1.0$ per-sample gradient clipping shared across all DP rows—which alone yields an AUC of $0.5230$ and gap of $+3.37\%$ at $\sigma = 0.0$—while scaling noise adds essentially no empirical defense on top. Loss, confidence, and mentr behave almost identically across all conditions.
2. **Epsilon Invariance Across Practical Regimes (>110x Range)**: Between $\sigma=0.30$ ($\epsilon = 193.50$) and $\sigma=2.00$ ($\epsilon = 1.71$), Attack AUC across all three signals remains locked within an extremely narrow band ($0.5186 - 0.5239$). Tightening the formal theoretical bound by more than two orders of magnitude produces negligible change in empirical attack performance. The fact that this null result holds across the entire signal family (loss, confidence, and mentr) substantially strengthens the finding relative to a single-signal evaluation.
3. **The Utility Cliff**: Test accuracy exhibits an initial plateau between $\sigma=0.30$ and $\sigma=1.10$ (~$38.3\% - 38.8\%$), representing a fixed ~$12.5$ percentage point utility penalty relative to the baseline. Past $\sigma=1.50$ ($\epsilon < 2.51$), utility degrades severely: accuracy drops to $31.40\%$ at $\sigma=2.00$, $20.24\%$ at $\sigma=3.00$, and collapses to $13.75\%$ (near random guess) at $\sigma=5.00$.
4. **Per-Seed Significance Dynamics and Transition to Exact Randomness**:
   - At $\sigma = 3.00$ ($\epsilon = 1.05$), the configuration yields a **Partial (1/3)** BH-FDR verdict: Seed 43 ($0.5149$) clears the BH-corrected threshold ($p = 0.0098, q = 0.0129$), while Seed 42 ($0.5023$, $p = 0.6919, q = 0.7131$) and Seed 44 ($0.5098$, $p = 0.0899, q = 0.1012$) fail to clear BH for loss and confidence.
   - Only at $\sigma = 5.00$ ($\epsilon = 0.59$) does the model reach exact statistical randomness: all three random seeds fail to clear BH ($0/3$; Seed 42: $0.5048$, Seed 43: $0.4969$, Seed 44: $0.4968$), and the 95% bootstrap CI spans $0.50$ across all three signals ($[0.4930, 0.5061]$ for loss, $[0.4963, 0.5073]$ for confidence, $[0.4921, 0.5042]$ for mentr). This proves exact empirical indistinguishability from a coin toss, but at the cost of catastrophic model collapse ($13.75\%$ test accuracy).

---

### 4.3 CIFAR-10 Gradient Clipping Ablation: Clip Norm as a Dose-Response Curve ($\sigma = 0.0$, Seeds 42, 43, 44)

To determine whether the $0.34$-point drop in Attack AUC is caused by noise or gradient clipping, we evaluate five clipping thresholds ($C \in \{0.5, 1.0, 5.0, 10.0, 50.0\}$) at strictly zero noise ($\sigma = 0.0$) across all three membership signals. This sweep acts as a direct **dose-response curve** for the optimization constraint imposed by per-sample gradient projection:

| Condition | Clip Norm ($C$) | Test Acc (mean ± std) | Train Loss (mean ± std) | Gen Gap (Train-Test) | Gen Gap (Member-Nonmember, MIA Target) | Loss AUC (95% CI) | Conf AUC (95% CI) | Mentr AUC (95% CI) | TPR @ 1% FPR (mean ± std) | Distinguishable (BH-FDR $q=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Clip** | **0.5** | $35.40\% \pm 0.73\%$ | $1.8759 \pm 0.0419$ | $+1.76\% \pm 0.37\%$ | $+1.91\% \pm 1.00\%$ | **$0.5138$** $[0.5071, 0.5201]$ | **$0.5138$** $[0.5071, 0.5201]$ | **$0.5135$** $[0.5068, 0.5199]$ | $0.0095 \pm 0.0008$ | Partial (1/3) |
| **Pure Clip** | **1.0** | $38.85\% \pm 0.85\%$ | $1.8445 \pm 0.0395$ | $+3.47\% \pm 0.56\%$ | $+3.37\% \pm 0.74\%$ | **$0.5230$** $[0.5166, 0.5298]$ | **$0.5230$** $[0.5166, 0.5298]$ | **$0.5231$** $[0.5165, 0.5297]$ | $0.0093 \pm 0.0002$ | **Yes (3/3)** |
| **Pure Clip** | **5.0** | $45.34\% \pm 1.28\%$ | $1.5296 \pm 0.1710$ | $+18.09\% \pm 2.79\%$ | $+18.91\% \pm 2.95\%$ | **$0.6198$** $[0.6136, 0.6262]$ | **$0.6198$** $[0.6136, 0.6262]$ | **$0.6201$** $[0.6140, 0.6266]$ | $0.0132 \pm 0.0025$ | **Yes (3/3)** |
| **Pure Clip** | **10.0** | $46.72\% \pm 0.82\%$ | $0.3722 \pm 0.0302$ | $+45.66\% \pm 0.64\%$ | $+46.11\% \pm 0.77\%$ | **$0.7780$** $[0.7726, 0.7834]$ | **$0.7780$** $[0.7726, 0.7834]$ | **$0.7786$** $[0.7731, 0.7838]$ | $0.0149 \pm 0.0017$ | **Yes (3/3)** |
| **Pure Clip** | **50.0** | $49.94\% \pm 0.61\%$ | $0.0005 \pm 0.0006$ | $+50.04\% \pm 0.62\%$ | $+50.14\% \pm 1.02\%$ | **$0.8421$** $[0.8370, 0.8471]$ | **$0.8421$** $[0.8370, 0.8471]$ | **$0.8422$** $[0.8371, 0.8472]$ | $0.0231 \pm 0.0010$ | **Yes (3/3)** |
| *Baseline (no DP)* | *None* | *$51.13\% \pm 0.30\%$* | *$0.0001 \pm 0.0001$* | *$+48.87\% \pm 0.30\%$* | *$+49.11\% \pm 0.75\%$* | **$0.8578$** $[0.8530, 0.8623]$ | **$0.8578$** $[0.8530, 0.8623]$ | **$0.8579$** $[0.8531, 0.8624]$ | $0.0311 \pm 0.0019$ | *Yes (3/3)* |
| *Full DP-SGD* | *1.0 ($\sigma=1.1$)* | *$38.31\% \pm 1.08\%$* | *$1.9494 \pm 0.0410$* | *$+2.13\% \pm 0.81\%$* | *$+2.34\% \pm 0.95\%$* | **$0.5188$** $[0.5124, 0.5256]$ | **$0.5188$** $[0.5124, 0.5256]$ | **$0.5186$** $[0.5123, 0.5253]$ | $0.0121 \pm 0.0032$ | *Partial (2/3)* |

#### Per-Sample Gradient Norm Diagnostics and Binding Dynamics

To inspect how gradient clipping affects internal optimization capacity across training epochs, we log per-sample gradient $L_2$ norm statistics and the empirical clipped fraction at Epoch 1 and Epoch 50:

| Clip Norm ($C$) | Epoch 1 Mean Norm | Epoch 1 Median Norm | Epoch 1 p95 Norm | Epoch 1 Clipped Fraction | Epoch 50 Mean Norm | Epoch 50 Median Norm | Epoch 50 p95 Norm | Epoch 50 Clipped Fraction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5** | $1.63 \pm 0.07$ | $1.62 \pm 0.07$ | $1.96 \pm 0.14$ | **$100.0\% \pm 0.0\%$** | $58.06 \pm 7.42$ | $54.84 \pm 6.65$ | $112.22 \pm 18.00$ | **$99.7\% \pm 0.2\%$** |
| **1.0** | $1.70 \pm 0.07$ | $1.67 \pm 0.07$ | $2.10 \pm 0.15$ | **$100.0\% \pm 0.0\%$** | $91.14 \pm 14.08$ | $84.61 \pm 11.68$ | $200.05 \pm 36.35$ | **$98.1\% \pm 0.7\%$** |
| **5.0** | $2.17 \pm 0.32$ | $1.85 \pm 0.12$ | $4.07 \pm 1.57$ | $3.1\% \pm 3.9\%$ | $128.30 \pm 7.88$ | $62.49 \pm 13.75$ | $414.07 \pm 35.42$ | **$72.6\% \pm 2.4\%$** |
| **10.0** | $2.17 \pm 0.33$ | $1.85 \pm 0.12$ | $4.11 \pm 1.62$ | $0.3\% \pm 0.5\%$ | $58.91 \pm 3.61$ | $0.50 \pm 0.05$ | $481.59 \pm 27.12$ | **$24.3\% \pm 0.5\%$** |
| **50.0** | $2.17 \pm 0.33$ | $1.85 \pm 0.12$ | $4.10 \pm 1.60$ | $0.0\% \pm 0.0\%$ | $0.07 \pm 0.04$ | $0.005 \pm 0.0005$ | $0.23 \pm 0.02$ | **$0.0\% \pm 0.0\%$** |

#### Key Analytical Observations:
1. **The Dose-Response Curve: Accuracy and Vulnerability Rise Strictly Together**:
   As $C$ increases from $0.5$ to $50.0$, test accuracy increases monotonically ($35.40\% \to 38.85\% \to 45.34\% \to 46.72\% \to 49.94\%$), and Attack AUC rises monotonically in lockstep ($0.5138 \to 0.5230 \to 0.6198 \to 0.7780 \to 0.8421$). If evaluated strictly by Attack AUC, no intermediate operating point exists where utility has partly recovered while Attack AUC remains near $0.50$. The empirical privacy protection of clipping is physically inseparable from its underfitting utility penalty.

   *Metric-Dependent Leakage Dynamics at $C = 5.0$*: Evaluating $C = 5.0$ illustrates how metric choice alters the apparent severity of privacy leakage. At $C=5.0$, test accuracy recovers over half the lost baseline utility to $45.34\% \pm 1.28\%$ (+6.49 percentage points above $C=1.0$), with a generalization gap of $+18.91\% \pm 2.95\%$. Under global AUC, $C=5.0$ appears heavily compromised ($0.6198$ $[0.6136, 0.6262]$ vs. $0.5230$ at $C=1.0$, a surge of $+0.0968$). Under the low-FPR metric (TPR @ 1% FPR), attack success across all five clipping thresholds progresses as $0.0095 \pm 0.0008 \to 0.0093 \pm 0.0002 \to 0.0132 \pm 0.0025 \to 0.0149 \pm 0.0017 \to 0.0231 \pm 0.0010$ (rising to $0.0311 \pm 0.0019$ in the unclipped baseline). While $C=0.5$ and $C=1.0$ are statistically indistinguishable at the 1.0% random baseline, TPR @ 1% FPR rises monotonically for $C \ge 1.0$, and at $C=5.0$, it is statistically distinguishable from $C=1.0$ (non-overlapping error bars). However, while global AUC rises by nearly $0.10$ points, TPR @ 1% FPR rises by only $0.0039$. The low-FPR metric does not exonerate $C=5.0$, but demonstrates that the perceived magnitude of empirical leakage is strongly metric-dependent even though monotonic ordering for $C \ge 1.0$ is preserved.
2. **Causal Binding at Epoch 1 vs. Downstream Convergence at Epoch 50**:
   The mechanistic driver of underfitting is established at the very beginning of training. At Epoch 1, per-sample gradient norms are concentrated above $1.0$ (mean $1.63 - 1.70$), causing clipping to be **$100.0\% \pm 0.0\%$ binding from the first epoch for $C \le 1.0$**. This early bottleneck prevents the optimizer from fitting sample-specific features. Conversely, at $C \ge 5.0$, clipping is essentially inactive from the start ($3.1\% \pm 3.9\%$ clipped at $C = 5.0$, $0.3\% \pm 0.5\%$ at $C = 10.0$, and $0.0\% \pm 0.0\%$ at $C = 50.0$), allowing the network to freely minimize loss on training examples.

   The epoch-50 clipping diagnostics are the **downstream consequence of convergence, not the causal mechanism**. At $C = 50.0$, the model converges fully (train loss $0.0005 \pm 0.0006$, epoch-50 mean norm $0.07$, p95 $0.23$); per-sample gradients are small because the optimization has succeeded, so clipping cannot bind ($0.0\%$). At $C = 10.0$, the model reaches near-convergence (train loss $0.3722$), so epoch-50 mean gradient norm drops to $58.91$ with median $0.50$, resulting in a $24.3\% \pm 0.5\%$ clipped fraction. In stark contrast, at $C \le 1.0$, the model never converges (train loss $\approx 1.85 - 1.88$); the persistent large gradients at epoch 50 (mean norm $58.06$ at $C=0.5$, $91.14$ at $C=1.0$) keep clipping binding ($98.1\% - 99.7\%$). The late-stage unbinding at $C \ge 10.0$ is therefore a downstream symptom of successful fitting, whereas early binding at epoch 1 is the causal constraint.
3. **Strengthened Null Result for Output Entropy**:
   Modified prediction entropy ($\text{Mentr}$) tracks loss AUC with extreme fidelity across every clipping norm ($\Delta \text{AUC} \le 0.0006$ everywhere), and their per-sample score vectors exhibit a near-unity rank correlation (Spearman $\rho = 0.9917$ at $C=0.5$, $\rho = 0.9930$ at $C=1.0$, and $\rho = 0.9994$ at $C=50.0$). We frame this as a **critical positive finding**: once clipping-induced underfitting suppresses memorization, membership signal is genuinely absent from the entire multi-class output distribution, not merely obscured within the scalar loss.

---

### 4.4 Statistical vs. Practical Significance in Empirical Privacy Auditing

While Attack AUC values of ~0.52 under DP-SGD and pure clipping ($C=1.0$) frequently reject the null hypothesis of exact random guessing under high-sample tests (clearing the BH-FDR threshold with 10,000 evaluation pairs), **statistical distinguishability from chance must not be conflated with practical attack success**.

To understand why, we quantify what an AUC of $0.52$ means operationally:
- **Pairwise Ranking Accuracy**: Presented with one randomly chosen training member and one held-out non-member, a threshold attacker choosing the point with higher membership score will be correct only $52\%$ of the time—barely distinguishable from an unbiased $50\%$ coin flip.
- **Low-FPR Behavior (TPR @ 1% FPR)**: In practical security audits and adversarial settings, an attacker cannot afford an avalanche of false accusations. Decisions must be made at strict false positive rates (e.g., $\text{FPR} \le 0.01$). Across all evaluated DP-SGD configurations ($\sigma \in [0.30, 5.00]$) and pure clipping at $C=1.0$, the True Positive Rate at 1% FPR remains locked between $0.0093$ and $0.0121$—sitting directly at the $1.0\%$ baseline expected under purely random guessing.

This distinction strengthens our central thesis: the residual signal that gradient clipping leaves behind is detectable only through high-powered statistical aggregations, but is completely unusable for practical exploitation. 

Conversely, the non-private baseline demonstrates the limitations of global-threshold attacks: despite achieving an aggregate AUC of $0.8578$ (and $0.8535$ on seed 42), its TPR @ 1% FPR reaches only $0.0311$ ($0.0287$ on seed 42). Even under severe overfitting ($\Delta_{\text{gen}} \approx +49\%$), a global threshold is unable to identify members with high confidence at low false alarm rates, confirming that global-threshold attacks are inherently weak in the decision-relevant low-FPR regime.

---

### 4.5 Unifying Analysis: Attack AUC vs. Generalization Gap Across Mechanisms

The decisive test of the clipping-versus-noise decomposition is whether the two mechanisms operate through distinct privacy pathways or whether both act solely by modulating model underfitting. To address this, we pool all 15 evaluated experimental configurations—the non-private baseline, all 9 noise multipliers ($\sigma \in [0.3, 5.0]$ at fixed $C=1.0$), and all 5 clipping bounds ($C \in [0.5, 50.0]$ at $\sigma=0.0$)—spanning generalization gaps from $+0.38\%$ to $+50.14\%$ and Attack AUC from $0.4995$ to $0.8578$.

The resulting empirical relationship is visualized in **Figure 4 (`experiments/cifar10/results/gap_vs_auc.png`)**, which plots mean member–nonmember generalization gap ($\pm$ seed standard deviation) against mean Loss Attack AUC (with 95% bootstrap confidence intervals).

#### Table 4.4: Model Fits and Goodness of Fit ($R^2$) Across Mechanisms

| Group / Subset | Model Specification | Functional Form | $R^2$ | Model Parameters |
| :--- | :--- | :--- | :---: | :--- |
| **All Pooled ($N=15$)** | **3-Parameter Logistic** | $y = 0.50 + \frac{L}{1 + \exp(-k(\Delta_{\text{gen}} - x_0))}$ | **$0.9855$** | $L = 0.3463, k = 0.1224, x_0 = 25.17\%$ |
| **All Pooled ($N=15$)** | Logarithmic | $y = a + b \ln(\Delta_{\text{gen}})$ | **$0.8380$** | $a = 0.4716, b = 0.0787$ |
| **Clipping-Varied ($N=5$)** | 3-Parameter Logistic | $y = 0.50 + \frac{L}{1 + \exp(-k(\Delta_{\text{gen}} - x_0))}$ | **$0.9817$** | $L = 0.3449, k = 0.1036, x_0 = 26.35\%$ |
| **Clipping-Varied ($N=5$)** | Logarithmic | $y = a + b \ln(\Delta_{\text{gen}})$ | **$0.8872$** | $a = 0.4213, b = 0.0932$ |
| **Noise-Varied ($N=9$)** | 3-Parameter Logistic | $y = 0.50 + \frac{L}{1 + \exp(-k(\Delta_{\text{gen}} - x_0))}$ | **$0.9704$** | $L = 0.0209, k = 34.95, x_0 = 0.88\%$ |
| **Noise-Varied ($N=9$)** | Logarithmic | $y = a + b \ln(\Delta_{\text{gen}})$ | **$0.9845$** | $a = 0.5106, b = 0.0111$ |

*Note on Parameter Divergence and Fit Instability*: The steepness parameter $k$ diverges by more than two orders of magnitude between the pooled/clipping fits ($k \approx 0.10 - 0.12$) and the noise-varied fit ($k = 34.95, x_0 = 0.88\%$). The pooled fit is effectively dominated by the five clipping points, which supply almost all the dynamic range, while a 3-parameter logistic on five points is near-saturated and sensitive to fit instability (e.g., shifting from $k = 0.1232, x_0 = 25.07\%$ to $k = 0.1036, x_0 = 26.35\%$ across analysis runs). Consequently, the pooled $R^2 = 0.9855$ serves as a descriptive goodness-of-fit statistic dominated by the clipping points rather than evidence of a shared mathematical curve.

#### Key Analytical Conclusions:
1. **Primary Evidence: Co-Location in the Low-Gap Overlap Regime**:
   The primary evidence for mechanism co-location lies in the narrow overlap regime where both clipping and noise variations produce comparable generalization gaps ($\Delta_{\text{gen}} \in [1.91\%, 3.43\%]$):
   - Pure clipping at $C = 0.50$ ($\sigma=0.0$): $\Delta_{\text{gen}} = 1.91\% \pm 1.00\%$, $\text{AUC} = 0.5138$ $[0.5071, 0.5201]$.
   - Noise-varied at $\sigma = 0.90$ ($C=1.0$): $\Delta_{\text{gen}} = 1.92\% \pm 0.91\%$, $\text{AUC} = 0.5199$ $[0.5136, 0.5265]$.
   - Pure clipping at $C = 1.00$ ($\sigma=0.0$): $\Delta_{\text{gen}} = 3.37\% \pm 0.74\%$, $\text{AUC} = 0.5230$ $[0.5166, 0.5298]$.
   - Noise-varied at $\sigma = 2.00$ ($C=1.0$): $\Delta_{\text{gen}} = 3.43\% \pm 0.24\%$, $\text{AUC} = 0.5235$ $[0.5173, 0.5302]$.

   At these matched generalization gaps, the clipping-only and DP-SGD points co-locate directly, with fully overlapping 95% bootstrap confidence intervals. Crucially, this co-location test is strictly confined to a narrow gap band ($\le 3.43\%$) because varying Gaussian noise at fixed $C = 1.0$ cannot produce large generalization gaps.
2. **High-Gap Separation: Residual Clipping Costs the Attacker**:
   At the unconstrained high-gap end, the two mechanisms do not sit on a shared plateau; rather, they separate measurably:
   - Non-private baseline: $\Delta_{\text{gen}} = +49.11\% \pm 0.75\%$, $\text{AUC} = 0.8578$ $[0.8530, 0.8623]$.
   - Pure clipping at $C = 50.0$ ($\sigma=0.0$): $\Delta_{\text{gen}} = +50.14\% \pm 1.02\%$, $\text{AUC} = 0.8421$ $[0.8370, 0.8471]$.

   Despite exhibiting a higher generalization gap ($+50.14\%$ vs $+49.11\%$), the $C = 50.0$ model has a measurably lower Attack AUC ($0.8421$ vs $0.8578$), and their 95% bootstrap confidence intervals do not overlap. At matched or greater generalization gap, the clipped model is measurably less attackable: residual clipping still costs the attacker roughly $0.016$ AUC beyond what the generalization gap alone predicts.
3. **Reframing the Role of DP-SGD Components**:
   The empirical data reveal that the two mechanisms co-locate in the underfitting regime and separate at high gap. This demonstrates that **neither gradient clipping nor Gaussian noise defends against global-threshold membership inference directly; both govern threshold attack vulnerability primarily by dictating the generalization gap**, while residual clipping provides a modest additional reduction at high capacity.

---

## 5. Synthesis: The Two Orthogonal Mechanisms of DP-SGD (An Empirical Decomposition)

We emphasize that the orthogonal decomposition presented below is an **empirical finding** scoped specifically to our experimental testbed: a standard convolutional network (`CifarCNN`) trained on subsampled CIFAR-10 under global-threshold membership inference attacks across loss, confidence, and modified entropy signals. It is not presented as a universal theoretical theorem, law, or principle; no formal mathematical proof is offered.

Combining the findings from the $\epsilon$-sweep and clipping ablation yields a coherent, empirical understanding of how DP-SGD operates in this setting:

```text
                                 ┌──────────────────────────────────────────────┐
                                 │                 DP-SGD Step                  │
                                 └──────────────────────┬───────────────────────┘
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         ▼                                                             ▼
         ┌──────────────────────────────┐                              ┌──────────────────────────────┐
         │  Gradient Clipping (Bound C) │                              │    Gaussian Noise (Scale σ)  │
         └───────────────┬──────────────┘                              └───────────────┬──────────────┘
                         │                                                             │
                         ▼                                                             ▼
     Controls Utility & Underfitting Dynamics                      Controls Formal Mathematical Guarantees
   - Throttles optimization capacity                             - Perturbs aggregated batch gradient
   - Prevents fitting outlier samples (C <= 1.0)                 - Establishes (ε, δ)-Differential Privacy
   - Suppresses Generalization Gap (+49% -> <= +3.4%)            - Decisive for formal certificate: ε in [0.59, 193]
   - Dictates empirical MIA vulnerability: 0.32 AUC swing        - Empirical MIA variation: <= 0.02 AUC (near-zero)
     across dose-response C in [0.5, 50.0] (0.5138 -> 0.8421)
```

### 5.1 Empirical Orthogonal Decomposition Matrix

| Mechanism | Parameter | Primary Operational Role | Governs | Empirical MIA Impact (3 Signals) | Formal Privacy Impact |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **Gradient Clipping** | $C$ | Optimization capacity throttling / underfitting induction | Utility vs. Generalization Gap vs. Empirical Vulnerability | **Decisive**: $0.32$ AUC swing across $C$ ($0.5138 \to 0.8421$) | None ($\epsilon = \infty$) |
| **Gaussian Noise** | $\sigma$ | Stochastic gradient perturbation | Theoretical Worst-Case Bounds | **Negligible**: $< 0.006$ AUC variation across $110\times \epsilon$ ($\sigma \in [0.3, 2.0]$); $\le 0.024$ across full sweep confounded by utility collapse | **Decisive**: $\epsilon \in [0.59, 193.50]$ |

### 5.2 Why Does Noise Add No Empirical Defense at C=1.0?
The intuition is geometric: membership inference exploits the difference in how a model responds to points it has memorized versus points it has not seen. 

In standard SGD, unclipped gradient updates allow high-loss outlier points to exert disproportionately large updates on the model parameters. Over 50 epochs, the network memorizes these individual points, driving training loss near zero while test loss diverges (Generalization Gap $+49.11\%$). The global-threshold MIA exploits this macroscopic separation directly across loss, confidence, and entropy.

When per-sample gradient clipping ($C=1.0$) is applied, the optimizer cannot allocate high parameter capacity to outlier points. The model is forced to prioritize consensus features shared across the mini-batch, inducing underfitting. As a result, sample memorization is halted at the threshold of empirical train/test equivalence: the generalization gap collapses to $+3.37\%$. 

Because clipping at $C=1.0$ has already eliminated the macroscopic distribution gap between members and non-members across all three signals through clipping-induced underfitting, **there is simply no residual memorization signal remaining for Gaussian noise to mask**. Additive noise injects variance that degrades test accuracy (from $38.76\%$ at $\sigma=0.30$ to $13.75\%$ at $\sigma=5.00$), but cannot defend against a signal that has already been erased by the optimization bottleneck.

---

## 6. Core Paper Visualizations

The four central figures substantiating this multi-signal framework are:

1. **Figure 1: The Privacy–Utility Frontier (`privacy_utility_curve.png`)**  
   *Description*: Test accuracy vs. privacy budget $\epsilon$ across 10 multi-seed conditions with standard deviation error bars. Shows the initial accuracy plateau at $\approx 38.3\% - 38.8\%$ down to $\epsilon = 4.09$ ($\sigma=1.10$), followed by a sharp cliff at $\epsilon \approx 2.51$ ($\sigma=1.50$), collapsing to $13.75\%$ at $\epsilon = 0.59$.
2. **Figure 2: Empirical Multi-Signal Attack Invariance Across Epsilon (`multisignal_auc_vs_epsilon.png`)**  
   *Description*: Attack AUC across Loss, Confidence, and Modified Prediction Entropy (Mentr) vs. $\epsilon$ featuring 95% bootstrap confidence bands. Depicts flat, co-linear lines across all privacy budgets ($0.4980 - 0.5239$), highlighting the $0.34$-point step function between the unclipped baseline ($0.8578 - 0.8579$) and all DP-SGD configurations.
3. **Figure 3: Coupled Clipping Dynamics Across Signals (`multisignal_auc_vs_clip_norm.png` & `multisignal_roc_curves.png`)**  
   *Description*: Attack AUC as a function of clipping norm $C \in [0.5, 50.0]$ at $\sigma=0.0$ across all three signals alongside linear and log-log ROC curves. Demonstrates that loss, confidence, and mentr co-evolve monotonically across clipping norms, confirming that clipping alone suppresses the attack surface across all three signals when $C \le 1.0$.
4. **Figure 4: Attack AUC vs. Generalization Gap Collapse Across Mechanisms (`gap_vs_auc.png`)**  
   *Description*: Attack AUC plotted against member–nonmember generalization gap $\Delta_{\text{gen}}$ across all 15 evaluated configurations (baseline, 9 noise multipliers, 5 clipping norms). Shows that pure clipping and noise-varied DP-SGD co-locate within overlapping 95% bootstrap confidence intervals in the narrow low-gap overlap regime ($\Delta_{\text{gen}} \in [1.91\%, 3.43\%]$), while separating at high gap where residual clipping at $C=50.0$ costs the attacker ~0.016 AUC relative to the unclipped baseline. A pooled 3-parameter logistic fit is shown for descriptive reference ($R^2 = 0.9855$).

---

## 7. Limitations & Future Work

We clearly acknowledge the methodological scope and limitations of this study:

1. **Global-Threshold Attacks and Low-FPR Weakness**:
   All three signals evaluated in this work (loss, confidence, and modified prediction entropy) operate under a global-threshold threat model: a single decision threshold is swept across the entire dataset. While this effectively measures aggregate prediction margin separation, global-threshold attacks are known to suffer from severe sensitivity limits in the decision-relevant low-FPR regime ($\text{FPR} \le 0.01$), because a single global threshold cannot account for per-sample intrinsic difficulty or variance.
2. **Potential for Stronger State-Dependent Attacks (LiRA & Shadow Models)**:
   Advanced membership inference methodologies—specifically Likelihood Ratio Attacks (LiRA; Carlini et al., 2022) and shadow model classifiers (Shokri et al., 2017)—fit parametric models to query outputs across out-of-bag shadow ensembles, establishing sample-specific calibrated thresholds.
   - *Hypothesis for Future Work*: While clipping at $C=1.0$ completely neutralizes global-threshold attacks across all three signals, it remains an open question whether subtle, sample-specific parameter representations persist in un-noised models ($\sigma=0.0$) that a calibrated attack like LiRA could detect, and which calibrated Gaussian noise ($\sigma > 0$) provably disrupts.
   - *Framing*: This observation highlights the value of our baseline findings: by demonstrating that global-threshold attacks across the entire signal family are completely blind to noise scaling once clipping is enforced, we establish a rigorous reference point against which per-example calibrated attacks can be systematically benchmarked.
3. **Contingency on the Tight-Clipping Underfitting Regime**:
   Our central finding—that clipping suppresses global-threshold MIA vulnerability—is strictly contingent on operating in the tight-clipping regime ($C \le 1.0$) where clipping prevents fitting and collapses the generalization gap. When the clipping threshold is relaxed to $C \ge 5.0$ or $C=50.0$, gradient clipping remains active in the training loop yet ceases to provide defense (yielding Attack AUC of $0.8421$, measurably below the $0.8578$ unclipped baseline despite a slightly higher generalization gap of $+50.14\%$ vs $+49.11\%$ due to residual clipping constraint). Thus, clipping per se is not a privacy defense; rather, the empirical protection is an indirect byproduct of clipping-induced underfitting.
4. **Hardware Non-Determinism and Replicability at Fixed Seed**:
   Per-sample gradient computation under Opacus on GPU does not produce bit-deterministic outputs under standard execution due to non-deterministic atomic additions, CUDA allocator state, and cuDNN heuristic algorithm selection. Repeated runs at fixed seed vary by up to ~1.5 percentage points in test accuracy (e.g., $36.10\%$ vs. $37.66\%$ at $C=1.0, \sigma=0.0$, Seed 42), which exceeds the inter-seed standard deviation ($\pm 0.85\%$). Reported standard deviations thus capture seed-to-seed variance across a single execution pass, not total replicator variance. Bit-level reproduction requires setting `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

---

## 8. Conclusion

This paper presents an empirical deconstruction of Differential Privacy in deep learning under Membership Inference Attacks. By evaluating 45 models across both noise-multiplier and gradient-clipping sweeps with multi-seed bootstrap confidence intervals and Benjamini-Hochberg FDR correction across 90 non-duplicate tests, we show that:
1. Across practical operating regimes ($\sigma \in [0.30, 2.00]$), scaling Gaussian noise in DP-SGD tightens formal $(\epsilon, \delta)$-DP guarantees from $\epsilon = 193.50$ down to $\epsilon = 1.71$ (over $110\times$), but produces zero measurable reduction in global-threshold MIA vulnerability across loss, confidence, and modified entropy signals ($0.5186 - 0.5239$). While extreme noise ($\sigma = 5.00$, $\epsilon = 0.59$) drives Attack AUC to exact statistical randomness ($0.4995$ [0.4930, 0.5061], 0/3 BH significant), this marginal empirical gain is entirely confounded by catastrophic model collapse to $13.75\%$ test accuracy (random guess).
2. Empirical global-threshold MIA mitigation is driven by **clipping-induced underfitting**: per-sample gradient clipping tight enough to prevent memorization ($C \le 1.0$) crushes the generalization gap ($+49.11\% \to \le +3.43\%$) and neutralizes all three attack signals, but exacts an unavoidable 12.5 percentage point utility penalty. When clipping is relaxed ($C \ge 5.0$), memorization and vulnerability immediately re-emerge ($0.8421$ AUC at $C=50.0$).
3. Comparing configurations across generalization gaps reveals that pure clipping and noise-varied DP-SGD co-locate in the narrow low-gap overlap regime ($\Delta_{\text{gen}} \in [1.91\%, 3.43\%]$) with overlapping bootstrap confidence intervals, but separate at high gap where residual clipping costs the attacker ~0.016 AUC, demonstrating that empirical threshold attack vulnerability is mediated primarily by the generalization gap rather than the specific DP mechanism employed.
4. Empirical privacy evaluations must decouple clipping dynamics from noise injection to avoid falsely attributing clipping-induced optimization constraints and underfitting to formal differential privacy noise mechanisms.

We emphasize that this empirical orthogonal decomposition is an empirical finding scoped specifically to `CifarCNN` on subsampled CIFAR-10 under global-threshold attacks, establishing a necessary foundation for future investigations into per-example calibrated attacks such as LiRA.

---

## References

- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 308–318.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300.
- Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership inference attacks from first principles. *2022 IEEE Symposium on Security and Privacy (S&P)*, 1897–1914.
- Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. *Theory of Cryptography Conference (TCC)*, 265–284.
- Mironov, I. (2017). Rényi differential privacy. *2017 IEEE 30th Computer Security Foundations Symposium (CSF)*, 263–275.
- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *2017 IEEE Symposium on Security and Privacy (S&P)*, 3–18.
- Song, C., & Mittal, P. (2021). Systematic evaluation of privacy risks of machine learning models. *Proceedings of the 30th USENIX Security Symposium (USENIX Security 21)*, 2615–2632.
- Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy risk in machine learning: Analyzing the connection to overfitting. *2018 IEEE 31st Computer Security Foundations Symposium (CSF)*, 268–282.
- Yousefpour, A., Shintre, I., Mathews, A., Wang, J., Voss, C., Gu, C., ... & Stock, P. (2021). Opacus: User-friendly differential privacy library in PyTorch. *arXiv preprint arXiv:2109.12298*.
