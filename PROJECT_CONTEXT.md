# PROJECT_CONTEXT.md

# Empirical Evaluation of Differential Privacy in Machine Learning under Membership Inference Attacks

## Purpose of This Context

This document provides persistent **project-level context** for AI agents and collaborators working on this project.

It is **not** an implementation guide, task list, roadmap, or set of instructions for solving the project. Specific tasks, steps, experiments, and implementation decisions will be provided separately in each working session.

The purpose of this document is to ensure that every new agent understands:

- what the project fundamentally studies,
- why the system exists,
- the conceptual relationship between Differential Privacy and Membership Inference Attacks,
- the architecture and experimental context already established,
- the terminology and experimental baseline of the project,
- what has actually been verified versus what remains outside the current evidence.

---

# 1. Project Identity

**Title:**  
**Empirical Evaluation of Differential Privacy in Machine Learning under Membership Inference Attacks**

This is a machine learning privacy research project focused on empirically studying the interaction between:

- **Differential Privacy (DP)**
- **Model utility**
- **Training-data memorization**
- **Membership Inference Attacks (MIA)**

The project uses a controlled machine learning environment to study how privacy-preserving training changes both model performance and the observable relationship between training members and non-members.

The current implementation environment is based on:

- PyTorch
- Opacus
- MNIST
- A CNN model referred to as `SampleCNN`

---

# 2. The Fundamental Research Problem

Machine learning models are trained using data that may contain sensitive individual records.

During training, a model may develop different behavior for samples that were part of its training set compared with samples that were never used during training. Such differences can potentially be exploited by an adversary.

A **Membership Inference Attack** asks a question of the following form:

> Given a trained model and a particular data sample, can an attacker infer whether that sample was part of the model's training data?

This creates a privacy concern because successful membership inference may reveal information about participation in a dataset.

The project studies Differential Privacy as a defense mechanism against excessive influence from individual training records.

At the deepest level, the project is concerned with the following tension:

```text
Individual data influence
          ↓
Potential memorization / distinguishability
          ↓
Membership inference risk

Differential Privacy attempts to limit
the influence of individual records
          ↓
But this can affect optimization
          ↓
And therefore affect model utility
```

The research problem is therefore not simply:

> "Does Differential Privacy work?"

Instead, the project investigates the empirical relationship between:

```text
Privacy strength
       ↕
Model utility
       ↕
Empirical membership inference behavior
```

---

# 3. Core Conceptual Model

The project can be understood through three connected dimensions.

## 3.1 Privacy

Privacy is represented through Differential Privacy guarantees produced by DP-SGD training.

The project uses privacy parameters such as:

- ε (epsilon)
- δ (delta)

The reported epsilon represents privacy expenditure under the selected accounting framework and configuration.

In the project's conceptual framing, epsilon is part of the **formal privacy guarantee**.

---

## 3.2 Utility

Utility represents how useful the trained model remains for its intended machine learning task.

For the current MNIST classification environment, utility is primarily represented by measurements such as:

- Test accuracy
- Test loss

A privacy-preserving model may experience lower utility because the private optimization process constrains and perturbs training updates.

---

## 3.3 Membership Inference Resistance

Membership Inference resistance is an **empirical security property** evaluated through attack experiments.

It concerns how successfully an attack can distinguish:

```text
Member
= sample used during model training

Non-member
= sample not used during model training
```

Potential attack metrics include:

- Attack accuracy
- AUC

This dimension must remain conceptually separate from the formal Differential Privacy guarantee.

A DP guarantee and an empirically measured attack metric are related to the same privacy problem but are not identical measurements.

---

# 4. The System at a High Level

The project environment contains a machine learning dataset, a model architecture, and parallel training conditions.

Conceptually:

```text
                         MNIST DATA
                             |
                             v
                       Preprocessing
                             |
                             v
                         SampleCNN
                             |
                  +----------+----------+
                  |                     |
                  v                     v
             Standard SGD             DP-SGD
                  |                     |
                  v                     v
           Non-private model      Private model
                  |                     |
                  +----------+----------+
                             |
                             v
                       Model evaluation
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
              Utility              Privacy context
                 |
                 v
          Comparative analysis
```

The broader research context extends this comparison toward multiple privacy configurations and empirical Membership Inference Attack evaluation.

The important system-level idea is that the project is based on **controlled comparison** rather than treating DP training as an isolated model.

---

# 5. Differential Privacy in This Project

The privacy-preserving training mechanism used by the project is **Differentially Private Stochastic Gradient Descent (DP-SGD)** through Opacus.

The mechanism involves three central ideas.

## Per-Sample Gradients

Training behavior is considered at the level of individual samples rather than only as a single aggregate batch update.

This is necessary because the privacy mechanism needs to control the influence of individual records.

---

## Gradient Clipping

Individual sample gradients are bounded using a clipping norm.

The currently verified experiment uses:

```text
max_grad_norm = 1.0
```

The conceptual purpose of clipping is to limit the maximum contribution of any individual training sample to a model update.

---

## Noise Addition

Noise is introduced into the optimization process.

The currently verified experiment uses:

```text
noise_multiplier = 1.1
```

Noise addition is part of the mechanism used to reduce the extent to which the final model depends on any individual record.

---

## Privacy Accounting

The project uses Opacus privacy accounting with an RDP accountant.

Privacy expenditure is reported in terms of epsilon for a selected delta.

The verified experiment reports:

```text
ε = 0.30
δ = 1e-5
```

with the reported privacy guarantee:

```text
(ε = 0.30, δ = 1e-5)-DP
```

---

# 6. Important Conceptual Boundary: Formal DP vs Empirical MIA

All agents working on this project must preserve the distinction below.

## Formal Differential Privacy

Differential Privacy provides a mathematical privacy framework associated with the training mechanism and its parameters.

## Membership Inference Evaluation

Membership Inference performance is measured empirically by attacking a trained model and evaluating attack outcomes.

Therefore:

```text
Formal DP guarantee
        ≠
Empirical MIA metric
```

The project studies the relationship between these two dimensions.

The existence of a DP guarantee should not automatically be described as proof that every possible empirical attack will have a particular measured accuracy.

Likewise, an empirical attack result should not be presented as a replacement for the formal DP guarantee.

---

# 7. Dataset Context

The current experimental dataset is **MNIST**.

MNIST properties:

- 60,000 training samples
- 10,000 test samples
- 28 × 28 grayscale images
- 10 digit classes

Current normalization parameters:

```text
Mean = 0.1307
Standard deviation = 0.3081
```

MNIST is currently the controlled environment in which the DP-SGD pipeline and baseline comparison have been established.

---

# 8. Model Context

The project uses a CNN referred to as **SampleCNN**.

The model sequence is:

```text
Conv2D
  ↓
ReLU
  ↓
MaxPool2D
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool2D
  ↓
Linear
  ↓
ReLU
  ↓
Linear
```

The architecture does not use BatchNorm layers.

The project context identifies this as supporting straightforward per-sample gradient computation and Opacus compatibility.

The model passes Opacus `ModuleValidator` checks in the verified pipeline.

---

# 9. Current Experimental Environment

The verified experiment uses the following training configuration:

```text
Epochs: 5
Batch Size: 64
Learning Rate: 0.05
Optimizer: SGD
Momentum: 0.0
Random Seed: 42
```

The verified DP configuration is:

```text
Max Grad Norm: 1.0
Noise Multiplier: 1.1
Target Delta: 1e-5
Final Epsilon: 0.30
```

These values describe the currently established experiment. They should not automatically be interpreted as permanent constraints for every future experiment unless a specific task explicitly requires them.

---

# 10. Software and Technical Environment

The documented software environment is:

```text
Python 3.9.6
PyTorch 2.8.0
torchvision 0.23.0
Opacus 1.6.0
pytest 8.4.2
NumPy 2.0.2
```

The main implementation components documented in the project are:

```text
src/train_baseline.py
src/train_dp.py
tests/test_pipeline.py
```

Conceptually:

- `train_baseline.py` represents the non-private SGD training pipeline.
- `train_dp.py` represents the Opacus DP-SGD training pipeline.
- `test_pipeline.py` represents automated verification of the pipeline.

Relevant Opacus components include:

```text
GradSampleModule
DPOptimizer
DPDataLoader
```

These are part of the established technical context surrounding per-sample gradients, private optimization, and data loading/sampling behavior.

---

# 11. Established Experimental Baseline

The project has one currently verified primary comparison.

## Standard SGD Baseline

```text
Test Accuracy: 98.75%
Test Loss: 0.0354
```

This is the non-private reference model.

---

## DP-SGD Model

```text
Test Accuracy: 90.27%
Test Loss: 0.5328

Privacy:
ε = 0.30
δ = 1e-5
```

---

## Observed Utility Difference

The measured difference in test accuracy is:

```text
98.75% - 90.27%
= 8.48 percentage points
```

This result is the current empirical evidence of a privacy–utility trade-off for this specific model and experimental configuration.

It should not be generalized beyond the context of the measured experiment without additional evidence.

---

# 12. Epoch-Level Context of the Verified Experiment

The currently documented progression is:

| Epoch | Baseline Accuracy | DP Accuracy | ε |
|---|---:|---:|---:|
| 1 | 97.56% | 84.87% | 0.14 |
| 2 | 98.09% | 87.59% | 0.19 |
| 3 | 98.85% | 88.76% | 0.23 |
| 4 | 98.65% | 89.68% | 0.27 |
| 5 | 98.75% | 90.27% | 0.30 |

This illustrates two aspects of the current experiment:

1. Model utility changes as training progresses.
2. Privacy expenditure, represented by epsilon, also changes over training.

The documented project interpretation is that clipping and noise introduce optimization distortion under the DP constraints.

---

# 13. Research Scope

The project is broader than simply building a private MNIST classifier.

The intended research scope is the empirical relationship between:

```text
Differential Privacy configuration
              ↓
      Model optimization behavior
              ↓
         Model utility
              ↓
Observable member/non-member behavior
              ↓
Membership Inference Attack outcomes
```

The project therefore combines:

- formal privacy parameters,
- standard machine learning evaluation,
- empirical security evaluation.

The final research identity is the analysis of these dimensions together.

---

# 14. Current Evidence Boundary

The following are **implemented and verified in the currently documented project state**:

- MNIST-based pipeline
- Standard SGD baseline
- DP-SGD pipeline using PyTorch and Opacus
- Opacus compatibility validation
- Privacy accounting
- Baseline versus DP-SGD utility comparison
- Verified result at ε = 0.30 and δ = 1e-5
- Automated pipeline testing
- Five documented automated unit tests

The following are part of the project's broader research scope but are **not established as completed experimental results in the current context**:

- Comprehensive training across multiple privacy budgets
- Complete privacy–utility curves across multiple epsilon configurations
- Membership Inference Attack implementation and full empirical evaluation
- Final privacy–utility–MIA correlation analysis

Agents must not silently convert planned research scope into completed work.

---

# 15. How Results Should Be Interpreted

This project is empirical.

Therefore, agents should preserve the distinction between:

```text
Observed result
```

and:

```text
General theoretical claim
```

For example, the currently observed 8.48 percentage-point accuracy difference belongs to the documented experiment and configuration.

It should not automatically be treated as a universal cost of Differential Privacy.

Similarly, future MIA results must be interpreted as results of the specific attack methodology, dataset partitioning, model, and experimental configuration used.

The project values controlled comparison and reproducible evidence over unsupported conclusions.

---

# 16. The Central Trade-Off

The deepest conceptual theme of the project is the trade-off between individual privacy and model utility.

Conceptually:

```text
Stronger privacy constraints
          |
          v
More restricted individual influence
          |
          v
Potentially greater optimization distortion
          |
          v
Potential utility reduction
```

At the same time, the project is interested in whether changes in privacy configuration correspond to changes in empirically measured Membership Inference vulnerability.

Thus, the project's conceptual space can be represented as:

```text
                  PRIVACY
                     /                    /                     /                      /                       /                        /                    UTILITY -------- SECURITY
                     (MIA)
```

The goal is not to assume a universal relationship among these dimensions, but to empirically characterize them within the project's experimental environment.

---

# 17. Persistent Context for Future Agents

When starting a new session, an agent should understand the following without needing the original presentation:

> This project studies Differential Privacy in machine learning through an empirical comparison of non-private and DP-SGD training, with the broader objective of understanding how privacy configuration relates to model utility and Membership Inference Attack behavior.

> The current technical environment uses MNIST, PyTorch, Opacus, and a SampleCNN architecture.

> A non-private SGD baseline and a DP-SGD pipeline have already been established and verified.

> The documented baseline achieved 98.75% test accuracy, while the documented DP-SGD experiment achieved 90.27% test accuracy at ε = 0.30 and δ = 1e-5, representing an 8.48 percentage-point difference for that experiment.

> The project is ultimately concerned with three connected dimensions: formal privacy guarantees, model utility, and empirical membership inference resistance.

> Formal DP guarantees and empirical MIA performance must remain conceptually distinct.

> The current documented context should not be interpreted as evidence that comprehensive multi-epsilon experiments or full MIA evaluations have already been completed.

---

# 18. Agent Operating Context

This document provides **background knowledge**, not session-specific instructions.

Future agents may receive separate prompts containing:

- implementation tasks,
- experiment specifications,
- debugging requests,
- research questions,
- analysis requirements,
- code changes,
- or explicit methodological decisions.

Those instructions should be interpreted within the project context defined here.

If a session-specific instruction conflicts with the currently documented experimental state, the agent should recognize the distinction between:

- the established historical baseline,
- the current task being requested,
- and newly generated experimental evidence.

The project context should evolve only when new verified work or results are explicitly incorporated into it.

---

# 19. One-Paragraph Project Definition

**Empirical Evaluation of Differential Privacy in Machine Learning under Membership Inference Attacks** is a privacy-preserving machine learning research project that investigates the relationship between Differential Privacy, model utility, and empirical vulnerability to Membership Inference Attacks. The project currently uses a PyTorch and Opacus implementation of DP-SGD on MNIST with a SampleCNN architecture, alongside a non-private SGD baseline. The established experiment demonstrates a measurable utility difference between standard and private training, while the broader research context examines how varying privacy conditions relate to both predictive performance and membership inference behavior. The project treats formal Differential Privacy guarantees and empirical attack resistance as distinct but connected dimensions that must be evaluated and interpreted carefully.

---

# 20. Context Integrity Rules

For consistency across sessions:

- Do not state that planned experiments are completed unless new evidence is provided.
- Do not confuse epsilon with an empirical MIA accuracy or AUC.
- Do not confuse a formal DP guarantee with absolute immunity from attacks.
- Do not generalize one experiment's measured utility difference as a universal property of DP.
- Preserve the distinction between the non-private baseline and private training conditions.
- Treat reproducibility and controlled comparisons as important research principles.
- Use this document as project background; follow separate session instructions for actual implementation or research tasks.

---

**End of persistent project context.**
