"""
Objective: Dual-axis plot for the gradient clipping sweep (Utility vs MIA Attack AUC).

Generates a dual-y-axis plot:
  - Left axis: mean test accuracy with std error bars
  - Right axis: mean attack AUC with bootstrap 95% CI shaded band
  - Reference lines with full labels:
      - Unclipped baseline accuracy (50.96%)
      - Unclipped baseline attack AUC (0.8535)
      - Random guessing dotted line (AUC = 0.50)
  - x-axis: clip norm C on log scale

Usage:
    python -m src.plot_clipping
"""

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import config

RESULTS_DIR = getattr(config, "RESULTS_DIR", "experiments/cifar10/results")
DEFAULT_INPUT = os.path.join(RESULTS_DIR, "clipping_sweep_attack.json")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "clipping_sweep.png")

BASELINE_ACCURACY = 50.96
BASELINE_ATTACK_AUC = 0.8535
RANDOM_GUESS_AUC = 0.50


def load_attack_data(input_path: str) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    """Load and aggregate data from clipping_sweep_attack.json."""
    with open(input_path, "r") as f:
        data = json.load(f)

    if "summary" in data and data["summary"]:
        summary = data["summary"]
        c_vals = sorted([float(c) for c in summary.keys()])
        acc_means = [summary[str(c)]["test_accuracy_mean"] for c in c_vals]
        acc_stds = [summary[str(c)]["test_accuracy_std"] for c in c_vals]
        auc_means = [summary[str(c)]["attack_auc_mean"] for c in c_vals]
        ci_lowers = [summary[str(c)]["attack_auc_ci_lower"] for c in c_vals]
        ci_uppers = [summary[str(c)]["attack_auc_ci_upper"] for c in c_vals]
        return c_vals, acc_means, acc_stds, auc_means, ci_lowers, ci_uppers

    # Fallback if summary was not precomputed:
    results = data["results"] if isinstance(data, dict) and "results" in data else data
    grouped: dict[float, list[dict]] = {}
    for r in results:
        c = float(r["clip_norm"])
        grouped.setdefault(c, []).append(r)

    c_vals = sorted(grouped.keys())
    acc_means, acc_stds = [], []
    auc_means, ci_lowers, ci_uppers = [], [], []

    for c in c_vals:
        runs = grouped[c]
        accs = [r["test_accuracy"] for r in runs if r.get("test_accuracy") is not None]
        aucs = [r["attack_auc"] for r in runs]

        acc_means.append(float(np.mean(accs)) if accs else 0.0)
        acc_stds.append(float(np.std(accs)) if accs else 0.0)
        mean_auc = float(np.mean(aucs))
        auc_means.append(mean_auc)
        # Approximate CI if score arrays unavailable
        std_auc = float(np.std(aucs))
        ci_lowers.append(mean_auc - 1.96 * std_auc)
        ci_uppers.append(mean_auc + 1.96 * std_auc)

    return c_vals, acc_means, acc_stds, auc_means, ci_lowers, ci_uppers


def plot_clipping_sweep(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
):
    """Generate dual y-axis plot of clipping sweep results."""
    c_vals, acc_means, acc_stds, auc_means, ci_lowers, ci_uppers = load_attack_data(input_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # ── Left Axis: Test Accuracy (Utility) ──
    color_acc = "#1f77b4"
    line_acc = ax1.errorbar(
        c_vals, acc_means, yerr=acc_stds,
        fmt="o-", color=color_acc, linewidth=2.2, markersize=8,
        capsize=4, capthick=1.5,
        label="Test Accuracy (mean ± std)",
    )
    ref_base_acc = ax1.axhline(
        BASELINE_ACCURACY,
        color=color_acc, linestyle="--", linewidth=1.5, alpha=0.85,
        label=f"Unclipped Baseline Accuracy ({BASELINE_ACCURACY:.2f}%)",
    )

    # ── Right Axis: MIA Attack AUC (Privacy) ──
    color_auc = "#d62728"
    line_auc = ax2.plot(
        c_vals, auc_means,
        "s-", color=color_auc, linewidth=2.2, markersize=8,
        label="Attack AUC (mean)",
    )[0]
    band_ci = ax2.fill_between(
        c_vals, ci_lowers, ci_uppers,
        color=color_auc, alpha=0.20,
        label="Attack AUC (95% Bootstrap CI)",
    )
    ref_base_auc = ax2.axhline(
        BASELINE_ATTACK_AUC,
        color=color_auc, linestyle="--", linewidth=1.5, alpha=0.85,
        label=f"Unclipped Baseline Attack AUC ({BASELINE_ATTACK_AUC:.4f})",
    )
    ref_rand = ax2.axhline(
        RANDOM_GUESS_AUC,
        color="#7f7f7f", linestyle=":", linewidth=1.5,
        label=f"Random Guessing (AUC = {RANDOM_GUESS_AUC:.2f})",
    )

    # ── Log Scale and Tick Formatting ──
    ax1.set_xscale("log")
    ax1.set_xticks(c_vals)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax1.set_xlabel("Gradient Clip Norm C (log scale)", fontsize=12, fontweight="medium")
    ax1.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=12, fontweight="medium")
    ax2.set_ylabel("MIA Attack AUC-ROC", color=color_auc, fontsize=12, fontweight="medium")

    ax1.tick_params(axis="y", labelcolor=color_acc, labelsize=10)
    ax2.tick_params(axis="y", labelcolor=color_auc, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)

    # Adjust axis limits for clarity
    min_acc = min(acc_means) if acc_means else 30.0
    ax1.set_ylim(bottom=max(0.0, min_acc - 10.0), top=BASELINE_ACCURACY + 5.0)
    ax2.set_ylim(bottom=0.45, top=max(BASELINE_ATTACK_AUC + 0.05, 0.90))

    # ── Combined Legend (Every line has an entry) ──
    handles = [line_acc, ref_base_acc, line_auc, band_ci, ref_base_auc, ref_rand]
    labels = [h.get_label() for h in handles]
    ax1.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.03, 0.48),
        fontsize=9.5,
        framealpha=0.92,
        edgecolor="#cccccc",
    )

    ax1.grid(True, which="major", linestyle="--", alpha=0.35)
    plt.title(
        "CIFAR-10 Gradient Clipping Sweep: Utility vs. MIA Vulnerability (No Noise)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot gradient clipping sweep results")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to clipping_sweep_attack.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save clipping_sweep.png")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}. Run src.sweep_clipping first.")

    plot_clipping_sweep(args.input, args.output)


if __name__ == "__main__":
    main()
