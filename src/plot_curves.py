"""
Plot privacy–utility curves from the epsilon sweep results.

Handles:
  - "inf" string from JSON serialization (baseline)
  - Multi-seed aggregation (mean ± std error bars)
  - Log-scale ε axis

Usage:
    python -m src.plot_curves
    python -m src.plot_curves --input experiments/results/epsilon_sweep.json
"""

import argparse
import json
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    """Load sweep manifest, separating baseline from DP runs."""
    with open(path) as f:
        results = json.load(f)

    baselines = [r for r in results if r["epsilon"] == "inf"]
    dp_runs = [r for r in results if r["epsilon"] != "inf"]

    return baselines, dp_runs


def aggregate_by_noise_multiplier(runs):
    """
    Group runs by noise_multiplier, compute mean and std
    for accuracy, loss, and epsilon across seeds.
    """
    groups = defaultdict(list)
    for r in runs:
        groups[r["noise_multiplier"]].append(r)

    aggregated = []
    for nm in sorted(groups.keys()):
        group = groups[nm]
        epsilons = [r["epsilon"] for r in group]
        accs = [r["test_accuracy"] for r in group]
        losses = [r["test_loss"] for r in group]

        aggregated.append({
            "noise_multiplier": nm,
            "epsilon_mean": np.mean(epsilons),
            "epsilon_std": np.std(epsilons),
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "loss_mean": np.mean(losses),
            "loss_std": np.std(losses),
            "n_seeds": len(group),
        })

    return aggregated


def plot_privacy_utility(baselines, dp_aggregated, output_path, show=False):
    """Generate the privacy–utility tradeoff plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # ── DP curve ──
    epsilons = [d["epsilon_mean"] for d in dp_aggregated]
    accs = [d["accuracy_mean"] for d in dp_aggregated]
    acc_stds = [d["accuracy_std"] for d in dp_aggregated]

    multi_seed = dp_aggregated[0]["n_seeds"] > 1

    if multi_seed:
        ax.errorbar(
            epsilons, accs, yerr=acc_stds,
            fmt="bo-", linewidth=2, markersize=8,
            capsize=4, capthick=1.5,
            label="DP-SGD (mean ± std)",
        )
    else:
        ax.plot(
            epsilons, accs, "bo-",
            linewidth=2, markersize=8,
            label="DP-SGD",
        )

    # ── Baseline reference ──
    baseline_accs = [b["test_accuracy"] for b in baselines]
    baseline_mean = np.mean(baseline_accs)

    ax.axhline(
        y=baseline_mean, color="r", linestyle="--", linewidth=1.5,
        label=f"Baseline (no DP): {baseline_mean:.2f}%",
    )

    if len(baselines) > 1:
        baseline_std = np.std(baseline_accs)
        ax.axhspan(
            baseline_mean - baseline_std,
            baseline_mean + baseline_std,
            color="r", alpha=0.08,
        )

    # ── Annotations ──
    for d in dp_aggregated:
        label = f"nm={d['noise_multiplier']}\nε={d['epsilon_mean']:.2f}"
        if multi_seed:
            label += f"\n{d['accuracy_mean']:.1f}±{d['accuracy_std']:.1f}%"
        else:
            label += f"\n{d['accuracy_mean']:.1f}%"

        ax.annotate(
            label,
            (d["epsilon_mean"], d["accuracy_mean"]),
            textcoords="offset points",
            xytext=(0, 15), ha="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="gray", alpha=0.7),
        )

    # ── Formatting ──
    ax.set_xlabel("Privacy Budget (ε)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Privacy–Utility Tradeoff on MNIST (DP-SGD)", fontsize=15)
    ax.set_xscale("log")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Guide regions ──
    ax.axvspan(ax.get_xlim()[0], 1.0, color="green", alpha=0.03)
    ax.axvspan(1.0, 10.0, color="yellow", alpha=0.03)
    ax.axvspan(10.0, ax.get_xlim()[1], color="red", alpha=0.03)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_results_table(baselines, dp_aggregated):
    """Print a formatted summary table to stdout."""
    multi_seed = dp_aggregated[0]["n_seeds"] > 1

    print(f"\n{'─' * 75}")
    if multi_seed:
        n = dp_aggregated[0]["n_seeds"]
        print(f"  Results Summary ({n} seeds per configuration)")
        print(f"{'─' * 75}")
        print(f"  {'Noise Mult':>10}  {'ε (mean±std)':>16}  "
              f"{'Accuracy (mean±std)':>22}  {'Loss':>10}")
        print(f"{'─' * 75}")

        baseline_mean = np.mean([b["test_accuracy"] for b in baselines])
        baseline_std = np.std([b["test_accuracy"] for b in baselines])
        print(f"  {'Baseline':>10}  {'∞':>16}  "
              f"{baseline_mean:>8.2f} ± {baseline_std:<6.2f}%      "
              f"{np.mean([b['test_loss'] for b in baselines]):>10.4f}")

        for d in dp_aggregated:
            print(f"  {d['noise_multiplier']:>10.2f}  "
                  f"{d['epsilon_mean']:>8.4f} ± {d['epsilon_std']:<6.4f}  "
                  f"{d['accuracy_mean']:>8.2f} ± {d['accuracy_std']:<6.2f}%"
                  f"      {d['loss_mean']:>10.4f}")
    else:
        print(f"  Results Summary (single seed)")
        print(f"{'─' * 75}")
        print(f"  {'Noise Mult':>10}  {'ε':>10}  "
              f"{'Accuracy':>10}  {'Loss':>10}")
        print(f"{'─' * 75}")

        print(f"  {'Baseline':>10}  {'∞':>10}  "
              f"{baselines[0]['test_accuracy']:>9.2f}%  "
              f"{baselines[0]['test_loss']:>10.4f}")

        for d in dp_aggregated:
            print(f"  {d['noise_multiplier']:>10.2f}  "
                  f"{d['epsilon_mean']:>10.4f}  "
                  f"{d['accuracy_mean']:>9.2f}%  "
                  f"{d['loss_mean']:>10.4f}")

    print(f"{'─' * 75}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot privacy–utility curves from sweep results"
    )
    parser.add_argument(
        "--input", type=str,
        default="experiments/results/epsilon_sweep.json",
        help="Path to epsilon_sweep.json manifest",
    )
    parser.add_argument(
        "--output", type=str,
        default="experiments/results/privacy_utility_curve.png",
        help="Output path for the plot image",
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Display the plot window interactively",
    )
    args = parser.parse_args()

    baselines, dp_runs = load_results(args.input)

    if not dp_runs:
        print("No DP-SGD results found in manifest. Run sweep first.")
        return

    dp_aggregated = aggregate_by_noise_multiplier(dp_runs)

    print_results_table(baselines, dp_aggregated)
    plot_privacy_utility(baselines, dp_aggregated, args.output, show=args.show)


if __name__ == "__main__":
    main()
