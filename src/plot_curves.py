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
import os
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    """
    Load sweep manifest, supporting both multi-seed aggregated format
    (keyed by noise multiplier) and single-seed flat list format.
    Returns (baselines, dp_aggregated).
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        data = data["results"]

    if isinstance(data, dict):
        # Multi-seed dictionary keyed by noise multiplier
        baselines = []
        dp_aggregated = []
        for nm_key, entry in sorted(data.items(), key=lambda x: float(x[0])):
            nm = float(entry.get("noise_multiplier", nm_key))
            eps = entry.get("epsilon")
            if nm == 0.0 or eps == "inf" or eps is None:
                if "runs" in entry and entry["runs"]:
                    baselines.extend(entry["runs"])
                else:
                    baselines.append({
                        "test_accuracy": entry.get("test_accuracy_mean", 0.0),
                        "test_loss": entry.get("test_loss_mean", 0.0),
                        "epsilon": "inf",
                        "noise_multiplier": 0.0,
                    })
            else:
                eps_val = float(eps)
                dp_aggregated.append({
                    "noise_multiplier": nm,
                    "epsilon_mean": eps_val,
                    "epsilon_std": 0.0,
                    "accuracy_mean": float(entry["test_accuracy_mean"]),
                    "accuracy_std": float(entry["test_accuracy_std"]),
                    "loss_mean": float(entry["test_loss_mean"]),
                    "loss_std": float(entry["test_loss_std"]),
                    "n_seeds": int(entry.get("n_seeds", len(entry.get("runs", [])) or 1)),
                })
        return baselines, dp_aggregated

    # Flat list (legacy single-seed or flat format)
    baselines = [r for r in data if r.get("epsilon") == "inf" or r.get("noise_multiplier") == 0.0]
    dp_runs = [r for r in data if r.get("epsilon") != "inf" and r.get("noise_multiplier") != 0.0]
    return baselines, aggregate_by_noise_multiplier(dp_runs)


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
        epsilons = [float(r["epsilon"]) for r in group if r.get("epsilon") is not None]
        accs = [r["test_accuracy"] for r in group]
        losses = [r["test_loss"] for r in group]

        aggregated.append({
            "noise_multiplier": nm,
            "epsilon_mean": float(np.mean(epsilons)) if epsilons else 0.0,
            "epsilon_std": float(np.std(epsilons)) if epsilons else 0.0,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "n_seeds": len(group),
        })

    return aggregated


def plot_privacy_utility(baselines, dp_aggregated, output_path, dataset_name=None, show=False):
    """Generate the privacy–utility tradeoff plot."""
    if dataset_name is None:
        if baselines and "dataset" in baselines[0] and baselines[0].get("dataset"):
            dataset_name = baselines[0]["dataset"]
        else:
            try:
                import config
                dataset_name = getattr(config, "DATASET", "cifar10")
            except ImportError:
                dataset_name = "cifar10"

    ds_str = str(dataset_name).lower()
    if "cifar" in ds_str:
        ds_display = "CIFAR-10"
    elif "mnist" in ds_str:
        ds_display = "MNIST"
    else:
        ds_display = str(dataset_name).upper()

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── DP curve with std error bars on every point ──
    epsilons = [d["epsilon_mean"] for d in dp_aggregated]
    accs = [d["accuracy_mean"] for d in dp_aggregated]
    acc_stds = [d["accuracy_std"] for d in dp_aggregated]

    multi_seed = any(d["n_seeds"] > 1 for d in dp_aggregated) or any(s > 0 for s in acc_stds)

    ax.errorbar(
        epsilons, accs, yerr=acc_stds,
        fmt="bo-", linewidth=2, markersize=8,
        capsize=4, capthick=1.5,
        label="DP-SGD (mean ± std)" if multi_seed else "DP-SGD",
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

    # ── Annotations (repositioned nm=1.5, 1.1, 0.9 to avoid curve overlap) ──
    for d in dp_aggregated:
        nm = d["noise_multiplier"]
        label = f"nm={nm}\nε={d['epsilon_mean']:.2f}"
        if multi_seed and d["accuracy_std"] > 0:
            label += f"\n{d['accuracy_mean']:.1f}±{d['accuracy_std']:.1f}%"
        else:
            label += f"\n{d['accuracy_mean']:.1f}%"

        # Explicitly reposition nm=1.5, 1.1, 0.9 to avoid overlapping each other and the curve
        if abs(nm - 1.5) < 1e-3:
            offset = (-30, 22)
        elif abs(nm - 1.1) < 1e-3:
            offset = (0, -42)
        elif abs(nm - 0.9) < 1e-3:
            offset = (30, 22)
        else:
            offset = (0, 15)

        ax.annotate(
            label,
            (d["epsilon_mean"], d["accuracy_mean"]),
            textcoords="offset points",
            xytext=offset, ha="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="gray", alpha=0.7),
        )

    # ── Formatting ──
    ax.set_xlabel("Privacy Budget (ε)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title(f"Privacy–Utility Tradeoff on {ds_display} (DP-SGD)", fontsize=15)
    ax.set_xscale("log")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

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
    try:
        import config
        default_results_dir = getattr(config, "RESULTS_DIR", "experiments/results")
    except ImportError:
        default_results_dir = "experiments/results"

    multiseed_path = f"{default_results_dir}/epsilon_sweep_multiseed.json"
    if os.path.exists(multiseed_path):
        default_input = multiseed_path
    elif os.path.exists(f"{default_results_dir}/epsilon_sweep.json"):
        default_input = f"{default_results_dir}/epsilon_sweep.json"
    elif os.path.exists("experiments/results/epsilon_sweep.json"):
        default_input = "experiments/results/epsilon_sweep.json"
    else:
        default_input = multiseed_path

    default_output = f"{default_results_dir}/privacy_utility_curve.png"

    parser = argparse.ArgumentParser(
        description="Plot privacy–utility curves from sweep results"
    )
    parser.add_argument(
        "--input", type=str,
        default=default_input,
        help="Path to epsilon_sweep_multiseed.json or epsilon_sweep.json manifest",
    )
    parser.add_argument(
        "--output", type=str,
        default=default_output,
        help="Output path for the plot image",
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Display the plot window interactively",
    )
    args = parser.parse_args()

    baselines, dp_aggregated = load_results(args.input)

    if not dp_aggregated:
        print("No DP-SGD results found in manifest. Run sweep first.")
        return

    print_results_table(baselines, dp_aggregated)
    plot_privacy_utility(baselines, dp_aggregated, args.output, show=args.show)


if __name__ == "__main__":
    main()
