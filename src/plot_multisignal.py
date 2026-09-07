"""
Plotting Script for Multi-Signal Threshold Membership Inference Attacks.

Generates three publication-quality figures:
  1. Attack AUC vs Epsilon (linear/log scale, 3 signal lines, bootstrap CI bands,
     AUC=0.50 reference line, non-private baseline references, all in legend)
  2. Attack AUC vs Clip Norm C (same treatment)
  3. ROC Curves at 3 Representative Configurations (Baseline, Clip-only C=1.0,
     Full DP-SGD sigma=1.1), linear panel + log-log panel for low-FPR regime,
     all 3 signals overlaid.

Outputs:
  experiments/cifar10/results/multisignal_auc_vs_epsilon.png
  experiments/cifar10/results/multisignal_auc_vs_clip_norm.png
  experiments/cifar10/results/multisignal_roc_curves.png
"""

from __future__ import annotations

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
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from src.threshold_attack import roc_curve_np, auc_np, _load_config

# Styling constants
COLOR_LOSS = "#1f77b4"       # Blue
COLOR_CONF = "#ff7f0e"       # Orange
COLOR_MENTR = "#2ca02c"      # Green
COLOR_RANDOM = "#7f7f7f"     # Grey
COLOR_BASELINE = "#d62728"   # Red

SIGNAL_COLORS = {
    "loss": COLOR_LOSS,
    "confidence": COLOR_CONF,
    "mentr": COLOR_MENTR,
}

SIGNAL_LABELS = {
    "loss": "Loss (-CE)",
    "confidence": "Confidence (p_y)",
    "mentr": "Modified Entropy (-Mentr)",
}

SIGNAL_MARKERS = {
    "loss": "o",
    "confidence": "s",
    "mentr": "^",
}


def load_multisignal_data(json_path: Path) -> dict:
    """Load multisignal attack JSON results."""
    assert json_path.exists(), f"Results file not found: {json_path}"
    return json.loads(json_path.read_text())


def plot_auc_vs_epsilon(data: dict, out_path: Path):
    """
    Plot Attack AUC vs Epsilon.
    One line per signal, bootstrap CI bands, AUC=0.50 line, and baseline reference.
    """
    aggregates = data["aggregates"]

    # Filter epsilon sweep
    eps_entries = []
    baseline_entry = aggregates.get("epsilon_baseline")

    for key, val in aggregates.items():
        if val["sweep"] == "epsilon" and not val["config_key"] == "baseline":
            eps_entries.append(val)

    # Sort by epsilon ascending
    eps_entries.sort(key=lambda x: x["epsilon"])

    eps_vals = np.array([x["epsilon"] for x in eps_entries])

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Reference line for random guessing AUC = 0.50
    ax.axhline(0.50, color=COLOR_RANDOM, linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.50)")

    # Baseline references
    if baseline_entry:
        b_loss = baseline_entry["signals"]["loss"]["mean_auc"]
        b_conf = baseline_entry["signals"]["confidence"]["mean_auc"]
        b_mentr = baseline_entry["signals"]["mentr"]["mean_auc"]

        ax.axhline(b_loss, color=COLOR_LOSS, linestyle=":", linewidth=1.4, alpha=0.8,
                   label=f"Baseline Loss (AUC = {b_loss:.4f})")
        ax.axhline(b_conf, color=COLOR_CONF, linestyle=":", linewidth=1.4, alpha=0.8,
                   label=f"Baseline Confidence (AUC = {b_conf:.4f})")
        ax.axhline(b_mentr, color=COLOR_MENTR, linestyle=":", linewidth=1.4, alpha=0.8,
                   label=f"Baseline Mentr (AUC = {b_mentr:.4f})")

    # Signal curves
    for sig in ["loss", "confidence", "mentr"]:
        mean_aucs = np.array([x["signals"][sig]["mean_auc"] for x in eps_entries])
        ci_lowers = np.array([x["signals"][sig]["ci_lower"] for x in eps_entries])
        ci_uppers = np.array([x["signals"][sig]["ci_upper"] for x in eps_entries])

        color = SIGNAL_COLORS[sig]
        marker = SIGNAL_MARKERS[sig]
        label = f"{SIGNAL_LABELS[sig]} (DP-SGD)"

        ax.plot(eps_vals, mean_aucs, marker=marker, markersize=6, linewidth=2.0, color=color, label=label)
        ax.fill_between(eps_vals, ci_lowers, ci_uppers, color=color, alpha=0.18)

    ax.set_xscale("log")
    ax.set_xlabel(r"Differential Privacy Budget $\epsilon$ ($\delta=10^{-5}$, log scale)", fontsize=12, fontweight="medium")
    ax.set_ylabel("MIA Attack AUC", fontsize=12, fontweight="medium")
    ax.set_title("CIFAR-10: Multi-Signal Membership Inference Attack AUC vs. Epsilon\n"
                 r"(Subsampled $N=5{,}000$, 50 epochs, $C=1.0$, seeds 42, 43, 44 with 95% Bootstrap CIs)",
                 fontsize=13, fontweight="bold", pad=12)

    ax.set_ylim(0.46, 0.90)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_auc_vs_clip_norm(data: dict, out_path: Path):
    """
    Plot Attack AUC vs Gradient Clipping Norm C (Noise sigma = 0.0).
    One line per signal, bootstrap CI bands, AUC=0.50 line, and baseline reference.
    """
    aggregates = data["aggregates"]

    # Filter clipping sweep
    clip_entries = []
    baseline_entry = aggregates.get("epsilon_baseline")

    for key, val in aggregates.items():
        if val["sweep"] == "clipping":
            clip_entries.append(val)

    # Sort by clip norm ascending
    clip_entries.sort(key=lambda x: x["clip_norm"])

    clip_vals = np.array([x["clip_norm"] for x in clip_entries])

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Reference line for random guessing AUC = 0.50
    ax.axhline(0.50, color=COLOR_RANDOM, linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.50)")

    # Baseline references (unclipped model)
    if baseline_entry:
        b_loss = baseline_entry["signals"]["loss"]["mean_auc"]
        b_conf = baseline_entry["signals"]["confidence"]["mean_auc"]
        b_mentr = baseline_entry["signals"]["mentr"]["mean_auc"]

        ax.axhline(b_loss, color=COLOR_LOSS, linestyle=":", linewidth=1.4, alpha=0.8,
                   label=f"Unclipped Baseline Loss (AUC = {b_loss:.4f})")
        ax.axhline(b_conf, color=COLOR_CONF, linestyle=":", linewidth=1.4, alpha=0.8,
                   label=f"Unclipped Baseline Confidence (AUC = {b_conf:.4f})")
        ax.axhline(b_mentr, color=COLOR_MENTR, linestyle=":", linewidth=1.4, alpha=0.8,
                   label=f"Unclipped Baseline Mentr (AUC = {b_mentr:.4f})")

    # Signal curves
    for sig in ["loss", "confidence", "mentr"]:
        mean_aucs = np.array([x["signals"][sig]["mean_auc"] for x in clip_entries])
        ci_lowers = np.array([x["signals"][sig]["ci_lower"] for x in clip_entries])
        ci_uppers = np.array([x["signals"][sig]["ci_upper"] for x in clip_entries])

        color = SIGNAL_COLORS[sig]
        marker = SIGNAL_MARKERS[sig]
        label = f"{SIGNAL_LABELS[sig]} (Clip Only, $\sigma=0.0$)"

        ax.plot(clip_vals, mean_aucs, marker=marker, markersize=6, linewidth=2.0, color=color, label=label)
        ax.fill_between(clip_vals, ci_lowers, ci_uppers, color=color, alpha=0.18)

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 5.0, 10.0, 50.0])
    ax.set_xticklabels(["0.5", "1.0", "5.0", "10.0", "50.0"])

    ax.set_xlabel(r"Gradient Clipping Norm $C$ ($\sigma=0.0$, log scale)", fontsize=12, fontweight="medium")
    ax.set_ylabel("MIA Attack AUC", fontsize=12, fontweight="medium")
    ax.set_title("CIFAR-10: Multi-Signal Membership Inference Attack AUC vs. Clip Norm $C$\n"
                 r"(Pure Gradient Clipping at $\sigma=0.0$, seeds 42, 43, 44 with 95% Bootstrap CIs)",
                 fontsize=13, fontweight="bold", pad=12)

    ax.set_ylim(0.46, 0.90)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_roc_curves_representative(scores_dir: Path, out_path: Path):
    """
    Plot ROC curves at three representative configurations:
      1. Non-private baseline
      2. Clip-only C=1.0 (sigma=0.0)
      3. Full DP-SGD sigma=1.1 (C=1.0, epsilon ~ 4.09)

    Overlays all three signals (loss, confidence, mentr) on both linear and log-log panels.
    """
    configs_to_plot = [
        ("Non-Private Baseline (No DP)", ["baseline_seed42", "baseline_seed43", "baseline_seed44"]),
        ("Clip-Only (C=1.0, $\sigma=0.0$)", ["clip_1.00_nm_0.00_seed42", "clip_1.00_nm_0.00_seed43", "clip_1.00_nm_0.00_seed44"]),
        ("Full DP-SGD ($\sigma=1.1$, $C=1.0$, $\epsilon=4.09$)", ["dp_nm_1.10_seed42", "dp_nm_1.10_seed43", "dp_nm_1.10_seed44"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)

    for col_idx, (col_title, stems) in enumerate(configs_to_plot):
        # Pool scores across the 3 seeds
        pooled = {sig: {"member": [], "nonmember": []} for sig in ["loss", "confidence", "mentr"]}

        for stem in stems:
            npz_path = scores_dir / f"{stem}.npz"
            if not npz_path.exists():
                print(f"[warn] missing score file {npz_path}")
                continue
            data = np.load(npz_path)
            for sig in ["loss", "confidence", "mentr"]:
                pooled[sig]["member"].append(data[f"member_{sig}"])
                pooled[sig]["nonmember"].append(data[f"nonmember_{sig}"])

        ax_lin = axes[0, col_idx]
        ax_log = axes[1, col_idx]

        # Diagonal random guessing lines
        ax_lin.plot([0, 1], [0, 1], color=COLOR_RANDOM, linestyle="--", linewidth=1.2, label="Random Guess")
        ax_log.plot([1e-4, 1.0], [1e-4, 1.0], color=COLOR_RANDOM, linestyle="--", linewidth=1.2, label="Random Guess")

        for sig in ["loss", "confidence", "mentr"]:
            mem = np.concatenate(pooled[sig]["member"])
            non = np.concatenate(pooled[sig]["nonmember"])

            y_true = np.r_[np.ones_like(mem), np.zeros_like(non)]
            scores = np.r_[mem, non]

            fpr, tpr, _ = roc_curve_np(y_true, scores)
            auc = auc_np(fpr, tpr)

            color = SIGNAL_COLORS[sig]
            label = f"{SIGNAL_LABELS[sig]} (AUC={auc:.4f})"

            # Top row: Linear ROC
            ax_lin.plot(fpr, tpr, color=color, linewidth=2.0, label=label)

            # Bottom row: Log-log ROC in low-FPR regime
            fpr_clipped = np.clip(fpr, 1e-5, 1.0)
            tpr_clipped = np.clip(tpr, 1e-5, 1.0)
            ax_log.plot(fpr_clipped, tpr_clipped, color=color, linewidth=2.0, label=label)

        # Configure Linear panel
        ax_lin.set_xlim(-0.02, 1.02)
        ax_lin.set_ylim(-0.02, 1.02)
        ax_lin.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax_lin.set_ylabel("True Positive Rate (TPR)", fontsize=11)
        ax_lin.set_title(f"{col_title}\n[Linear ROC]", fontsize=12, fontweight="bold")
        ax_lin.grid(True, linestyle="--", alpha=0.35)
        ax_lin.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

        # Configure Log-Log panel
        ax_log.set_xscale("log")
        ax_log.set_yscale("log")
        ax_log.set_xlim(1e-4, 1.0)
        ax_log.set_ylim(1e-4, 1.0)
        ax_log.set_xlabel("False Positive Rate (FPR, log scale)", fontsize=11)
        ax_log.set_ylabel("True Positive Rate (TPR, log scale)", fontsize=11)
        ax_log.set_title(f"{col_title}\n[Log-Log ROC: Low-FPR Regime]", fontsize=12, fontweight="bold")
        ax_log.grid(True, which="both", linestyle="--", alpha=0.35)
        ax_log.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    plt.suptitle("CIFAR-10: Multi-Signal Membership Inference ROC Curves Across Representative Models\n"
                 r"(Pooled $N=15{,}000$ members & $15{,}000$ non-members across seeds 42, 43, 44)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-signal attack plots")
    parser.add_argument("--json", default=str(REPO_ROOT / "experiments/cifar10/results/multisignal_attack.json"))
    parser.add_argument("--scores-dir", default=str(REPO_ROOT / "experiments/cifar10/results/mia_scores"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "experiments/cifar10/results"))
    args = parser.parse_args()

    json_path = Path(args.json)
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise SystemExit(f"JSON results file not found: {json_path}. Run src.evaluate_multisignal first.")

    data = load_multisignal_data(json_path)

    plot_auc_vs_epsilon(data, out_dir / "multisignal_auc_vs_epsilon.png")
    plot_auc_vs_clip_norm(data, out_dir / "multisignal_auc_vs_clip_norm.png")
    plot_roc_curves_representative(scores_dir, out_dir / "multisignal_roc_curves.png")

    print("\nAll multi-signal plots successfully generated!")


if __name__ == "__main__":
    main()
