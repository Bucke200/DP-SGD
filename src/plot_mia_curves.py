"""
Objective 3 / Phase A — plots for the threshold membership inference attack.

Reads `experiments/results/threshold_attack.json` (metrics) and
`experiments/results/mia_scores/*.npz` (per-sample scores) and writes:

  mia_roc_curves.png       per-epsilon ROC, linear + log-log panels
  mia_auc_vs_epsilon.png   attack AUC against epsilon, with the 0.5 floor
  privacy_utility_attack.png   the three-way tradeoff (Phase C, objective 4)

Headless-safe: uses the Agg backend and never calls plt.show().

Reading the results
-------------------
Two things decide whether this is a publishable result:

1. The baseline must be attackable. MNIST + a small CNN generalizes almost
   perfectly, so the train/test gap is often under 1% and the loss-threshold
   attack sits at AUC ~0.50 even with no privacy at all. If that happens, the
   plot shows a flat line and there is no defense to demonstrate. Fixes, in
   order of preference: train the target models on a small subset (2k-5k MNIST
   samples) so they memorize, drop augmentation and train longer, or move to
   CIFAR-10. Report the generalization gap column next to AUC either way — it
   is the honest explanation for why the attack succeeds or fails.

2. AUC alone hides the interesting behaviour. An attack at AUC 0.51 can still
   identify a handful of samples with near-certainty, which is exactly what a
   privacy guarantee is supposed to prevent. That is why the log-log ROC panel
   and the TPR@0.1%/1% FPR columns exist; report them alongside AUC.

Usage
-----
    python -m src.plot_mia_curves
    python -m src.plot_mia_curves --signal mentr
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def roc_from_scores(member: np.ndarray, nonmember: np.ndarray):
    y = np.r_[np.ones_like(member), np.zeros_like(nonmember)]
    s = np.r_[member, nonmember]
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    idx = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tpr = np.r_[0.0, tps[idx] / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps[idx] / max(fps[-1], 1)]
    return fpr, tpr


def load(args):
    data = json.loads(Path(args.results).read_text())
    rows = data["results"]
    for row in rows:
        stem = Path(row["checkpoint"]).stem
        npz_path = Path(args.scores_dir) / f"{stem}.npz"
        row["_scores"] = np.load(npz_path) if npz_path.exists() else None
    # baseline (epsilon=None) first, then increasing privacy budget
    rows.sort(key=lambda r: (r["epsilon"] is not None, r["epsilon"] or 0.0))
    return data, rows


def plot_roc(rows, signal, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(rows), 2)))

    for row, color in zip(rows, cmap):
        if row["_scores"] is None:
            continue
        fpr, tpr = roc_from_scores(
            row["_scores"][f"member_{signal}"],
            row["_scores"][f"nonmember_{signal}"],
        )
        auc = row["attacks"][signal]["auc"]
        style = dict(color=color, lw=2.0)
        if row["epsilon"] is None:
            style = dict(color="crimson", lw=2.4, ls="--")
        label = f"{row['label']} (AUC {auc:.3f})"
        axes[0].plot(fpr, tpr, label=label, **style)
        axes[1].plot(np.clip(fpr, 1e-5, 1), np.clip(tpr, 1e-5, 1), label=label, **style)

    axes[0].plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label="random guess")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC (linear)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="lower right")

    axes[1].plot([1e-5, 1], [1e-5, 1], color="grey", lw=1, ls=":", label="random guess")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC (log-log, low-FPR regime)")
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(1e-4, 1.0)
    axes[1].set_ylim(1e-4, 1.0)

    fig.suptitle(f"Threshold MIA per privacy budget — {signal} signal")
    fig.tight_layout()
    path = out_dir / "mia_roc_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def aggregate(rows, signal):
    """Group by noise multiplier so multi-seed runs collapse to mean +- std."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("noise_multiplier")].append(row)

    dp, baseline = [], None
    for nm, members in groups.items():
        eps = [r["epsilon"] for r in members if r["epsilon"] is not None]
        auc = np.array([r["attacks"][signal]["auc"] for r in members])
        acc = np.array([r["test_accuracy"] for r in members
                        if r["test_accuracy"] is not None])
        record = {
            "noise_multiplier": nm,
            "epsilon": float(np.mean(eps)) if eps else None,
            "auc_mean": float(auc.mean()), "auc_std": float(auc.std()),
            "acc_mean": float(acc.mean()) if acc.size else None,
            "acc_std": float(acc.std()) if acc.size else None,
            "n_seeds": len(members),
        }
        if eps:
            dp.append(record)
        else:
            baseline = record

    dp.sort(key=lambda r: r["epsilon"])
    return baseline, dp


def plot_auc_vs_epsilon(baseline, dp, signal, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    eps = [r["epsilon"] for r in dp]
    auc = [r["auc_mean"] for r in dp]
    err = [r["auc_std"] for r in dp]

    if any(e > 0 for e in err):
        ax.errorbar(eps, auc, yerr=err, marker="o", capsize=3, lw=2, color="#1f77b4",
                    label="DP-SGD")
    else:
        ax.plot(eps, auc, marker="o", lw=2, color="#1f77b4", label="DP-SGD")

    if baseline:
        ax.axhline(baseline["auc_mean"], color="crimson", ls="--", lw=2,
                   label=f"non-private baseline ({baseline['auc_mean']:.3f})")
    ax.axhline(0.5, color="grey", ls=":", lw=1.5, label="random guess (0.50)")

    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale)")
    ax.set_ylabel(f"Attack AUC-ROC ({signal})")
    ax.set_title("Membership inference risk vs privacy budget")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "mia_auc_vs_epsilon.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_tradeoff(baseline, dp, signal, out_dir):
    if not any(r["acc_mean"] is not None for r in dp):
        return None
    fig, ax = plt.subplots(figsize=(7.5, 5))
    eps = [r["epsilon"] for r in dp]

    ax.plot(eps, [r["acc_mean"] for r in dp], marker="o", color="#2ca02c", lw=2,
            label="test accuracy")
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Test accuracy (%)", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(eps, [r["auc_mean"] for r in dp], marker="s", color="#d62728", lw=2,
             label="attack AUC")
    ax2.axhline(0.5, color="grey", ls=":", lw=1.5)
    ax2.set_ylabel(f"Attack AUC-ROC ({signal})", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    if baseline and baseline["acc_mean"] is not None:
        ax.axhline(baseline["acc_mean"], color="#2ca02c", ls="--", lw=1, alpha=0.6)
        ax2.axhline(baseline["auc_mean"], color="#d62728", ls="--", lw=1, alpha=0.6)

    handles = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], loc="center right")
    ax.set_title("Privacy / utility / attack-risk tradeoff")
    fig.tight_layout()
    path = out_dir / "privacy_utility_attack.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def print_table(rows, signal):
    header = (f"{'model':<20}{'eps':>9}{'test acc':>11}{'gap':>9}"
              f"{'AUC':>8}{'atk acc':>9}{'TPR@1%':>9}{'TPR@0.1%':>10}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        a = row["attacks"][signal]
        eps = "inf" if row["epsilon"] is None else f"{row['epsilon']:.2f}"
        acc = "-" if row["test_accuracy"] is None else f"{row['test_accuracy']:.2f}%"
        print(f"{row['label']:<20}{eps:>9}{acc:>11}"
              f"{row['generalization_gap']:>+9.4f}{a['auc']:>8.4f}"
              f"{a['attack_accuracy']:>9.4f}{a['tpr_at_fpr_0.01']:>9.4f}"
              f"{a['tpr_at_fpr_0.001']:>10.4f}")


def main():
    try:
        import config
        default_out_dir = Path(getattr(config, "RESULTS_DIR", REPO_ROOT / "experiments/results"))
    except ImportError:
        default_out_dir = REPO_ROOT / "experiments/results"

    default_results = default_out_dir / "threshold_attack.json"
    if not default_results.exists():
        legacy = REPO_ROOT / "experiments/results/threshold_attack.json"
        if legacy.exists():
            default_results = legacy

    default_scores = default_out_dir / "mia_scores"
    if not default_scores.exists():
        legacy_scores = REPO_ROOT / "experiments/results/mia_scores"
        if legacy_scores.exists():
            default_scores = legacy_scores

    ap = argparse.ArgumentParser(description="Plot threshold MIA results")
    ap.add_argument("--results", default=str(default_results))
    ap.add_argument("--scores-dir", default=str(default_scores))
    ap.add_argument("--out-dir", default=str(default_out_dir))
    ap.add_argument("--signal", default="loss", choices=["loss", "confidence", "mentr"])
    args = ap.parse_args()

    if not Path(args.results).exists():
        raise SystemExit(f"{args.results} not found. Run src.threshold_attack first.")

    _, rows = load(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_table(rows, args.signal)

    written = [plot_roc(rows, args.signal, out_dir)]
    baseline, dp = aggregate(rows, args.signal)
    if dp:
        written.append(plot_auc_vs_epsilon(baseline, dp, args.signal, out_dir))
        tradeoff = plot_tradeoff(baseline, dp, args.signal, out_dir)
        if tradeoff:
            written.append(tradeoff)

    print("\nWrote:")
    for path in written:
        print(f"  {path}")

    if baseline and baseline["auc_mean"] < 0.55:
        print("\n[!] Baseline AUC < 0.55 — the non-private model is not memorizing "
              "enough for this attack to land. Read the 'Reading the results' note "
              "at the top of this file before writing up the finding.")


if __name__ == "__main__":
    main()
