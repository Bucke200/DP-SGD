"""
Plot Attack AUC vs Generalization Gap across DP-SGD Mechanisms.

Unifying visualization analyzing whether empirical Membership Inference Attack (MIA)
vulnerability is governed strictly by the generalization gap across both
per-sample gradient clipping and Gaussian noise mechanisms.

Pools 15 configurations across:
  - Baseline (non-private, unclipped)
  - 9 Noise-varied configurations (C=1.0, sigma in [0.3, 5.0])
  - 5 Clipping-varied configurations (sigma=0.0, C in [0.5, 50.0])

Saves visualization to:
  experiments/cifar10/results/gap_vs_auc.png
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
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

RESULTS_DIR = Path(getattr(config, "RESULTS_DIR", REPO_ROOT / "experiments/cifar10/results")).resolve()
DEFAULT_INPUT = RESULTS_DIR / "multisignal_attack.json"
DEFAULT_OUTPUT = RESULTS_DIR / "gap_vs_auc.png"


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


def log_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Logarithmic model: y = a + b * ln(x)."""
    return a + b * np.log(np.maximum(x, 1e-6))


def logistic_3p(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """3-parameter logistic model with lower asymptote fixed at random guessing (0.50)."""
    return 0.50 + L / (1.0 + np.exp(-np.clip(k * (x - x0), -50, 50)))


def load_gap_auc_data(input_path: Path) -> list[dict]:
    """Extract gap and AUC points across all 15 configurations."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for k, v in data["aggregates"].items():
        gap_pct = v["generalization_gap_mean"] * 100.0  # in percent
        gap_std = v["generalization_gap_std"] * 100.0
        auc = v["signals"]["loss"]["mean_auc"]
        ci = v["signals"]["loss"]["bootstrap_ci"]
        ci_err_lower = auc - ci[0]
        ci_err_upper = ci[1] - auc

        if k == "epsilon_baseline":
            mech = "baseline"
            group_label = "Baseline (no DP, unclipped)"
        elif k.startswith("epsilon_nm_"):
            mech = "noise_varied"
            group_label = "Noise-varied (C=1.0, varying \u03c3)"
        elif k.startswith("clipping_clip_"):
            mech = "clipping_varied"
            group_label = "Clipping-varied (\u03c3=0.0, varying C)"
        else:
            continue

        points.append({
            "key": k,
            "mech": mech,
            "group_label": group_label,
            "config_label": v["config_label"],
            "noise_multiplier": v["noise_multiplier"],
            "clip_norm": v["clip_norm"],
            "gap": gap_pct,
            "gap_std": gap_std,
            "auc": auc,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "ci_err_lower": ci_err_lower,
            "ci_err_upper": ci_err_upper,
        })

    return points


def fit_models(points: list[dict]) -> dict:
    """Fit logistic and logarithmic models across pooled data and by mechanism group."""
    x_all = np.array([p["gap"] for p in points])
    y_all = np.array([p["auc"] for p in points])

    # 1. Pooled Fits
    popt_logis_pooled, _ = curve_fit(logistic_3p, x_all, y_all, p0=[0.36, 0.12, 25.0], maxfev=10000)
    r2_logis_pooled = compute_r2(y_all, logistic_3p(x_all, *popt_logis_pooled))

    popt_log_pooled, _ = curve_fit(log_func, x_all, y_all)
    r2_log_pooled = compute_r2(y_all, log_func(x_all, *popt_log_pooled))

    # 2. Clipping-varied Group Fits (N=5)
    pts_clip = [p for p in points if p["mech"] == "clipping_varied"]
    x_clip = np.array([p["gap"] for p in pts_clip])
    y_clip = np.array([p["auc"] for p in pts_clip])

    popt_logis_clip, _ = curve_fit(logistic_3p, x_clip, y_clip, p0=[0.36, 0.12, 25.0], maxfev=10000)
    r2_logis_clip = compute_r2(y_clip, logistic_3p(x_clip, *popt_logis_clip))

    popt_log_clip, _ = curve_fit(log_func, x_clip, y_clip)
    r2_log_clip = compute_r2(y_clip, log_func(x_clip, *popt_log_clip))

    # 3. Noise-varied Group Fits (N=9)
    pts_noise = [p for p in points if p["mech"] == "noise_varied"]
    x_noise = np.array([p["gap"] for p in pts_noise])
    y_noise = np.array([p["auc"] for p in pts_noise])

    popt_logis_noise, _ = curve_fit(logistic_3p, x_noise, y_noise, p0=[0.05, 1.0, 1.0], maxfev=10000)
    r2_logis_noise = compute_r2(y_noise, logistic_3p(x_noise, *popt_logis_noise))

    popt_log_noise, _ = curve_fit(log_func, x_noise, y_noise)
    r2_log_noise = compute_r2(y_noise, log_func(x_noise, *popt_log_noise))

    return {
        "pooled": {
            "logistic": {"params": popt_logis_pooled.tolist(), "r2": r2_logis_pooled},
            "log": {"params": popt_log_pooled.tolist(), "r2": r2_log_pooled},
        },
        "clipping_varied": {
            "n": len(pts_clip),
            "logistic": {"params": popt_logis_clip.tolist(), "r2": r2_logis_clip},
            "log": {"params": popt_log_clip.tolist(), "r2": r2_log_clip},
        },
        "noise_varied": {
            "n": len(pts_noise),
            "logistic": {"params": popt_logis_noise.tolist(), "r2": r2_logis_noise},
            "log": {"params": popt_log_noise.tolist(), "r2": r2_log_noise},
        },
    }


def plot_gap_vs_auc(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict:
    """Generate and save publication figure for generalization gap vs attack AUC."""
    points = load_gap_auc_data(input_path)
    fit_res = fit_models(points)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)

    # Styling settings per mechanism group
    group_styles = {
        "baseline": {
            "color": "#111111",
            "marker": "*",
            "markersize": 14,
            "label": "Baseline (no DP, unclipped)",
            "zorder": 5,
        },
        "noise_varied": {
            "color": "#1f77b4",
            "marker": "s",
            "markersize": 8,
            "label": "Noise-varied (C=1.0, varying \u03c3 \u2208 [0.3, 5.0])",
            "zorder": 4,
        },
        "clipping_varied": {
            "color": "#d62728",
            "marker": "o",
            "markersize": 8,
            "label": "Clipping-varied (\u03c3=0.0, varying C \u2208 [0.5, 50.0])",
            "zorder": 4,
        },
    }

    # Plot data points with error bars (seed std on x, bootstrap 95% CI on y)
    for mech, style in group_styles.items():
        mech_pts = [p for p in points if p["mech"] == mech]
        if not mech_pts:
            continue

        x = np.array([p["gap"] for p in mech_pts])
        y = np.array([p["auc"] for p in mech_pts])
        xerr = np.array([p["gap_std"] for p in mech_pts])
        yerr = np.array([[p["ci_err_lower"] for p in mech_pts],
                         [p["ci_err_upper"] for p in mech_pts]])

        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.0,
            markersize=style["markersize"],
            label=style["label"],
            zorder=style["zorder"],
            alpha=0.92,
        )

    # Horizontal reference line at AUC = 0.50 (Random Guessing)
    ax.axhline(
        0.50,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.8,
        label="Random Guessing (AUC = 0.50)",
        zorder=2,
    )

    # Plot fitted pooled curves across full range [0, 52]
    x_dense = np.linspace(0.1, 52.0, 500)
    popt_logis = fit_res["pooled"]["logistic"]["params"]
    r2_logis = fit_res["pooled"]["logistic"]["r2"]
    y_dense_logis = logistic_3p(x_dense, *popt_logis)

    ax.plot(
        x_dense,
        y_dense_logis,
        color="#2ca02c",
        linestyle="-",
        linewidth=2.4,
        label=f"Pooled Logistic Fit ($R^2 = {r2_logis:.4f}$)",
        zorder=3,
    )

    # Annotate key landmark points for clear paper reference
    annotations = [
        ("nm=5.00", (0.38, 0.4995), (-25, 14), "\u03c3=5.0 (\u03b5=0.59)"),
        ("nm=3.00", (0.87, 0.5090), (-25, -18), "\u03c3=3.0 (\u03b5=1.05)"),
        ("nm=1.10", (2.34, 0.5188), (10, -18), "\u03c3=1.1 (default)"),
        ("clip_0.50", (1.91, 0.5138), (-35, 12), "C=0.5 (\u03c3=0)"),
        ("clip_1.00", (3.37, 0.5230), (10, 10), "C=1.0 (\u03c3=0)"),
        ("clip_5.00", (18.91, 0.6198), (12, -12), "C=5.0 (\u03c3=0)"),
        ("clip_10.00", (46.11, 0.7780), (-65, -16), "C=10.0 (\u03c3=0)"),
        ("clip_50.00", (50.14, 0.8421), (-65, 10), "C=50.0 (\u03c3=0)"),
        ("baseline", (49.11, 0.8578), (-70, -22), "Baseline (unclipped)"),
    ]

    for key_pattern, (px, py), (dx, dy), text in annotations:
        ax.annotate(
            text,
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb", alpha=0.85),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#666666", lw=0.8),
            zorder=6,
        )

    # Inset / Text Box with regression statistics
    r2_clip_logis = fit_res["clipping_varied"]["logistic"]["r2"]
    r2_noise_logis = fit_res["noise_varied"]["logistic"]["r2"]
    r2_clip_log = fit_res["clipping_varied"]["log"]["r2"]
    r2_noise_log = fit_res["noise_varied"]["log"]["r2"]
    r2_log_pooled = fit_res["pooled"]["log"]["r2"]

    stats_text = (
        "Model Fits & R\u00b2:\n"
        f"  Pooled Logistic: R\u00b2 = {r2_logis:.4f}\n"
        f"  Pooled Log: R\u00b2 = {r2_log_pooled:.4f}\n"
        "Separate Logistic R\u00b2:\n"
        f"  Clipping-varied: {r2_clip_logis:.4f}\n"
        f"  Noise-varied: {r2_noise_logis:.4f}\n"
        "Separate Log R\u00b2:\n"
        f"  Clipping-varied: {r2_clip_log:.4f}\n"
        f"  Noise-varied: {r2_noise_log:.4f}"
    )
    ax.text(
        0.03, 0.46,
        stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.92),
        zorder=5,
    )

    # Axes styling
    ax.set_xlabel("Mean Generalization Gap $\\Delta_{\\mathrm{gen}}$ (%) [\u00b1 Seed Std]", fontsize=12)
    ax.set_ylabel("MIA Attack AUC (Loss Signal) [95% Bootstrap CI]", fontsize=12)
    ax.set_title(
        "MIA Vulnerability vs. Generalization Gap Across DP-SGD Mechanisms (CIFAR-10)\n"
        "Clipping-Varied (\u03c3=0.0) vs. Noise-Varied (C=1.0) Collapse onto a Single Curve",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlim(-1.5, 54.0)
    ax.set_ylim(0.48, 0.88)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=1)

    ax.legend(
        loc="upper left",
        fontsize=9.5,
        framealpha=0.92,
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated plot at: {output_path}")

    return {
        "points": points,
        "fit_results": fit_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot generalization gap vs attack AUC")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to multisignal_attack.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to save gap_vs_auc.png")
    args = parser.parse_args()

    res = plot_gap_vs_auc(Path(args.input), Path(args.output))

    print("\n" + "=" * 80)
    print("REGRESSION FIT SUMMARY (Attack AUC vs Generalization Gap)")
    print("=" * 80)
    fits = res["fit_results"]
    print(f"Pooled Logistic Fit: R^2 = {fits['pooled']['logistic']['r2']:.4f}")
    print(f"Pooled Log Fit     : R^2 = {fits['pooled']['log']['r2']:.4f}")
    print("\nSeparate Group Fits:")
    print(f"  Clipping-varied (N=5): Logistic R^2 = {fits['clipping_varied']['logistic']['r2']:.4f}, Log R^2 = {fits['clipping_varied']['log']['r2']:.4f}")
    print(f"  Noise-varied    (N=9): Logistic R^2 = {fits['noise_varied']['logistic']['r2']:.4f}, Log R^2 = {fits['noise_varied']['log']['r2']:.4f}")


if __name__ == "__main__":
    main()
