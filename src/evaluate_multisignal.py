"""
Full-Grid Multi-Signal MIA Evaluator with Multiple Comparison Correction.

Evaluates differential privacy against Membership Inference Attacks on CIFAR-10
across all 45 existing checkpoints using three signals:
  1. loss: negative cross-entropy
  2. confidence: softmax probability of true class
  3. mentr: negative modified prediction entropy (Song & Mittal, USENIX Security 2021)

Computes 1000-resample bootstrap 95% CIs (via compute_bootstrap_auc_ci from src.sweep_clipping),
derives p-values for H0: AUC = 0.50, and applies Benjamini-Hochberg FDR correction at q = 0.05
across the full family of 135 tests (45 checkpoints x 3 signals).

Outputs:
  experiments/cifar10/results/multisignal_attack.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy import stats
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from src.utils import get_device
from src.sweep_clipping import compute_bootstrap_auc_ci
from src.threshold_attack import (
    load_model,
    build_loaders,
    score_dataset,
    attack_metrics,
    score_loss,
    score_confidence,
    score_mentr,
    SCORING_FUNCTIONS,
    _load_config,
)

# ── Configurations ─────────────────────────────────────────────────────────

EPSILON_NMS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0]
CLIP_NORMS = [0.5, 1.0, 5.0, 10.0, 50.0]
SEEDS = [42, 43, 44]
SIGNALS = ["loss", "confidence", "mentr"]


def get_checkpoint_list(checkpoints_dir: Path) -> list[dict]:
    """Build canonical list of 45 checkpoints with experimental metadata."""
    items = []

    # 1. Epsilon sweep: baseline + 9 noise multipliers x 3 seeds = 30 runs
    for nm in EPSILON_NMS:
        is_baseline = (nm == 0.0)
        for seed in SEEDS:
            fname = f"baseline_seed{seed}.pt" if is_baseline else f"dp_nm_{nm:.2f}_seed{seed}.pt"
            path = checkpoints_dir / fname
            label = "baseline (no DP)" if is_baseline else f"nm={nm:.2f}"
            items.append({
                "sweep": "epsilon",
                "noise_multiplier": nm,
                "clip_norm": 1.0,
                "seed": seed,
                "is_baseline": is_baseline,
                "config_key": "baseline" if is_baseline else f"nm_{nm:.2f}",
                "config_label": label,
                "checkpoint_path": path,
                "checkpoint_name": fname,
                "stem": path.stem,
            })

    # 2. Clipping sweep: 5 clip norms x 3 seeds = 15 runs
    for c in CLIP_NORMS:
        for seed in SEEDS:
            fname = f"clip_{c:.2f}_nm_0.00_seed{seed}.pt"
            path = checkpoints_dir / fname
            label = f"C={c:.2f} (nm=0.00)"
            items.append({
                "sweep": "clipping",
                "noise_multiplier": 0.0,
                "clip_norm": c,
                "seed": seed,
                "is_baseline": False,
                "config_key": f"clip_{c:.2f}",
                "config_label": label,
                "checkpoint_path": path,
                "checkpoint_name": fname,
                "stem": path.stem,
            })

    return items


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg procedure for controlling the False Discovery Rate at level q.

    Args:
        p_values: 1D array of raw p-values (length m).
        q: Target FDR level (default: 0.05).

    Returns:
        rejected: 1D boolean array where True indicates statistically significant under BH.
        q_values: 1D array of adjusted p-values (BH q-values).
    """
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    # Critical value at rank k (1-indexed): (k / m) * q
    k_indices = np.arange(1, m + 1)
    passed = sorted_p <= (k_indices / m) * q
    max_k = np.where(passed)[0]
    cutoff_rank = max_k[-1] if len(max_k) > 0 else -1

    # Adjusted p-values (q-values)
    adj_p = np.zeros(m)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        raw = (m / (i + 1)) * sorted_p[i]
        running_min = min(running_min, raw)
        adj_p[i] = min(1.0, running_min)

    rejected = np.zeros(m, dtype=bool)
    q_values = np.zeros(m, dtype=np.float64)

    for rank_idx, orig_idx in enumerate(order):
        rejected[orig_idx] = (rank_idx <= cutoff_rank)
        q_values[orig_idx] = adj_p[rank_idx]

    return rejected, q_values


def derive_auc_pvalue(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> tuple[float, float, float]:
    """
    Derive p-values for H0: AUC = 0.50 using asymptotic normal test (Mann-Whitney U).
    Returns (z_score, p_two_sided, p_greater).
    """
    n1 = len(member_scores)
    n2 = len(nonmember_scores)
    se0 = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))

    atk = attack_metrics(member_scores, nonmember_scores)
    auc = atk["auc"]
    z = (auc - 0.50) / se0

    p_two_sided = float(2.0 * stats.norm.sf(abs(z)))
    p_greater = float(stats.norm.sf(z))

    return float(z), p_two_sided, p_greater


def evaluate_grid(
    device: torch.device | None = None,
    save_scores: bool = True,
    n_bootstraps: int = 1000,
) -> dict:
    """Run full evaluation across the 45-checkpoint grid and 3 signals."""
    if device is None:
        device = get_device()

    cfg = _load_config()
    dataset = cfg.get("DATASET", "cifar10")
    checkpoints_dir = Path(cfg.get("CHECKPOINT_DIR", REPO_ROOT / f"experiments/{dataset}/checkpoints")).resolve()
    results_dir = Path(cfg.get("RESULTS_DIR", REPO_ROOT / f"experiments/{dataset}/results")).resolve()
    splits_dir = Path(cfg.get("SPLITS_DIR", REPO_ROOT / f"experiments/{dataset}/splits")).resolve()
    scores_dir = (results_dir / "mia_scores").resolve()
    scores_dir.mkdir(parents=True, exist_ok=True)

    n_train = getattr(config, "N_TRAIN", 5000)
    batch_size = getattr(config, "TEST_BATCH_SIZE", 1000)

    # 1. Preload data loaders per seed with strict split assertions
    loaders_by_seed = {}
    for seed in SEEDS:
        seed_npy = splits_dir / f"member_indices_n{n_train}_seed{seed}.npy"
        seed_json = splits_dir / f"member_indices_n{n_train}_seed{seed}.json"

        if seed_npy.exists():
            member_file = str(seed_npy)
        elif seed_json.exists():
            member_file = str(seed_json)
        elif seed == 42 and (splits_dir / "member_indices.npy").exists():
            member_file = str(splits_dir / "member_indices.npy")
        else:
            raise FileNotFoundError(f"Missing member index file for seed {seed} in {splits_dir}")

        # SPLIT CORRECTNESS ASSERTION 1: Verify split file belongs to this seed
        assert f"seed{seed}" in Path(member_file).name or (seed == 42 and Path(member_file).name == "member_indices.npy"), (
            f"FATAL: Member file '{member_file}' does not match seed {seed}!"
        )

        args_seed = SimpleNamespace(
            data_root=cfg["DATA_ROOT"],
            member_index_file=member_file,
            n_samples=n_train,
            batch_size=batch_size,
            num_workers=0,
            seed=seed,
        )
        mem_ldr, non_ldr, n_actual = build_loaders(args_seed, cfg)
        loaders_by_seed[seed] = (mem_ldr, non_ldr, n_actual, member_file)

    ckpt_items = get_checkpoint_list(checkpoints_dir)
    assert len(ckpt_items) == 45, f"Expected 45 checkpoints, found {len(ckpt_items)}"

    print("=" * 80)
    print(f"EVALUATING 3 SIGNALS ACROSS 45 CHECKPOINTS [{dataset.upper()}] on {device}")
    print(f"Signals: {SIGNALS}")
    print(f"Total hypothesis tests: 45 x 3 = 135")
    print("=" * 80)

    individual_tests = []
    checkpoint_results = []
    scores_cache = {}

    t0 = time.time()
    for idx, item in enumerate(ckpt_items, 1):
        ckpt_path = item["checkpoint_path"]
        seed = item["seed"]
        stem = item["stem"]

        assert ckpt_path.exists(), f"FATAL: Checkpoint missing on disk: {ckpt_path}"

        # SPLIT CORRECTNESS ASSERTION 2: Checkpoint file name must explicitly match target seed
        m = re.search(r"seed(\d+)", ckpt_path.name)
        assert m is not None and int(m.group(1)) == seed, (
            f"FATAL: Checkpoint {ckpt_path.name} seed mismatch: parsed {m.group(1) if m else None} vs {seed}"
        )

        mem_ldr, non_ldr, n_samples, split_file = loaders_by_seed[seed]

        # SPLIT CORRECTNESS ASSERTION 3: Loader split file must correspond to checkpoint seed
        assert f"seed{seed}" in Path(split_file).name or (seed == 42 and Path(split_file).name == "member_indices.npy"), (
            f"FATAL: Split file {split_file} does not belong to seed {seed} for checkpoint {ckpt_path.name}"
        )

        # Load model and score dataset
        model = load_model(ckpt_path, device)
        mem_scores = score_dataset(model, mem_ldr, device)
        non_scores = score_dataset(model, non_ldr, device, log_probs=mem_scores["_log_probs"])

        # Load checkpoint metadata for test accuracy and epsilon
        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            test_acc = ckpt_data.get("test_accuracy")
            test_loss = ckpt_data.get("test_loss")
            epsilon_meta = ckpt_data.get("epsilon")
        except Exception:
            test_acc = None
            test_loss = None
            epsilon_meta = None

        if item["is_baseline"]:
            epsilon_val = None
        elif epsilon_meta is not None and epsilon_meta != "inf":
            epsilon_val = float(epsilon_meta)
        else:
            epsilon_val = None

        # Save per-sample score arrays to disk for persistence and bootstrap pooling
        if save_scores:
            np.savez_compressed(
                scores_dir / f"{stem}.npz",
                member_loss=mem_scores["loss"],
                nonmember_loss=non_scores["loss"],
                member_confidence=mem_scores["confidence"],
                nonmember_confidence=non_scores["confidence"],
                member_mentr=mem_scores["mentr"],
                nonmember_mentr=non_scores["mentr"],
            )

        scores_cache[stem] = {
            "member_loss": mem_scores["loss"],
            "nonmember_loss": non_scores["loss"],
            "member_confidence": mem_scores["confidence"],
            "nonmember_confidence": non_scores["confidence"],
            "member_mentr": mem_scores["mentr"],
            "nonmember_mentr": non_scores["mentr"],
        }

        ckpt_row = {
            "sweep": item["sweep"],
            "config_key": item["config_key"],
            "config_label": item["config_label"],
            "noise_multiplier": item["noise_multiplier"],
            "clip_norm": item["clip_norm"],
            "seed": seed,
            "epsilon": epsilon_val,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "member_accuracy": float(mem_scores["accuracy"]),
            "nonmember_accuracy": float(non_scores["accuracy"]),
            "generalization_gap": float(mem_scores["accuracy"] - non_scores["accuracy"]),
            "checkpoint_path": str(ckpt_path.resolve().relative_to(REPO_ROOT.resolve())),
            "split_file": str(Path(split_file).resolve().relative_to(REPO_ROOT.resolve())),
            "signals": {},
        }

        for sig in SIGNALS:
            m_s = mem_scores[sig]
            nm_s = non_scores[sig]

            # Numerical stability assertion: strictly finite scores
            assert np.all(np.isfinite(m_s)), f"FATAL: Non-finite values in {stem} member_{sig}"
            assert np.all(np.isfinite(nm_s)), f"FATAL: Non-finite values in {stem} nonmember_{sig}"

            atk = attack_metrics(m_s, nm_s)
            z_score, p_two, p_grt = derive_auc_pvalue(m_s, nm_s)

            sig_metrics = {
                "signal": sig,
                "attack_auc": float(atk["auc"]),
                "attack_accuracy": float(atk["attack_accuracy"]),
                "tpr_at_1pct_fpr": float(atk["tpr_at_fpr_0.01"]),
                "tpr_at_0.1pct_fpr": float(atk["tpr_at_fpr_0.001"]),
                "best_threshold": float(atk["best_threshold"]) if atk["best_threshold"] is not None else None,
                "z_score": z_score,
                "uncorrected_p": p_two,
                "p_value_greater": p_grt,
                "significant_uncorrected": bool(p_two < 0.05),
            }
            ckpt_row["signals"][sig] = sig_metrics

            test_record = {
                "test_id": len(individual_tests),
                "sweep": item["sweep"],
                "config_key": item["config_key"],
                "config_label": item["config_label"],
                "noise_multiplier": item["noise_multiplier"],
                "clip_norm": item["clip_norm"],
                "seed": seed,
                "signal": sig,
                "stem": stem,
                "checkpoint": str(ckpt_path.resolve().relative_to(REPO_ROOT.resolve())),
                "attack_auc": float(atk["auc"]),
                "attack_accuracy": float(atk["attack_accuracy"]),
                "tpr_at_1pct_fpr": float(atk["tpr_at_fpr_0.01"]),
                "tpr_at_0.1pct_fpr": float(atk["tpr_at_fpr_0.001"]),
                "z_score": z_score,
                "uncorrected_p": p_two,
                "p_value_greater": p_grt,
                "significant_uncorrected": bool(p_two < 0.05),
            }
            individual_tests.append(test_record)

        checkpoint_results.append(ckpt_row)

        l_auc = ckpt_row["signals"]["loss"]["attack_auc"]
        c_auc = ckpt_row["signals"]["confidence"]["attack_auc"]
        m_auc = ckpt_row["signals"]["mentr"]["attack_auc"]
        print(
            f"[{idx:02d}/45] {item['sweep']:8} | {item['config_label']:16} | seed={seed} | "
            f"Loss AUC={l_auc:.4f} | Conf AUC={c_auc:.4f} | Mentr AUC={m_auc:.4f}"
        )

    scoring_time = time.time() - t0
    print(f"\nCompleted grid scoring in {scoring_time:.2f}s")
    assert len(individual_tests) == 135, f"Expected 135 tests, got {len(individual_tests)}"

    # ── Multiple Comparison Correction (Benjamini-Hochberg) ────────────
    print("\nApplying Benjamini-Hochberg FDR correction at q = 0.05 across full family of 135 tests...")

    # Two-sided p-values for H0: AUC = 0.50
    p_vals_2 = np.array([t["uncorrected_p"] for t in individual_tests])
    rej_2, q_vals_2 = benjamini_hochberg(p_vals_2, q=0.05)

    # One-sided (greater) p-values for H1: AUC > 0.50
    p_vals_g = np.array([t["p_value_greater"] for t in individual_tests])
    rej_g, q_vals_g = benjamini_hochberg(p_vals_g, q=0.05)

    for i, t in enumerate(individual_tests):
        t["bh_q_value"] = float(q_vals_2[i])
        t["bh_significant"] = bool(rej_2[i])
        t["bh_verdict"] = "Significant" if rej_2[i] else "Not Significant"

        t["bh_q_value_greater"] = float(q_vals_g[i])
        t["bh_significant_greater"] = bool(rej_g[i])
        t["bh_verdict_greater"] = "Significant" if rej_g[i] else "Not Significant"

        # Update in checkpoint_results
        stem = t["stem"]
        sig = t["signal"]
        for cr in checkpoint_results:
            if cr["checkpoint_path"] == t["checkpoint"]:
                cr["signals"][sig]["bh_q_value"] = float(q_vals_2[i])
                cr["signals"][sig]["bh_significant"] = bool(rej_2[i])
                cr["signals"][sig]["bh_verdict"] = "Significant" if rej_2[i] else "Not Significant"
                cr["signals"][sig]["bh_q_value_greater"] = float(q_vals_g[i])
                cr["signals"][sig]["bh_significant_greater"] = bool(rej_g[i])
                cr["signals"][sig]["bh_verdict_greater"] = "Significant" if rej_g[i] else "Not Significant"

    n_sig_uncorr_2 = sum(1 for t in individual_tests if t["significant_uncorrected"])
    n_sig_bh_2 = sum(1 for t in individual_tests if t["bh_significant"])
    n_sig_uncorr_g = sum(1 for t in individual_tests if t["p_value_greater"] < 0.05)
    n_sig_bh_g = sum(1 for t in individual_tests if t["bh_significant_greater"])

    print(f"Two-sided (AUC != 0.50): uncorrected p < 0.05 = {n_sig_uncorr_2}/135, BH-corrected = {n_sig_bh_2}/135")
    print(f"One-sided (AUC > 0.50) : uncorrected p < 0.05 = {n_sig_uncorr_g}/135, BH-corrected = {n_sig_bh_g}/135")

    # ── Per-Configuration Aggregation ──────────────────────────────────
    print(f"\nComputing 1000-resample bootstrap 95% CIs and configuration aggregates...")
    aggregates = {}

    configs_meta = []
    for nm in EPSILON_NMS:
        configs_meta.append({
            "sweep": "epsilon",
            "config_key": "baseline" if nm == 0.0 else f"nm_{nm:.2f}",
            "config_label": "baseline (no DP)" if nm == 0.0 else f"nm={nm:.2f}",
            "noise_multiplier": nm,
            "clip_norm": 1.0,
            "is_baseline": (nm == 0.0),
        })
    for c in CLIP_NORMS:
        configs_meta.append({
            "sweep": "clipping",
            "config_key": f"clip_{c:.2f}",
            "config_label": f"C={c:.2f} (nm=0.00)",
            "noise_multiplier": 0.0,
            "clip_norm": c,
            "is_baseline": False,
        })

    for cfg_meta in configs_meta:
        sweep = cfg_meta["sweep"]
        cfg_key = cfg_meta["config_key"]
        cfg_label = cfg_meta["config_label"]

        matching_ckpts = [
            cr for cr in checkpoint_results
            if cr["sweep"] == sweep and cr["config_key"] == cfg_key
        ]
        assert len(matching_ckpts) == 3, f"Expected 3 runs for {sweep} {cfg_key}, got {len(matching_ckpts)}"

        eps_vals = [cr["epsilon"] for cr in matching_ckpts if cr["epsilon"] is not None]
        eps_scalar = eps_vals[0] if eps_vals else None

        test_accs = [cr["test_accuracy"] for cr in matching_ckpts if cr["test_accuracy"] is not None]
        test_losses = [cr["test_loss"] for cr in matching_ckpts if cr["test_loss"] is not None]
        gen_gaps = [cr["generalization_gap"] for cr in matching_ckpts]

        agg_entry = {
            "sweep": sweep,
            "config_key": cfg_key,
            "config_label": cfg_label,
            "noise_multiplier": cfg_meta["noise_multiplier"],
            "clip_norm": cfg_meta["clip_norm"],
            "epsilon": eps_scalar,
            "n_seeds": len(matching_ckpts),
            "test_accuracy_mean": float(np.mean(test_accs)) if test_accs else None,
            "test_accuracy_std": float(np.std(test_accs)) if test_accs else None,
            "test_loss_mean": float(np.mean(test_losses)) if test_losses else None,
            "test_loss_std": float(np.std(test_losses)) if test_losses else None,
            "generalization_gap_mean": float(np.mean(gen_gaps)),
            "generalization_gap_std": float(np.std(gen_gaps)),
            "signals": {},
        }

        for sig in SIGNALS:
            aucs = [cr["signals"][sig]["attack_auc"] for cr in matching_ckpts]
            accs = [cr["signals"][sig]["attack_accuracy"] for cr in matching_ckpts]
            tprs_1 = [cr["signals"][sig]["tpr_at_1pct_fpr"] for cr in matching_ckpts]
            tprs_01 = [cr["signals"][sig]["tpr_at_0.1pct_fpr"] for cr in matching_ckpts]
            p_uncorr = [cr["signals"][sig]["uncorrected_p"] for cr in matching_ckpts]
            bh_sigs = [cr["signals"][sig]["bh_significant"] for cr in matching_ckpts]

            # Prepare seed scores for compute_bootstrap_auc_ci
            seed_scores = []
            for cr in matching_ckpts:
                stem = Path(cr["checkpoint_path"]).stem
                d = scores_cache[stem]
                seed_scores.append({
                    "mem_loss": d[f"member_{sig}"],
                    "non_loss": d[f"nonmember_{sig}"],
                })

            rng = np.random.default_rng(42)
            ci_lower, ci_upper = compute_bootstrap_auc_ci(
                seed_scores, n_bootstraps=n_bootstraps, rng=rng
            )

            # Derive p-value per configuration pooling member/non-member scores across the 3 seeds
            pooled_m = np.concatenate([s["mem_loss"] for s in seed_scores])
            pooled_nm = np.concatenate([s["non_loss"] for s in seed_scores])
            n1_pool = len(pooled_m)
            n2_pool = len(pooled_nm)
            se_pool = np.sqrt((n1_pool + n2_pool + 1) / (12.0 * n1_pool * n2_pool))
            mean_auc = float(np.mean(aucs))
            z_pool = (mean_auc - 0.50) / se_pool
            p_pooled_two = float(2.0 * stats.norm.sf(abs(z_pool)))
            p_pooled_grt = float(stats.norm.sf(z_pool))

            distinguishable = bool(ci_lower > 0.50)
            all_seeds_bh = bool(all(bh_sigs))
            any_seed_bh = bool(any(bh_sigs))

            agg_entry["signals"][sig] = {
                "signal": sig,
                "mean_auc": mean_auc,
                "std_auc": float(np.std(aucs)),
                "bootstrap_ci": [float(ci_lower), float(ci_upper)],
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "distinguishable_from_random": distinguishable,
                "mean_attack_accuracy": float(np.mean(accs)),
                "std_attack_accuracy": float(np.std(accs)),
                "mean_tpr_at_1pct_fpr": float(np.mean(tprs_1)),
                "std_tpr_at_1pct_fpr": float(np.std(tprs_1)),
                "mean_tpr_at_0.1pct_fpr": float(np.mean(tprs_01)),
                "std_tpr_at_0.1pct_fpr": float(np.std(tprs_01)),
                "uncorrected_p": p_pooled_two,
                "uncorrected_p_greater": p_pooled_grt,
                "seed_uncorrected_p": p_uncorr,
                "seed_bh_significant": bh_sigs,
                "all_seeds_bh_significant": all_seeds_bh,
                "any_seed_bh_significant": any_seed_bh,
                "bh_verdict": "Significant" if all_seeds_bh else ("Partial" if any_seed_bh else "Not Significant"),
                "bh_verdict_any": "Significant" if any_seed_bh else "Not Significant",
            }

        aggregates[f"{sweep}_{cfg_key}"] = agg_entry

    payload = {
        "metadata": {
            "dataset": dataset,
            "n_checkpoints": 45,
            "n_tests": 135,
            "n_bootstraps": n_bootstraps,
            "fdr_q": 0.05,
            "signals": SIGNALS,
            "seeds": SEEDS,
            "scoring_wall_clock_seconds": round(scoring_time, 2),
        },
        "summary": {
            "n_uncorrected_significant_two_sided": n_sig_uncorr_2,
            "n_bh_significant_two_sided": n_sig_bh_2,
            "n_uncorrected_significant_greater": n_sig_uncorr_g,
            "n_bh_significant_greater": n_sig_bh_g,
        },
        "aggregates": aggregates,
        "runs": individual_tests,
        "checkpoints": checkpoint_results,
    }

    out_json = results_dir / "multisignal_attack.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSuccessfully wrote full multi-signal attack results to {out_json}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full grid multi-signal MIA evaluation")
    parser.add_argument("--device", default=None, help="Execution device (cuda or cpu)")
    parser.add_argument("--n-bootstraps", type=int, default=1000, help="Number of bootstrap resamples")
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else None
    evaluate_grid(device=dev, n_bootstraps=args.n_bootstraps)
