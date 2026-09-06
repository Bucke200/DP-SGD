"""
Objective 2: Multi-ε Sweep with Model Checkpointing & Multi-Seed Support

Trains models under multiple noise multipliers across multiple seeds to produce
a collection of models at different privacy levels. Reuses existing checkpoints
and data splits on disk, asserts epsilon consistency, evaluates threshold MIA,
and writes multi-seed aggregated metrics with bootstrap CIs to
experiments/{dataset}/results/epsilon_sweep_multiseed.json.

Usage:
    # Single seed (default backwards compatible)
    python -m src.sweep_epsilon

    # Multi-seed for paper-ready results (seeds 42 43 44)
    python -m src.sweep_epsilon --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

import config
from src.data_split import get_data_loaders
from src.model import get_model, CifarCNN
from src.evaluate import evaluate
from src.utils import seed_everything, get_device
from src.sweep_clipping import compute_bootstrap_auc_ci
from src.threshold_attack import (
    load_model,
    build_loaders,
    score_dataset,
    attack_metrics,
    _load_config,
)

# ── Sweep configuration ─────────────────────────────────────────
NOISE_MULTIPLIERS = [0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0]

CHECKPOINT_DIR = getattr(config, "CHECKPOINT_DIR", "experiments/cifar10/checkpoints")
RESULTS_DIR = getattr(config, "RESULTS_DIR", "experiments/cifar10/results")
SPLITS_DIR = getattr(config, "SPLITS_DIR", "experiments/cifar10/splits")


# ── Checkpoint naming & migration ────────────────────────────────

def baseline_ckpt_name(seed: int) -> str:
    """baseline_seed42.pt"""
    return f"baseline_seed{seed}.pt"


def dp_ckpt_name(noise_multiplier: float, seed: int) -> str:
    """dp_nm_1.10_seed42.pt"""
    return f"dp_nm_{noise_multiplier:.2f}_seed{seed}.pt"


def migrate_checkpoints(checkpoint_dir: str, default_seed: int = 42):
    """
    Migration step: Check for any seed-42 checkpoint files in checkpoint_dir that
    lack the '_seed' suffix and rename them to include '_seed42.pt' before the sweep starts.
    Prints what was renamed.
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Check unseeded baseline
    legacy_base = os.path.join(checkpoint_dir, "baseline.pt")
    seeded_base = os.path.join(checkpoint_dir, baseline_ckpt_name(default_seed))
    if os.path.exists(legacy_base) and not os.path.exists(seeded_base):
        os.rename(legacy_base, seeded_base)
        print(f"[migration] Renamed '{legacy_base}' -> '{seeded_base}'")

    # Check unseeded DP checkpoints
    for nm in NOISE_MULTIPLIERS:
        legacy_dp = os.path.join(checkpoint_dir, f"dp_nm_{nm:.2f}.pt")
        seeded_dp = os.path.join(checkpoint_dir, dp_ckpt_name(nm, default_seed))
        if os.path.exists(legacy_dp) and not os.path.exists(seeded_dp):
            os.rename(legacy_dp, seeded_dp)
            print(f"[migration] Renamed '{legacy_dp}' -> '{seeded_dp}'")


# ── Checkpoint I/O ───────────────────────────────────────

def save_checkpoint(model: nn.Module, metadata: dict, path: str):
    """
    Save model weights and full experimental metadata.
    Unwraps Opacus GradSampleModule via _module to ensure clean model loading.
    """
    if hasattr(model, "_module"):
        state_dict = model._module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save({"model_state_dict": state_dict, **metadata}, path)


def load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint into a clean model instance for verification."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = {k.replace("_module.", "", 1): v for k, v in state_dict.items()}

    model = get_model(config.DATASET).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


# ── Training routines ────────────────────────────────────────────

def train_baseline(seed: int, device: torch.device):
    """Train a standard SGD model (no privacy)."""
    seed_everything(seed)
    train_loader, test_loader, member_indices = get_data_loaders(seed=seed, save_indices=True)

    train_eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=getattr(config, "TEST_BATCH_SIZE", 1000),
        shuffle=False,
    )

    model = get_model(config.DATASET).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=getattr(config, "MOMENTUM", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    train_loss, train_acc = evaluate(model, train_eval_loader, device, criterion)
    generalization_gap = round((train_acc - test_acc) / 100.0, 6)

    metadata = {
        "dataset": config.DATASET,
        "model_type": "baseline",
        "epsilon": "inf",
        "delta": None,
        "noise_multiplier": 0.0,
        "max_grad_norm": None,
        "clip_norm": getattr(config, "MAX_GRAD_NORM", 1.0),
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "seed": seed,
        "test_accuracy": round(test_acc, 2),
        "test_loss": round(test_loss, 4),
        "train_accuracy": round(train_acc, 2),
        "train_loss": round(train_loss, 4),
        "generalization_gap": generalization_gap,
        "training_time_sec": round(elapsed, 2),
    }
    return model, metadata


def train_dp(noise_multiplier: float, seed: int, device: torch.device):
    """Train a DP-SGD model with a given noise multiplier."""
    seed_everything(seed)
    train_loader, test_loader, member_indices = get_data_loaders(seed=seed, save_indices=True)

    train_eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=getattr(config, "TEST_BATCH_SIZE", 1000),
        shuffle=False,
    )

    model = get_model(config.DATASET).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=getattr(config, "MOMENTUM", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    start = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    # ε is an OUTPUT of the accountant — never hardcode it
    epsilon = privacy_engine.get_epsilon(delta=config.DELTA)

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    train_loss, train_acc = evaluate(model, train_eval_loader, device, criterion)
    generalization_gap = round((train_acc - test_acc) / 100.0, 6)

    metadata = {
        "dataset": config.DATASET,
        "model_type": "dp-sgd",
        "epsilon": round(epsilon, 6),
        "delta": config.DELTA,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "clip_norm": config.MAX_GRAD_NORM,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "seed": seed,
        "test_accuracy": round(test_acc, 2),
        "test_loss": round(test_loss, 4),
        "train_accuracy": round(train_acc, 2),
        "train_loss": round(train_loss, 4),
        "generalization_gap": generalization_gap,
        "training_time_sec": round(elapsed, 2),
    }
    return model, metadata


# ── Training Sweep Manager ───────────────────────────────────────

def run_epsilon_sweep(
    seeds: list[int],
    noise_multipliers: list[float] = NOISE_MULTIPLIERS,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> list[dict]:
    """
    Run multi-seed epsilon sweep across all noise multipliers plus baseline.
    Reuses existing checkpoints on disk by re-deriving evaluation metrics.
    Prints C/seed/elapsed per run.
    """
    device = get_device()
    os.makedirs(checkpoint_dir, exist_ok=True)
    migrate_checkpoints(checkpoint_dir, default_seed=getattr(config, "SEED", 42))

    all_configs = [0.0] + list(noise_multipliers)
    total_runs = len(all_configs) * len(seeds)
    run_idx = 0
    all_runs = []

    print("=" * 70)
    print(f"DP-SGD Epsilon Sweep [{config.DATASET.upper()}]")
    print(f"  Clip norm C:        {config.MAX_GRAD_NORM}")
    print(f"  Noise multipliers:  {noise_multipliers}")
    print(f"  Seeds:              {seeds}")
    print(f"  Total runs:         {total_runs} (1 baseline + {len(noise_multipliers)} DP × {len(seeds)} seeds)")
    print(f"  Device:             {device}")
    print("=" * 70)

    # Dictionary to collect epsilons across seeds per noise multiplier for consistency check
    epsilons_by_nm: dict[float, list[tuple[int, float]]] = {}

    criterion = nn.CrossEntropyLoss()

    for nm in all_configs:
        for seed in seeds:
            run_idx += 1
            seed_everything(seed)

            is_baseline = (nm == 0.0)
            ckpt_file = baseline_ckpt_name(seed) if is_baseline else dp_ckpt_name(nm, seed)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)

            # Check if checkpoint already exists on disk -> REUSE WORK
            if os.path.exists(ckpt_path):
                t0 = time.time()
                try:
                    model, ckpt_data = load_checkpoint(ckpt_path, device)
                    train_loader, test_loader, _ = get_data_loaders(seed=seed, save_indices=False)
                    train_eval_loader = DataLoader(
                        train_loader.dataset,
                        batch_size=getattr(config, "TEST_BATCH_SIZE", 1000),
                        shuffle=False,
                    )
                    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
                    train_loss, train_acc = evaluate(model, train_eval_loader, device, criterion)
                    generalization_gap = round((train_acc - test_acc) / 100.0, 6)
                    elapsed = time.time() - t0

                    meta = {k: v for k, v in ckpt_data.items() if k != "model_state_dict"}
                    meta["checkpoint"] = ckpt_path
                    meta["checkpoint_path"] = ckpt_path
                    meta["test_accuracy"] = round(test_acc, 2)
                    meta["test_loss"] = round(test_loss, 4)
                    meta["train_accuracy"] = round(train_acc, 2)
                    meta["train_loss"] = round(train_loss, 4)
                    meta["generalization_gap"] = generalization_gap
                    meta["noise_multiplier"] = nm
                    meta["seed"] = seed
                    meta["clip_norm"] = getattr(config, "MAX_GRAD_NORM", 1.0)

                    eps_val = meta.get("epsilon")
                    eps_str = "inf" if is_baseline or eps_val == "inf" or eps_val is None else f"{float(eps_val):.4f}"

                    print(
                        f"[{run_idx}/{total_runs}] SKIP (cached): nm={nm:<4.2f} | "
                        f"C={config.MAX_GRAD_NORM} | seed={seed} | Acc={test_acc:.2f}% | "
                        f"Gap={generalization_gap:+.4f} | eps={eps_str} -> {ckpt_file}"
                    )
                    print(f"  C={config.MAX_GRAD_NORM} | seed={seed} | elapsed={elapsed:.2f}s")
                    all_runs.append(meta)

                    if not is_baseline:
                        epsilons_by_nm.setdefault(nm, []).append((seed, float(meta["epsilon"])))
                    continue

                except Exception as e:
                    print(f"[warn] Failed to reuse {ckpt_path} ({e}), will retrain.")

            # Train run
            t0 = time.time()
            if is_baseline:
                print(f"\n[{run_idx}/{total_runs}] Training Baseline | C={config.MAX_GRAD_NORM} | seed={seed} ...")
                model, meta = train_baseline(seed, device)
            else:
                print(f"\n[{run_idx}/{total_runs}] Training DP-SGD nm={nm:<4.2f} | C={config.MAX_GRAD_NORM} | seed={seed} ...")
                model, meta = train_dp(nm, seed, device)

            elapsed = time.time() - t0
            save_checkpoint(model, meta, ckpt_path)
            meta["checkpoint"] = ckpt_path
            meta["checkpoint_path"] = ckpt_path
            all_runs.append(meta)

            eps_val = meta.get("epsilon")
            eps_str = "inf" if is_baseline or eps_val == "inf" else f"{float(eps_val):.4f}"
            print(
                f"  Done in {elapsed:.1f}s | Acc={meta['test_accuracy']:.2f}% | "
                f"Gap={meta['generalization_gap']:+.4f} | eps={eps_str} -> {ckpt_file}"
            )
            print(f"  C={config.MAX_GRAD_NORM} | seed={seed} | elapsed={elapsed:.2f}s")

            if not is_baseline:
                epsilons_by_nm.setdefault(nm, []).append((seed, float(meta["epsilon"])))

    # ── EPSILON CONSISTENCY CHECK ────────────────────────────────
    print("\n" + "=" * 70)
    print("Checking Epsilon Consistency across Seeds...")
    for nm, seed_eps in epsilons_by_nm.items():
        if len(seed_eps) <= 1:
            continue
        base_seed, base_eps = seed_eps[0]
        for s, eps in seed_eps[1:]:
            diff = abs(eps - base_eps)
            assert diff < 1e-6, (
                f"FATAL: Epsilon consistency check FAILED for noise_multiplier={nm}! "
                f"Seed {base_seed} got ε={base_eps}, but seed {s} got ε={eps} "
                f"(diff={diff:.8e} > 1e-6). Aborting."
            )
        print(f"  nm={nm:<4.2f} -> ε={base_eps:.6f} identical across {len(seed_eps)} seeds [OK]")
    print("=" * 70 + "\n")

    return all_runs


# ── MIA Evaluation & Multi-Seed Aggregation ──────────────────────

def run_epsilon_mia(
    seeds: list[int],
    noise_multipliers: list[float] = NOISE_MULTIPLIERS,
    checkpoint_dir: str = CHECKPOINT_DIR,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Evaluate threshold MIA attack against every (noise_multiplier, seed) checkpoint.
    Strictly asserts that the loaded member index file corresponds to each checkpoint's seed.
    Aggregates across seeds with bootstrap 95% CIs and writes
    experiments/{dataset}/results/epsilon_sweep_multiseed.json.
    """
    device = get_device()
    cfg = _load_config()
    splits_dir = Path(getattr(config, "SPLITS_DIR", SPLITS_DIR))
    scores_dir = Path(results_dir) / "mia_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    n_train = getattr(config, "N_TRAIN", 5000)

    print("=" * 70)
    print(f"Evaluating Threshold MIA across DP-SGD Checkpoints [{config.DATASET.upper()}]")
    print("=" * 70)

    # 1. Pre-build data loaders per seed and strictly verify split matching
    loaders_by_seed = {}
    for seed in seeds:
        seed_json = splits_dir / f"member_indices_n{n_train}_seed{seed}.json"
        seed_npy = splits_dir / f"member_indices_n{n_train}_seed{seed}.npy"

        if seed_json.exists():
            member_file = str(seed_json)
        elif seed_npy.exists():
            member_file = str(seed_npy)
        elif seed == getattr(config, "SEED", 42) and (splits_dir / "member_indices.npy").exists():
            member_file = str(splits_dir / "member_indices.npy")
        else:
            raise FileNotFoundError(
                f"FATAL: No member index file found for seed {seed} under {splits_dir}!"
            )

        # STRICT ASSERTION
        assert f"seed{seed}" in Path(member_file).name, (
            f"FATAL: Member index file '{member_file}' does not match seed '{seed}'! "
            f"Evaluating against wrong seed indices produces false AUC near 0.50."
        )

        args = SimpleNamespace(
            data_root=cfg["DATA_ROOT"],
            member_index_file=member_file,
            n_samples=n_train,
            batch_size=getattr(config, "TEST_BATCH_SIZE", 1000),
            num_workers=0,
            seed=seed,
        )
        mem_ldr, non_ldr, n_actual = build_loaders(args, cfg)
        loaders_by_seed[seed] = (mem_ldr, non_ldr, n_actual, member_file)

    attack_results = []
    all_configs = [0.0] + list(noise_multipliers)

    for nm in all_configs:
        is_baseline = (nm == 0.0)
        for seed in seeds:
            ckpt_file = baseline_ckpt_name(seed) if is_baseline else dp_ckpt_name(nm, seed)
            ckpt_path = Path(checkpoint_dir) / ckpt_file

            if not ckpt_path.exists():
                print(f"[warn] Checkpoint missing: {ckpt_path}")
                continue

            # Load model
            model = load_model(ckpt_path, device)
            mem_loader, non_loader, n, loaded_split = loaders_by_seed[seed]

            # STRICT ASSERTION: verify loaded member index file corresponds to checkpoint's seed
            assert f"seed{seed}" in Path(loaded_split).name, (
                f"FATAL: Loaded split '{loaded_split}' does not match checkpoint seed '{seed}'!"
            )

            # Score dataset
            mem_scores = score_dataset(model, mem_loader, device)
            non_scores = score_dataset(model, non_loader, device, log_probs=mem_scores["_log_probs"])

            loss_atk = attack_metrics(mem_scores["loss"], non_scores["loss"])
            conf_atk = attack_metrics(mem_scores["confidence"], non_scores["confidence"])
            mentr_atk = attack_metrics(mem_scores["mentr"], non_scores["mentr"])

            gap = float(mem_scores["accuracy"] - non_scores["accuracy"])

            # Load checkpoint metadata
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            eps_val = ckpt_data.get("epsilon", "inf" if is_baseline else None)
            test_acc = ckpt_data.get("test_accuracy")
            test_loss = ckpt_data.get("test_loss")

            row = {
                "checkpoint": str(ckpt_path),
                "checkpoint_path": str(ckpt_path),
                "noise_multiplier": nm,
                "seed": seed,
                "epsilon": eps_val,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "member_accuracy": float(mem_scores["accuracy"]),
                "nonmember_accuracy": float(non_scores["accuracy"]),
                "generalization_gap": gap,
                "attack_auc": float(loss_atk["auc"]),
                "attack_accuracy": float(loss_atk["attack_accuracy"]),
                "tpr_at_1pct_fpr": float(loss_atk["tpr_at_fpr_0.01"]),
                "attacks": {
                    "loss": loss_atk,
                    "confidence": conf_atk,
                    "mentr": mentr_atk,
                },
            }
            attack_results.append(row)

            # Save per-sample score arrays for downstream bootstrap analysis
            np.savez_compressed(
                scores_dir / f"{ckpt_path.stem}.npz",
                member_loss=mem_scores["loss"],
                nonmember_loss=non_scores["loss"],
                member_confidence=mem_scores["confidence"],
                nonmember_confidence=non_scores["confidence"],
                member_mentr=mem_scores["mentr"],
                nonmember_mentr=non_scores["mentr"],
            )

            eps_display = "inf" if is_baseline or eps_val == "inf" else f"{float(eps_val):.2f}"
            print(
                f"nm={nm:<4.2f} | ε={eps_display:<7} | seed={seed} | Acc={test_acc:.2f}% | "
                f"Gap={gap:+.4f} | AUC={loss_atk['auc']:.4f} | TPR@1%FPR={loss_atk['tpr_at_fpr_0.01']:.4f}"
            )

    # ── Multi-Seed Aggregation ───────────────────────────────────
    summary = aggregate_epsilon_results(attack_results, scores_dir, seeds)

    multiseed_manifest_path = os.path.join(results_dir, "epsilon_sweep_multiseed.json")
    with open(multiseed_manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote multi-seed summary to {multiseed_manifest_path}")
    return summary


def aggregate_epsilon_results(
    attack_results: list[dict],
    scores_dir: Path | str,
    seeds: list[int],
    n_bootstraps: int = 1000,
) -> dict:
    """
    Produce summary keyed by noise_multiplier containing:
      - epsilon (scalar, not aggregated)
      - mean and std across seeds for:
          test_accuracy, test_loss, generalization_gap,
          attack_auc, attack_accuracy, tpr_at_1pct_fpr
      - per-seed raw values retained alongside the aggregates
      - bootstrap 95% CI for attack_auc, pooling score arrays across seeds (1000 resamples).
        Uses compute_bootstrap_auc_ci imported from src.sweep_clipping.
    """
    scores_dir = Path(scores_dir)
    grouped: dict[float, list[dict]] = {}
    for r in attack_results:
        nm = float(r["noise_multiplier"])
        grouped.setdefault(nm, []).append(r)

    summary = {}
    rng = np.random.default_rng(42)

    print("\n" + "=" * 104)
    print(f"DP-SGD MULTI-SEED EPSILON SWEEP SUMMARY [{config.DATASET.upper()}] (Seeds: {seeds})")
    print("=" * 104)
    print(f"  {'nm':>5} | {'ε':^10} | {'Test Acc (%)':^18} | {'Gen Gap':^16} | {'Attack AUC [95% CI]':^28} | {'TPR@1%FPR':^10}")
    print("-" * 104)

    for nm in sorted(grouped.keys()):
        runs = grouped[nm]
        is_baseline = (nm == 0.0)

        # Validate epsilon consistency across seeds
        if not is_baseline:
            eps_vals = [float(r["epsilon"]) for r in runs if r.get("epsilon") is not None]
            assert len(eps_vals) > 0, f"No epsilon values found for nm={nm}"
            for ev in eps_vals:
                assert abs(ev - eps_vals[0]) < 1e-6, (
                    f"FATAL: Epsilon inconsistency across seeds for nm={nm}: {ev} vs {eps_vals[0]}"
                )
            epsilon_scalar = eps_vals[0]
            eps_display = f"{epsilon_scalar:.4f}"
        else:
            epsilon_scalar = "inf"
            eps_display = "inf"

        accs = [r["test_accuracy"] for r in runs if r.get("test_accuracy") is not None]
        losses = [r["test_loss"] for r in runs if r.get("test_loss") is not None]
        gaps = [r["generalization_gap"] for r in runs]
        aucs = [r["attack_auc"] for r in runs]
        atk_accs = [r["attack_accuracy"] for r in runs]
        tprs = [r["tpr_at_1pct_fpr"] for r in runs]

        # Collect member and nonmember score arrays across seeds for bootstrap analysis
        seed_scores = []
        for r in runs:
            ckpt_stem = Path(r["checkpoint"]).stem
            score_file = scores_dir / f"{ckpt_stem}.npz"
            if score_file.exists():
                data = np.load(score_file)
                seed_scores.append({
                    "mem_loss": data["member_loss"],
                    "non_loss": data["nonmember_loss"],
                })

        if seed_scores:
            ci_lower, ci_upper = compute_bootstrap_auc_ci(
                seed_scores, n_bootstraps=n_bootstraps, rng=rng
            )
        else:
            mean_auc = float(np.mean(aucs)) if aucs else 0.5
            ci_lower = mean_auc
            ci_upper = mean_auc

        record = {
            "noise_multiplier": nm,
            "epsilon": epsilon_scalar,
            "n_seeds": len(runs),
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "test_loss_mean": float(np.mean(losses)),
            "test_loss_std": float(np.std(losses)),
            "generalization_gap_mean": float(np.mean(gaps)),
            "generalization_gap_std": float(np.std(gaps)),
            "attack_auc_mean": float(np.mean(aucs)),
            "attack_auc_std": float(np.std(aucs)),
            "attack_accuracy_mean": float(np.mean(atk_accs)),
            "attack_accuracy_std": float(np.std(atk_accs)),
            "tpr_at_1pct_fpr_mean": float(np.mean(tprs)),
            "tpr_at_1pct_fpr_std": float(np.std(tprs)),
            "attack_auc_ci_lower": ci_lower,
            "attack_auc_ci_upper": ci_upper,
            "distinguishable_from_random": bool(ci_lower > 0.50),
            "runs": runs,
        }
        summary[str(nm)] = record

        acc_str = f"{record['test_accuracy_mean']:.2f} ± {record['test_accuracy_std']:.2f}"
        gap_str = f"{record['generalization_gap_mean']:+.4f} ± {record['generalization_gap_std']:.4f}"
        auc_str = f"{record['attack_auc_mean']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
        tpr_str = f"{record['tpr_at_1pct_fpr_mean']:.4f}"

        print(f"  {nm:>5.2f} | {eps_display:^10} | {acc_str:^18} | {gap_str:^16} | {auc_str:^28} | {tpr_str:^10}")

    print("=" * 104 + "\n")
    return summary


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Objective 2: Multi-ε privacy sweep with multi-seed support"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[config.SEED],
        help="Random seeds to run. Default: [42] from config. Use '--seeds 42 43 44' for paper-ready results."
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training and proceed directly to MIA evaluation & aggregation"
    )
    parser.add_argument(
        "--skip-attack", action="store_true",
        help="Skip MIA attack and aggregation"
    )
    args = parser.parse_args()

    checkpoint_dir = getattr(config, "CHECKPOINT_DIR", CHECKPOINT_DIR)
    results_dir = getattr(config, "RESULTS_DIR", RESULTS_DIR)

    if not args.skip_training:
        run_epsilon_sweep(
            seeds=args.seeds,
            noise_multipliers=NOISE_MULTIPLIERS,
            checkpoint_dir=checkpoint_dir,
        )

    if not args.skip_attack:
        run_epsilon_mia(
            seeds=args.seeds,
            noise_multipliers=NOISE_MULTIPLIERS,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
        )


if __name__ == "__main__":
    main()

