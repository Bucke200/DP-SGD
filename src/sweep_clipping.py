"""
Objective: Multi-seed gradient clipping sweep isolating clipping from noise.

Sweeps max_grad_norm across [0.5, 1.0, 5.0, 10.0, 50.0] with noise_multiplier=0.0
at seeds [42, 43, 44] to evaluate differential privacy and membership inference
vulnerability under pure per-sample gradient clipping.

Usage:
    python -m src.sweep_clipping
    python -m src.sweep_clipping --seeds 42 43 44
"""

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
from src.utils import set_seed, get_device
from src.threshold_attack import (
    load_model,
    build_loaders,
    score_dataset,
    attack_metrics,
    roc_curve_np,
    auc_np,
    _load_config,
)

# ── Sweep Configuration ──────────────────────────────────────────────────────
CLIP_NORMS = [0.5, 1.0, 5.0, 10.0, 50.0]
SEEDS = [42, 43, 44]
NOISE_MULTIPLIER = 0.0

CHECKPOINT_DIR = getattr(config, "CHECKPOINT_DIR", "experiments/cifar10/checkpoints")
RESULTS_DIR = getattr(config, "RESULTS_DIR", "experiments/cifar10/results")
SPLITS_DIR = getattr(config, "SPLITS_DIR", "experiments/cifar10/splits")


def ckpt_name(clip_norm: float, seed: int) -> str:
    """Standard checkpoint naming: clip_{C:.2f}_nm_0.00_seed{seed}.pt"""
    return f"clip_{clip_norm:.2f}_nm_0.00_seed{seed}.pt"


def save_checkpoint(model, metadata: dict, path: str):
    """
    Save model weights and experimental metadata.
    Unwrap Opacus GradSampleModule via _module to ensure clean model loading.
    """
    if hasattr(model, "_module"):
        state_dict = model._module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save({"model_state_dict": state_dict, **metadata}, path)


# ── Training Routine ─────────────────────────────────────────────────────────

def train_clipping_run(
    clip_norm: float,
    seed: int,
    device: torch.device,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> tuple[nn.Module, dict]:
    """
    Train a CIFAR-10 model with per-sample gradient clipping and zero noise.
    Records per-sample gradient norm diagnostics at epoch 1 and epoch 50.
    """
    set_seed(seed)
    splits_dir = getattr(config, "SPLITS_DIR", SPLITS_DIR)
    n_train = getattr(config, "N_TRAIN", 5000)

    # 1. Member/non-member split identical to completed sweep
    train_loader, test_loader, member_indices = get_data_loaders(
        seed=seed,
        save_indices=True,
    )

    # 2. Strict assertion: loaded member index file matches sweep_epsilon for the same seed
    seed_json_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.json")
    seed_npy_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.npy")

    if os.path.exists(seed_json_path):
        with open(seed_json_path, "r") as f:
            saved_indices = np.array(json.load(f))
        assert np.array_equal(member_indices, saved_indices), (
            f"FATAL: Member indices for seed {seed} do not match {seed_json_path}!"
        )
    elif os.path.exists(seed_npy_path):
        saved_indices = np.load(seed_npy_path)
        assert np.array_equal(member_indices, saved_indices), (
            f"FATAL: Member indices for seed {seed} do not match {seed_npy_path}!"
        )

    # 3. Clean evaluation loader for training set to measure exact training accuracy & loss
    train_eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=getattr(config, "TEST_BATCH_SIZE", 1000),
        shuffle=False,
    )

    # 4. Model & Optimizer
    model = get_model(config.DATASET).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=getattr(config, "MOMENTUM", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Opacus make_private with noise_multiplier=0.0
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=clip_norm,
    )

    diagnostics = {}
    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        record_diag = (epoch == 1 or epoch == config.EPOCHS)
        epoch_norms = [] if record_diag else None

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if record_diag:
                # Per-sample gradient norms BEFORE clipping
                # Opacus exposes per-sample gradients on p.grad_sample (shape: batch_size, ...)
                batch_size = data.size(0)
                total_norm_sq = torch.zeros(batch_size, device=device)
                for p in model.parameters():
                    if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                        flat = p.grad_sample.view(batch_size, -1)
                        total_norm_sq += (flat ** 2).sum(dim=-1)
                norms = torch.sqrt(total_norm_sq).detach().cpu().numpy()
                epoch_norms.extend(norms.tolist())

            optimizer.step()

        if record_diag:
            norms_arr = np.array(epoch_norms, dtype=np.float64)
            epoch_key = f"epoch_{epoch}"
            diagnostics[epoch_key] = {
                "mean_norm": float(np.mean(norms_arr)),
                "median_norm": float(np.median(norms_arr)),
                "p95_norm": float(np.percentile(norms_arr, 95)),
                "percentile_95": float(np.percentile(norms_arr, 95)),
                "95th_percentile": float(np.percentile(norms_arr, 95)),
                "clipped_fraction": float(np.mean(norms_arr > clip_norm)),
                "max_grad_norm": clip_norm,
                "total_samples": len(norms_arr),
            }

    wall_clock_seconds = time.time() - start_time

    # 6. Final Evaluation
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    train_loss, train_acc = evaluate(model, train_eval_loader, device, criterion)
    generalization_gap = round((train_acc - test_acc) / 100.0, 6)

    ckpt_file = ckpt_name(clip_norm, seed)
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file)

    meta = {
        "clip_norm": clip_norm,
        "max_grad_norm": clip_norm,
        "seed": seed,
        "test_accuracy": round(test_acc, 2),
        "test_loss": round(test_loss, 4),
        "train_accuracy": round(train_acc, 2),
        "train_loss": round(train_loss, 4),
        "generalization_gap": generalization_gap,
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "checkpoint_path": ckpt_path,
        "checkpoint": ckpt_path,
        "clipping_diagnostics": diagnostics,
        "epsilon": None,
        "epsilon_note": "infinite (no noise)",
        "noise_multiplier": NOISE_MULTIPLIER,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "dataset": config.DATASET,
    }

    return model, meta


# ── Sweep Manager ────────────────────────────────────────────────────────────

def run_clipping_sweep(
    clip_norms: list[float] = CLIP_NORMS,
    seeds: list[int] = SEEDS,
    checkpoint_dir: str = CHECKPOINT_DIR,
    results_dir: str = RESULTS_DIR,
) -> list[dict]:
    """
    Run full multi-seed clipping sweep, skipping and reusing existing checkpoints.
    Persists results incrementally to results/clipping_sweep.json.
    """
    device = get_device()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    manifest_path = os.path.join(results_dir, "clipping_sweep.json")
    results_map: dict[tuple[float, int], dict] = {}

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                existing_data = json.load(f)
            entries = existing_data["results"] if isinstance(existing_data, dict) and "results" in existing_data else existing_data
            for entry in entries:
                key = (float(entry["clip_norm"]), int(entry["seed"]))
                results_map[key] = entry
            print(f"[info] Loaded {len(results_map)} existing run entries from {manifest_path}")
        except Exception as e:
            print(f"[warn] Could not parse existing manifest ({e}), starting fresh.")

    total_runs = len(clip_norms) * len(seeds)
    run_idx = 0

    print("=" * 70)
    print(f"DP-SGD Clipping Ablation Sweep [{config.DATASET.upper()}] (Noise multiplier = 0.0)")
    print(f"  Clip norms (C):  {clip_norms}")
    print(f"  Seeds:           {seeds}")
    print(f"  Device:          {device}")
    print(f"  Total runs:      {total_runs}")
    print("=" * 70)

    for clip_norm in clip_norms:
        for seed in seeds:
            run_idx += 1
            key = (float(clip_norm), int(seed))
            ckpt_file = ckpt_name(clip_norm, seed)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)

            # Check if checkpoint already exists on disk
            if os.path.exists(ckpt_path) and key in results_map:
                existing_meta = results_map[key]
                print(
                    f"[{run_idx}/{total_runs}] SKIP (already completed): C={clip_norm:<4.1f} | "
                    f"seed={seed} | Acc={existing_meta['test_accuracy']:.2f}% | "
                    f"Gap={existing_meta['generalization_gap']:+.4f} -> {ckpt_file}"
                )
                continue
            elif os.path.exists(ckpt_path):
                # Checkpoint on disk, try recovering metadata
                try:
                    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if "clipping_diagnostics" in ckpt_data and "test_accuracy" in ckpt_data:
                        meta = {k: v for k, v in ckpt_data.items() if k != "model_state_dict"}
                        meta["checkpoint_path"] = ckpt_path
                        meta["checkpoint"] = ckpt_path
                        results_map[key] = meta
                        print(
                            f"[{run_idx}/{total_runs}] REUSED from disk: C={clip_norm:<4.1f} | "
                            f"seed={seed} | Acc={meta['test_accuracy']:.2f}% -> {ckpt_file}"
                        )
                        continue
                except Exception:
                    pass

            print(f"\n[{run_idx}/{total_runs}] Training C={clip_norm:<4.1f} | seed={seed} ...")
            t0 = time.time()
            model, meta = train_clipping_run(
                clip_norm=clip_norm,
                seed=seed,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )
            elapsed = time.time() - t0

            # Save unwrapped checkpoint
            save_checkpoint(model, meta, ckpt_path)
            results_map[key] = meta

            # Persist manifest immediately for crash resilience
            all_entries = [results_map[k] for k in sorted(results_map.keys())]
            with open(manifest_path, "w") as f:
                json.dump(all_entries, f, indent=2)

            diag1 = meta["clipping_diagnostics"].get("epoch_1", {})
            diag50 = meta["clipping_diagnostics"].get("epoch_50", {})
            print(
                f"  Done in {elapsed:.1f}s | Test Acc: {meta['test_accuracy']:.2f}% | "
                f"Train Loss: {meta['train_loss']:.4f} | Gap: {meta['generalization_gap']:+.4f} | "
                f"Clipped: ep1={diag1.get('clipped_fraction', 0.0)*100:.1f}%, "
                f"ep50={diag50.get('clipped_fraction', 0.0)*100:.1f}%"
            )

    all_entries = [results_map[k] for k in sorted(results_map.keys())]
    with open(manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\nAll {len(all_entries)} clipping runs completed. Saved to {manifest_path}")
    return all_entries


# ── MIA Evaluation ───────────────────────────────────────────────────────────

def run_clipping_mia(
    clip_norms: list[float] = CLIP_NORMS,
    seeds: list[int] = SEEDS,
    checkpoint_dir: str = CHECKPOINT_DIR,
    results_dir: str = RESULTS_DIR,
    out_file: str | None = None,
) -> list[dict]:
    """
    Run threshold MIA evaluation against all clipping checkpoints.
    Imports and reuses existing threshold_attack routines.
    """
    device = get_device()
    cfg = _load_config()
    splits_dir = getattr(config, "SPLITS_DIR", SPLITS_DIR)
    scores_dir = Path(results_dir) / "mia_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if out_file is None:
        out_file = os.path.join(results_dir, "clipping_sweep_attack.json")

    print("\n" + "=" * 70)
    print("Evaluating Membership Inference Attack across Clipping Checkpoints")
    print("=" * 70)

    # Pre-build data loaders per seed for efficiency and exact split matching
    loaders_by_seed = {}
    for seed in seeds:
        seed_file = os.path.join(splits_dir, f"member_indices_n{config.N_TRAIN}_seed{seed}.json")
        if not os.path.exists(seed_file):
            seed_file = os.path.join(splits_dir, f"member_indices_n{config.N_TRAIN}_seed{seed}.npy")
        if not os.path.exists(seed_file):
            seed_file = os.path.join(splits_dir, "member_indices.npy")

        args = SimpleNamespace(
            data_root=cfg["DATA_ROOT"],
            member_index_file=seed_file,
            n_samples=config.N_TRAIN,
            batch_size=getattr(config, "TEST_BATCH_SIZE", 1000),
            num_workers=0,
            seed=seed,
        )
        mem_loader, non_loader, n = build_loaders(args, cfg)
        loaders_by_seed[seed] = (mem_loader, non_loader, n)

    attack_results = []
    for clip_norm in clip_norms:
        for seed in seeds:
            ckpt_path = Path(checkpoint_dir) / ckpt_name(clip_norm, seed)
            if not ckpt_path.exists():
                print(f"[warn] Checkpoint missing: {ckpt_path}")
                continue

            model = load_model(ckpt_path, device)
            mem_loader, non_loader, n = loaders_by_seed[seed]

            mem_scores = score_dataset(model, mem_loader, device)
            non_scores = score_dataset(model, non_loader, device, log_probs=mem_scores["_log_probs"])

            loss_atk = attack_metrics(mem_scores["loss"], non_scores["loss"])
            conf_atk = attack_metrics(mem_scores["confidence"], non_scores["confidence"])
            mentr_atk = attack_metrics(mem_scores["mentr"], non_scores["mentr"])

            gap = float(mem_scores["accuracy"] - non_scores["accuracy"])

            row = {
                "checkpoint": str(ckpt_path),
                "checkpoint_path": str(ckpt_path),
                "clip_norm": clip_norm,
                "max_grad_norm": clip_norm,
                "seed": seed,
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

            # If checkpoint metadata has test_accuracy, copy it
            try:
                ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                row["test_accuracy"] = ckpt_data.get("test_accuracy")
                row["test_loss"] = ckpt_data.get("test_loss")
            except Exception:
                pass

            attack_results.append(row)

            # Persist per-sample score arrays for downstream bootstrap analysis
            np.savez_compressed(
                scores_dir / f"{ckpt_path.stem}.npz",
                member_loss=mem_scores["loss"],
                nonmember_loss=non_scores["loss"],
                member_confidence=mem_scores["confidence"],
                nonmember_confidence=non_scores["confidence"],
                member_mentr=mem_scores["mentr"],
                nonmember_mentr=non_scores["mentr"],
            )

            print(
                f"C={clip_norm:<4.1f} | seed={seed} | Acc={row.get('test_accuracy', 0.0):.2f}% | "
                f"Gap={gap:+.4f} | AUC={loss_atk['auc']:.4f} | TPR@1%FPR={loss_atk['tpr_at_fpr_0.01']:.4f}"
            )

    # Compute summary aggregation
    summary = aggregate_clipping_results(attack_results, scores_dir)

    payload = {
        "config": {
            "dataset": config.DATASET,
            "clip_norms": clip_norms,
            "seeds": seeds,
            "noise_multiplier": NOISE_MULTIPLIER,
            "n_samples": config.N_TRAIN,
        },
        "results": attack_results,
        "summary": summary,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote attack evaluation and summary to {out_file}")

    return attack_results


# ── Aggregation & Bootstrap Analysis ─────────────────────────────────────────

def aggregate_clipping_results(
    attack_results: list[dict],
    scores_dir: Path | str,
    n_bootstraps: int = 1000,
) -> dict:
    """
    Produce summary keyed by clip_norm with mean and std for:
      - test_accuracy
      - generalization_gap
      - attack_auc
      - tpr_at_1pct_fpr
    Computes a 95% bootstrap CI for attack AUC over resampled score arrays.
    """
    scores_dir = Path(scores_dir)
    grouped: dict[float, list[dict]] = {}
    for r in attack_results:
        c = float(r["clip_norm"])
        grouped.setdefault(c, []).append(r)

    summary = {}
    rng = np.random.default_rng(42)

    print("\n" + "─" * 78)
    print("CLIPPING ABLATION SUMMARY ACROSS SEEDS (Noise σ = 0.0)")
    print("─" * 78)
    print(f"  {'C':>5} | {'Test Acc (%)':^15} | {'Gen Gap':^13} | {'Attack AUC (95% CI)':^25} | {'TPR@1%FPR':^10}")
    print("─" * 78)

    for c in sorted(grouped.keys()):
        runs = grouped[c]
        accs = [r["test_accuracy"] for r in runs if r.get("test_accuracy") is not None]
        gaps = [r["generalization_gap"] for r in runs]
        aucs = [r["attack_auc"] for r in runs]
        tprs = [r["tpr_at_1pct_fpr"] for r in runs]

        # Load member and non-member score arrays across seeds for bootstrap CI
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

        # Bootstrap 95% CI resampling score arrays across iterations
        if seed_scores:
            boot_aucs = []
            for _ in range(n_bootstraps):
                seed_boot_aucs = []
                for s in seed_scores:
                    m = s["mem_loss"]
                    n = s["non_loss"]
                    m_samp = rng.choice(m, size=len(m), replace=True)
                    n_samp = rng.choice(n, size=len(n), replace=True)
                    fpr, tpr, _ = roc_curve_np(
                        np.r_[np.ones(len(m_samp)), np.zeros(len(n_samp))],
                        np.r_[m_samp, n_samp],
                    )
                    seed_boot_aucs.append(auc_np(fpr, tpr))
                boot_aucs.append(float(np.mean(seed_boot_aucs)))

            ci_lower = float(np.percentile(boot_aucs, 2.5))
            ci_upper = float(np.percentile(boot_aucs, 97.5))
        else:
            ci_lower = float(np.mean(aucs))
            ci_upper = float(np.mean(aucs))

        record = {
            "clip_norm": c,
            "n_seeds": len(runs),
            "test_accuracy_mean": float(np.mean(accs)) if accs else None,
            "test_accuracy_std": float(np.std(accs)) if accs else None,
            "generalization_gap_mean": float(np.mean(gaps)),
            "generalization_gap_std": float(np.std(gaps)),
            "attack_auc_mean": float(np.mean(aucs)),
            "attack_auc_std": float(np.std(aucs)),
            "attack_auc_ci_lower": ci_lower,
            "attack_auc_ci_upper": ci_upper,
            "tpr_at_1pct_fpr_mean": float(np.mean(tprs)),
            "tpr_at_1pct_fpr_std": float(np.std(tprs)),
            "distinguishable_from_random": bool(ci_lower > 0.50),
        }
        summary[str(c)] = record

        acc_str = f"{record['test_accuracy_mean']:.2f} ± {record['test_accuracy_std']:.2f}" if accs else "N/A"
        gap_str = f"{record['generalization_gap_mean']:+.4f} ± {record['generalization_gap_std']:.4f}"
        auc_str = f"{record['attack_auc_mean']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
        tpr_str = f"{record['tpr_at_1pct_fpr_mean']:.4f}"
        print(f"  {c:>5.1f} | {acc_str:^15} | {gap_str:^13} | {auc_str:^25} | {tpr_str:^10}")

    print("─" * 78 + "\n")
    return summary


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep gradient clipping norm C with noise=0.0")
    parser.add_argument("--clip-norms", type=float, nargs="+", default=CLIP_NORMS,
                        help="List of clipping norms to sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Random seeds to evaluate")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training sweep and proceed directly to MIA evaluation")
    parser.add_argument("--skip-attack", action="store_true",
                        help="Skip MIA evaluation")
    args = parser.parse_args()

    if not args.skip_training:
        run_clipping_sweep(
            clip_norms=args.clip_norms,
            seeds=args.seeds,
        )

    if not args.skip_attack:
        run_clipping_mia(
            clip_norms=args.clip_norms,
            seeds=args.seeds,
        )


if __name__ == "__main__":
    main()
