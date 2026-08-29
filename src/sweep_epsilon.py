"""
Objective 2: Multi-ε Sweep with Model Checkpointing

Trains the SampleCNN under multiple noise multipliers to produce
a collection of models at different privacy levels. Each model's
weights and full experimental metadata are saved as checkpoints
for downstream MIA evaluation (Objective 3).

Usage:
    # Single seed (quick validation)
    python -m src.sweep_epsilon

    # Multi-seed for paper-ready results
    python -m src.sweep_epsilon --seeds 42 123 256 512 1024
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine

from src.dataset import get_mnist_loaders
from src.model import SampleCNN
from src.evaluate import evaluate
from src.utils import seed_everything, get_device
import config

# ── Sweep configuration ─────────────────────────────────────────
# Lower noise_multiplier → higher ε (weaker privacy, better accuracy)
# Higher noise_multiplier → lower ε (stronger privacy, worse accuracy)
NOISE_MULTIPLIERS = [0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0]

CHECKPOINT_DIR = "experiments/checkpoints"
RESULTS_DIR = "experiments/results"


# ── Checkpoint I/O ───────────────────────────────────────────────

def save_checkpoint(model, metadata, path):
    """
    Save model weights and full experimental metadata.

    Opacus wraps the model inside GradSampleModule, so we unwrap
    via `_module` to get a state_dict that loads directly into a
    clean SampleCNN() — no Opacus dependency at inference time.
    """
    if hasattr(model, "_module"):
        state_dict = model._module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save({"model_state_dict": state_dict, **metadata}, path)


def load_checkpoint(path, device):
    """Load a checkpoint into a clean SampleCNN for verification."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SampleCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ── Training routines ────────────────────────────────────────────

def train_baseline(seed, device):
    """Train a standard SGD model (no privacy)."""
    seed_everything(seed)
    train_loader, test_loader = get_mnist_loaders(config.BATCH_SIZE)

    model = SampleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    start = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    test_loss, test_acc = evaluate(model, test_loader, device)

    metadata = {
        "model_type": "baseline",
        "epsilon": "inf",
        "delta": None,
        "noise_multiplier": 0.0,
        "max_grad_norm": None,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "seed": seed,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "training_time_sec": round(elapsed, 2),
    }
    return model, metadata


def train_dp(noise_multiplier, seed, device):
    """Train a DP-SGD model with a given noise multiplier."""
    seed_everything(seed)
    train_loader, test_loader = get_mnist_loaders(config.BATCH_SIZE)

    model = SampleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

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
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    # ε is an OUTPUT of the accountant — never hardcode it
    epsilon = privacy_engine.get_epsilon(delta=config.DELTA)

    test_loss, test_acc = evaluate(model, test_loader, device)

    metadata = {
        "model_type": "dp-sgd",
        "epsilon": round(epsilon, 6),
        "delta": config.DELTA,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "seed": seed,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "training_time_sec": round(elapsed, 2),
    }
    return model, metadata


# ── Filename helpers ─────────────────────────────────────────────

def baseline_ckpt_name(seed):
    """baseline_seed42.pt"""
    return f"baseline_seed{seed}.pt"


def dp_ckpt_name(noise_multiplier, seed):
    """
    dp_nm_1.10_seed42.pt

    ε is intentionally excluded from the filename — it's a derived
    quantity that would need rounding. The precise value lives in the
    checkpoint metadata and in epsilon_sweep.json.
    """
    return f"dp_nm_{noise_multiplier:.2f}_seed{seed}.pt"


# ── Main sweep ───────────────────────────────────────────────────

def run_sweep(seeds):
    device = get_device()
    all_results = []

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_runs = len(seeds) * (1 + len(NOISE_MULTIPLIERS))
    run_idx = 0

    for seed in seeds:
        # ── Baseline ──
        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] Baseline (seed={seed})...")
        model, meta = train_baseline(seed, device)

        ckpt_path = os.path.join(CHECKPOINT_DIR, baseline_ckpt_name(seed))
        save_checkpoint(model, meta, ckpt_path)
        meta["checkpoint"] = ckpt_path
        all_results.append(meta)
        print(f"  Acc={meta['test_accuracy']:.2f}%  "
              f"Time={meta['training_time_sec']}s  -> {ckpt_path}")

        # ── DP sweep ──
        for nm in NOISE_MULTIPLIERS:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] "
                  f"DP-SGD nm={nm} (seed={seed})...")
            model, meta = train_dp(nm, seed, device)

            ckpt_path = os.path.join(CHECKPOINT_DIR, dp_ckpt_name(nm, seed))
            save_checkpoint(model, meta, ckpt_path)
            meta["checkpoint"] = ckpt_path
            all_results.append(meta)
            print(f"  ε={meta['epsilon']:.4f}  "
                  f"Acc={meta['test_accuracy']:.2f}%  "
                  f"Time={meta['training_time_sec']}s  -> {ckpt_path}")

    # ── Save manifest ──
    manifest_path = os.path.join(RESULTS_DIR, "epsilon_sweep.json")
    with open(manifest_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Sweep complete: {len(all_results)} models trained and saved")
    print(f"  Seeds:       {seeds}")
    print(f"  Multipliers: {NOISE_MULTIPLIERS}")
    print(f"  Manifest:    {manifest_path}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Objective 2: Multi-ε privacy sweep"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[config.SEED],
        help="Random seeds to run. Default: single seed from config. "
             "Use multiple for paper-ready results (e.g. --seeds 42 123 256)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DP-SGD Epsilon Sweep")
    print(f"  Noise multipliers: {NOISE_MULTIPLIERS}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Runs per seed: 1 baseline + {len(NOISE_MULTIPLIERS)} DP")
    print(f"  Total runs: {len(args.seeds) * (1 + len(NOISE_MULTIPLIERS))}")
    print("=" * 60)

    run_sweep(args.seeds)


if __name__ == "__main__":
    main()
