"""
Verify all saved checkpoints load correctly into a clean model instance.

This is Step 5 of the Objective 2 workflow — run this BEFORE moving
to Objective 3 (MIA) to confirm every checkpoint is usable.

Usage:
    python -m src.verify_checkpoints
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

import config
from src.data_split import get_data_loaders
from src.model import get_model
from src.evaluate import evaluate
from src.utils import get_device


def verify_single_checkpoint(ckpt_path, test_loader, device):
    """
    Load a checkpoint into a clean model instance, run evaluation,
    and compare against the stored metadata.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = get_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Forward pass sanity check
    test_loss, test_acc = evaluate(model, test_loader, device)

    # Compare with stored values (small tolerance for float rounding)
    stored_acc = ckpt["test_accuracy"]
    acc_diff = abs(test_acc - stored_acc)

    return {
        "path": ckpt_path,
        "stored_accuracy": stored_acc,
        "verified_accuracy": test_acc,
        "accuracy_match": acc_diff < 0.1,
        "acc_diff": round(acc_diff, 4),
        "epsilon": ckpt.get("epsilon", "N/A"),
        "noise_multiplier": ckpt.get("noise_multiplier", "N/A"),
        "seed": ckpt.get("seed", "N/A"),
        "has_full_metadata": all(
            k in ckpt for k in [
                "epsilon", "delta", "noise_multiplier", "max_grad_norm",
                "epochs", "batch_size", "learning_rate", "seed",
                "test_accuracy", "test_loss",
            ]
        ),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify saved model checkpoints")
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to manifest json (defaults to epsilon_sweep_multiseed.json if present, else epsilon_sweep.json)"
    )
    args = parser.parse_args()

    if args.manifest:
        manifest_path = args.manifest
    else:
        multiseed_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep_multiseed.json")
        single_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep.json")
        if os.path.exists(multiseed_path):
            manifest_path = multiseed_path
        elif os.path.exists(single_path):
            manifest_path = single_path
        elif os.path.exists("experiments/results/epsilon_sweep.json"):
            manifest_path = "experiments/results/epsilon_sweep.json"
        else:
            manifest_path = multiseed_path

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        print("Run the sweep first: python -m src.sweep_epsilon")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, dict):
        results = []
        for nm_key, val in data.items():
            if isinstance(val, dict) and "runs" in val:
                results.extend(val["runs"])
            elif isinstance(val, dict):
                results.append(val)
    else:
        results = data

    device = get_device()
    _, test_loader, _ = get_data_loaders(batch_size=64, save_indices=False)

    print(f"Verifying {len(results)} checkpoints [{config.DATASET.upper()}] from {manifest_path}...\n")

    passed = 0
    failed = 0

    for entry in results:
        ckpt_path = entry.get("checkpoint_path") or entry.get("checkpoint")

        if not os.path.exists(ckpt_path):
            candidate = os.path.join(config.CHECKPOINT_DIR, os.path.basename(ckpt_path))
            if os.path.exists(candidate):
                ckpt_path = candidate
            else:
                print(f"  MISSING  {ckpt_path}")
                failed += 1
                continue

        try:
            result = verify_single_checkpoint(ckpt_path, test_loader, device)

            status = "OK" if result["accuracy_match"] else "MISMATCH"
            meta_status = "full" if result["has_full_metadata"] else "partial"

            if result["accuracy_match"]:
                passed += 1
            else:
                failed += 1

            eps_str = (
                "baseline"
                if result["epsilon"] == "inf" or result["epsilon"] is None
                else f"ε={float(result['epsilon']):.4f}"
            )

            print(f"  {status:>8}  {eps_str:<16}  "
                  f"acc={result['verified_accuracy']:.2f}%  "
                  f"meta={meta_status}  "
                  f"seed={result['seed']}  "
                  f"-> {os.path.basename(ckpt_path)}")

        except Exception as e:
            print(f"  FAILED   {ckpt_path}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(results)} total")

    if failed == 0:
        print("All checkpoints verified successfully!")
    else:
        print("Some checkpoints failed — fix before proceeding")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
