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
    manifest_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep.json")

    # Fallback to legacy path if dataset-scoped file does not exist yet
    if not os.path.exists(manifest_path):
        legacy_manifest = "experiments/results/epsilon_sweep.json"
        if os.path.exists(legacy_manifest):
            manifest_path = legacy_manifest
        else:
            print(f"Manifest not found: {manifest_path}")
            print("Run the sweep first: python -m src.sweep_epsilon")
            sys.exit(1)

    with open(manifest_path) as f:
        results = json.load(f)

    device = get_device()
    _, test_loader, _ = get_data_loaders(batch_size=64)

    print(f"Verifying {len(results)} checkpoints [{config.DATASET.upper()}]...\n")

    passed = 0
    failed = 0

    for entry in results:
        ckpt_path = entry["checkpoint"]

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
                if result["epsilon"] == "inf"
                else f"ε={result['epsilon']:.4f}"
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
    print(f"Results: {passed} passed, {failed} failed, "
          f"{len(results)} total")

    if failed == 0:
        print("All checkpoints verified — ready for Objective 3 (MIA)")
    else:
        print("Some checkpoints failed — fix before proceeding")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
