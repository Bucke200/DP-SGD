"""
Ablation Study: Per-sample gradient clipping without noise (C=1.0, sigma=0.0).

Isolates the effect of gradient clipping from noise addition to determine
whether clipping alone mitigates memorization and MIA vulnerability on CIFAR-10.
"""

import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

import config
from src.utils import set_seed, get_device
from src.data_split import get_data_loaders
from src.model import get_model, validate_model_for_opacus
from src.evaluate import evaluate
from src.threshold_attack import score_dataset, attack_metrics, build_loaders, _load_config


def run_clipping_ablation(
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 0.0,
    seed: int = config.SEED,
):
    print("=" * 65)
    print("STARTING ABLATION: Per-sample Clipping Only (No Noise)")
    print(f"  Dataset:          {config.DATASET}")
    print(f"  Max Grad Norm:    {max_grad_norm}")
    print(f"  Noise Multiplier: {noise_multiplier}")
    print(f"  Epochs:           {config.EPOCHS}")
    print(f"  Batch Size:       {config.BATCH_SIZE}")
    print(f"  Learning Rate:    {config.LEARNING_RATE}")
    print(f"  Seed:             {seed}")
    print("=" * 65)

    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # 1. Dataset loading (subsampled for MIA signal)
    train_loader, test_loader, member_indices = get_data_loaders(seed=seed)
    print(f"Dataset loaded: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples.")

    # 2. Model initialization & Opacus compatibility validation
    model = get_model().to(device)
    is_valid, errors = validate_model_for_opacus(model)
    print(f"Opacus ModuleValidator check: is_valid={is_valid}, errors={errors}")

    # 3. Criterion & Optimizer (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    # 4. Attach Opacus PrivacyEngine (C=1.0, sigma=0.0)
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    start_time = time.time()

    # 5. Training Loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        epoch_loss = running_loss / total_samples
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        if epoch % 5 == 0 or epoch == 1 or epoch == config.EPOCHS:
            print(
                f"Epoch [{epoch:02d}/{config.EPOCHS}] - "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Epsilon: inf (no noise)"
            )

    elapsed = time.time() - start_time
    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)

    # Save checkpoint
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_name = f"ablation_clip_{max_grad_norm:.2f}_nm_{noise_multiplier:.2f}_seed{seed}.pt"
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)

    state_dict = model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
    meta = {
        "model_type": "ablation-clip-only",
        "dataset": config.DATASET,
        "epsilon": "inf",
        "delta": None,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "seed": seed,
        "test_accuracy": final_test_acc,
        "test_loss": final_test_loss,
        "final_train_loss": epoch_loss,
        "training_time_sec": round(elapsed, 2),
        "checkpoint": ckpt_path,
    }
    torch.save({"model_state_dict": state_dict, **meta}, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # 6. Evaluate Membership Inference Attack against this ablation model
    print("\n" + "=" * 65)
    print("EVALUATING MEMBERSHIP INFERENCE ATTACK ON ABLATION MODEL")
    print("=" * 65)

    cfg = _load_config()
    attack_model = get_model().to(device)
    attack_model.load_state_dict(state_dict)
    attack_model.eval()

    from types import SimpleNamespace
    args = SimpleNamespace(
        data_root=cfg["DATA_ROOT"],
        member_index_file=os.path.join(config.SPLITS_DIR, "member_indices.npy"),
        n_samples=5000,
        batch_size=256,
        num_workers=0,
        seed=seed,
    )

    member_loader, nonmember_loader, n = build_loaders(args, cfg)
    mem_scores = score_dataset(attack_model, member_loader, device)
    non_scores = score_dataset(attack_model, nonmember_loader, device, log_probs=mem_scores["_log_probs"])

    gap = mem_scores["accuracy"] - non_scores["accuracy"]
    loss_atk = attack_metrics(mem_scores["loss"], non_scores["loss"])
    conf_atk = attack_metrics(mem_scores["confidence"], non_scores["confidence"])
    mentr_atk = attack_metrics(mem_scores["mentr"], non_scores["mentr"])

    ablation_results = {
        **meta,
        "member_accuracy": mem_scores["accuracy"],
        "nonmember_accuracy": non_scores["accuracy"],
        "generalization_gap": gap,
        "attacks": {
            "loss": loss_atk,
            "confidence": conf_atk,
            "mentr": mentr_atk,
        }
    }

    # Save ablation result json
    res_path = os.path.join(config.RESULTS_DIR, "ablation_clip_only.json")
    with open(res_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "─" * 65)
    print("ABLATION RESULTS SUMMARY (Clipping C=1.0, Noise σ=0.0):")
    print(f"  Test Accuracy:        {final_test_acc:.2f}%")
    print(f"  Final Train Loss:     {epoch_loss:.4f}")
    print(f"  Generalization Gap:   {gap:+.4f} ({gap*100:+.2f}%)")
    print(f"  MIA Loss Attack AUC:  {loss_atk['auc']:.4f}")
    print(f"  MIA Attack Accuracy:  {loss_atk['attack_accuracy']*100:.2f}%")
    print(f"  TPR @ 1% FPR:         {loss_atk['tpr_at_fpr_0.01']:.4f}")
    print(f"  TPR @ 0.1% FPR:       {loss_atk['tpr_at_fpr_0.001']:.4f}")
    print(f"  Training Time:        {elapsed:.2f}s")
    print("─" * 65)

    return ablation_results


if __name__ == "__main__":
    run_clipping_ablation()
