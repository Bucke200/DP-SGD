import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import config
from src.utils import set_seed, get_device
from src.data_split import get_data_loaders
from src.model import get_model, validate_model_for_opacus
from src.evaluate import evaluate


def run_dp_training():
    """Run Differentially Private SGD training using Opacus PrivacyEngine."""
    print("=" * 60)
    print("STARTING DP-SGD TRAINING (Opacus PrivacyEngine)")
    print("=" * 60)

    set_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 1. Dataset loading (subsampled for MIA signal)
    train_loader, test_loader, member_indices = get_data_loaders()
    print(f"Dataset loaded: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples.")

    # 2. Model initialization & Opacus compatibility validation
    model = get_model().to(device)
    is_valid, errors = validate_model_for_opacus(model)
    print(f"Opacus ModuleValidator check: is_valid={is_valid}, errors={errors}")
    if not is_valid:
        print("Model has invalid layers for Opacus. Fixing model with ModuleValidator.fix...")
        model = ModuleValidator.fix(model).to(device)
        is_valid_after, errors_after = validate_model_for_opacus(model)
        print(f"Post-fix ModuleValidator check: is_valid={is_valid_after}, errors={errors_after}")

    # 3. Criterion & Optimizer (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    # 4. Attach Opacus PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=config.NOISE_MULTIPLIER,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    # Sanity checks on Opacus attachment
    model_type_str = str(type(model))
    optimizer_type_str = str(type(optimizer))
    loader_type_str = str(type(train_loader))

    print("\nOpacus Attachment Inspection:")
    print(f"  Model wrapper: {model_type_str}")
    print(f"  Optimizer wrapper: {optimizer_type_str}")
    print(f"  DataLoader wrapper: {loader_type_str}")
    print(f"  Noise multiplier: {optimizer.noise_multiplier if hasattr(optimizer, 'noise_multiplier') else config.NOISE_MULTIPLIER}")
    print(f"  Max grad norm: {optimizer.max_grad_norm if hasattr(optimizer, 'max_grad_norm') else config.MAX_GRAD_NORM}")

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
        epsilon = privacy_engine.get_epsilon(delta=config.DELTA)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        print(
            f"Epoch [{epoch}/{config.EPOCHS}] - Train Loss: {epoch_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
            f"eps (delta={config.DELTA}): {epsilon:.2f}"
        )

    elapsed = time.time() - start_time
    final_epsilon = privacy_engine.get_epsilon(delta=config.DELTA)
    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)

    print("\nDP Training Complete")
    print("--------------------")
    print(f"Dataset: {config.DATASET}")
    print(f"Model: {model._module.__class__.__name__ if hasattr(model, '_module') else model.__class__.__name__}")
    print(f"Optimizer: DP-SGD (Opacus)")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Noise Multiplier: {config.NOISE_MULTIPLIER}")
    print(f"Max Grad Norm: {config.MAX_GRAD_NORM}")
    print(f"Delta: {config.DELTA}")
    print(f"Epsilon: {final_epsilon:.4f}")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    print(f"Total Time: {elapsed:.2f}s")
    print("=" * 60)

    return {
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "noise_multiplier": config.NOISE_MULTIPLIER,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "delta": config.DELTA,
        "epsilon": final_epsilon,
        "final_train_loss": epoch_loss,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "elapsed_seconds": elapsed,
        "model_wrapper": model_type_str,
        "optimizer_wrapper": optimizer_type_str
    }


if __name__ == "__main__":
    run_dp_training()
