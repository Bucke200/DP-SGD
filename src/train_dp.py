import time
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import config
from src.utils import set_seed, get_device
from src.dataset import get_mnist_dataloaders
from src.model import SampleCNN, validate_model_for_opacus
from src.evaluate import evaluate


def run_dp_training():
    """Run Differentially Private SGD training using Opacus PrivacyEngine on MNIST."""
    print("=" * 60)
    print("STARTING DP-SGD TRAINING (Opacus PrivacyEngine)")
    print("=" * 60)

    set_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 1. Dataset loading
    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        test_batch_size=config.TEST_BATCH_SIZE
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples.")

    # 2. Model initialization & Opacus compatibility validation
    model = SampleCNN().to(device)
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
            f"ε (delta={config.DELTA}): {epsilon:.2f}"
        )

    elapsed = time.time() - start_time
    final_epsilon = privacy_engine.get_epsilon(delta=config.DELTA)
    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)

    print("\nDP Training Complete")
    print("--------------------")
    print(f"Dataset: MNIST")
    print(f"Model: SampleCNN")
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
