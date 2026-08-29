import time
import torch
import torch.nn as nn
import torch.optim as optim

import config
from src.utils import set_seed, get_device
from src.dataset import get_mnist_dataloaders
from src.model import SampleCNN, validate_model_for_opacus
from src.evaluate import evaluate


def run_baseline_training():
    """Run baseline standard PyTorch SGD training on MNIST."""
    print("=" * 60)
    print("STARTING BASELINE TRAINING (Non-DP SGD)")
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

    # 3. Criterion & Optimizer (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    start_time = time.time()

    # 4. Training Loop
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
        print(f"Epoch [{epoch}/{config.EPOCHS}] - Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    elapsed = time.time() - start_time
    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)

    print("\nBaseline Training Complete")
    print("--------------------------")
    print(f"Dataset: MNIST")
    print(f"Model: SampleCNN")
    print(f"Optimizer: SGD")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print("=" * 60)

    return {
        "epochs": config.EPOCHS,
        "final_train_loss": epoch_loss,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "elapsed_seconds": elapsed
    }


if __name__ == "__main__":
    run_baseline_training()
