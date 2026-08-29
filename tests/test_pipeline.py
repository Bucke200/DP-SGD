import json
import os
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

import config
from src.utils import set_seed, get_device
from src.model import SampleCNN, validate_model_for_opacus
from src.evaluate import evaluate


def test_dataset_loading():
    """Verify synthetic dataset shapes and DataLoader batch iteration."""
    # Fast test with small synthetic data matching MNIST dimensions (N, 1, 28, 28)
    dummy_x = torch.randn(20, 1, 28, 28)
    dummy_y = torch.randint(0, 10, (20,))
    ds = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(ds, batch_size=5)

    assert len(loader) == 4
    for x, y in loader:
        assert x.shape == (5, 1, 28, 28)
        assert y.shape == (5,)
        break


def test_cnn_forward_pass():
    """Verify CNN forward pass produces logits of shape (batch_size, 10)."""
    model = SampleCNN()
    model.eval()
    dummy_input = torch.randn(8, 1, 28, 28)
    out = model(dummy_input)

    assert out.shape == (8, 10), f"Expected shape (8, 10), got {out.shape}"


def test_opacus_module_validator():
    """Verify Opacus ModuleValidator check on SampleCNN."""
    model = SampleCNN()
    is_valid, errors = validate_model_for_opacus(model)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)


def test_dp_setup_and_accounting():
    """Verify Opacus attaches to model/optimizer/dataloader and computes finite epsilon."""
    set_seed(42)
    device = get_device()

    model = SampleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    dummy_x = torch.randn(32, 1, 28, 28)
    dummy_y = torch.randint(0, 10, (32,))
    ds = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(ds, batch_size=8)

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )

    # Perform one step
    criterion = nn.CrossEntropyLoss()
    for x, y in dp_loader:
        x, y = x.to(device), y.to(device)
        dp_optimizer.zero_grad()
        out = dp_model(x)
        loss = criterion(out, y)
        loss.backward()
        dp_optimizer.step()
        break

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    assert isinstance(epsilon, float)
    assert epsilon > 0.0
    assert not torch.isnan(torch.tensor(epsilon))
    assert not torch.isinf(torch.tensor(epsilon))


def test_end_to_end_dp_subset():
    """Verify fast end-to-end training and evaluation on a tiny synthetic subset."""
    set_seed(42)
    device = get_device()

    model = SampleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dummy_x = torch.randn(64, 1, 28, 28)
    dummy_y = torch.randint(0, 10, (64,))
    train_ds = TensorDataset(dummy_x, dummy_y)
    train_loader = DataLoader(train_ds, batch_size=16)

    test_ds = TensorDataset(dummy_x[:16], dummy_y[:16])
    test_loader = DataLoader(test_ds, batch_size=16)

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )

    # 1 epoch training
    dp_model.train()
    for x, y in dp_loader:
        x, y = x.to(device), y.to(device)
        dp_optimizer.zero_grad()
        out = dp_model(x)
        loss = criterion(out, y)
        loss.backward()
        dp_optimizer.step()

    test_loss, test_acc = evaluate(dp_model, device, test_loader, criterion)
    epsilon = privacy_engine.get_epsilon(delta=1e-5)

    assert test_loss >= 0.0
    assert 0.0 <= test_acc <= 100.0
    assert epsilon > 0.0


def test_verify_checkpoints_import():
    """Ensure src.verify_checkpoints can be imported cleanly and has expected functions."""
    import src.verify_checkpoints
    assert hasattr(src.verify_checkpoints, "verify_single_checkpoint")
    assert hasattr(src.verify_checkpoints, "main")


def test_epsilon_sweep_json_exists_and_formatted():
    """Verify that experiments/results/epsilon_sweep.json exists and is formatted correctly."""
    manifest_path = os.path.join("experiments", "results", "epsilon_sweep.json")
    assert os.path.exists(manifest_path), f"Manifest file not found: {manifest_path}"
    with open(manifest_path, "r") as f:
        results = json.load(f)

    assert isinstance(results, list), "Manifest must contain a JSON list of run results"
    assert len(results) > 0, "Manifest should not be empty"

    required_fields = [
        "model_type", "epsilon", "delta", "noise_multiplier", "max_grad_norm",
        "epochs", "batch_size", "learning_rate", "seed", "test_accuracy",
        "test_loss", "training_time_sec", "checkpoint"
    ]
    for run in results:
        assert isinstance(run, dict), "Each entry in manifest must be a dictionary"
        for field in required_fields:
            assert field in run, f"Missing required field '{field}' in manifest entry: {run}"

