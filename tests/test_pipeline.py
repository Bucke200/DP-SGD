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
from src.model import SampleCNN, CifarCNN, get_model, validate_model_for_opacus
from src.evaluate import evaluate
from src.data_split import (
    get_data_loaders,
    get_split_loaders,
    get_cifar10_split_loaders,
    _cifar10_transform,
    _mnist_transform,
)


def test_config_paths():
    """Verify that dataset-scoped config paths match config.OUTPUT_DIR."""
    assert config.OUTPUT_DIR.startswith("experiments/")
    assert config.CHECKPOINT_DIR == f"{config.OUTPUT_DIR}/checkpoints"
    assert config.SPLITS_DIR == f"{config.OUTPUT_DIR}/splits"
    assert config.RESULTS_DIR == f"{config.OUTPUT_DIR}/results"


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


def test_data_transforms():
    """Verify MNIST and CIFAR-10 transform definitions."""
    mnist_tfm = _mnist_transform()
    cifar_tfm = _cifar10_transform()
    assert len(mnist_tfm.transforms) == 2
    assert len(cifar_tfm.transforms) == 2


def test_get_data_loaders_dispatch_invalid():
    """Verify dispatcher raises ValueError on unsupported dataset."""
    with pytest.raises(ValueError):
        get_data_loaders(dataset="invalid_dataset")


def test_cnn_forward_pass():
    """Verify CNN forward pass produces logits of shape (batch_size, 10)."""
    model = SampleCNN()
    model.eval()
    dummy_input = torch.randn(8, 1, 28, 28)
    out = model(dummy_input)

    assert out.shape == (8, 10), f"Expected shape (8, 10), got {out.shape}"


def test_cifar_cnn_forward_pass():
    """Verify CifarCNN forward pass produces logits of shape (batch_size, 10)."""
    model = CifarCNN()
    model.eval()
    dummy_input = torch.randn(8, 3, 32, 32)
    out = model(dummy_input)

    assert out.shape == (8, 10), f"Expected shape (8, 10), got {out.shape}"


def test_get_model_helper():
    """Verify get_model returns the expected architecture for each dataset."""
    m_mnist = get_model("mnist")
    assert isinstance(m_mnist, SampleCNN)

    m_cifar = get_model("cifar10")
    assert isinstance(m_cifar, CifarCNN)

    m_default = get_model()
    expected_cls = CifarCNN if getattr(config, "DATASET", "cifar10") == "cifar10" else SampleCNN
    assert isinstance(m_default, expected_cls)


def test_opacus_module_validator():
    """Verify Opacus ModuleValidator check on SampleCNN."""
    model = SampleCNN()
    is_valid, errors = validate_model_for_opacus(model)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
    assert is_valid is True
    assert len(errors) == 0


def test_cifar_cnn_opacus_module_validator():
    """Verify CifarCNN passes Opacus ModuleValidator.validate without errors."""
    cifar_model = CifarCNN()
    is_valid_cifar, errors_cifar = validate_model_for_opacus(cifar_model)
    assert isinstance(is_valid_cifar, bool)
    assert isinstance(errors_cifar, list)
    assert is_valid_cifar is True
    assert len(errors_cifar) == 0


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


def test_cifar_cnn_dp_setup():
    """Verify Opacus attaches to CifarCNN and performs DP-SGD step on (3, 32, 32) inputs."""
    set_seed(42)
    device = get_device()

    model = CifarCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    dummy_x = torch.randn(32, 3, 32, 32)
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

    criterion = nn.CrossEntropyLoss()
    for x, y in dp_loader:
        x, y = x.to(device), y.to(device)
        dp_optimizer.zero_grad()
        out = dp_model(x)
        loss = criterion(out, y)
        loss.backward()
        dp_optimizer.step()
        break

    epsilon = privacy_engine.get_epsilon(delta=config.DELTA)
    assert isinstance(epsilon, float)
    assert epsilon > 0.0


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


@pytest.mark.skipif(
    not os.path.exists(os.path.join(config.RESULTS_DIR, "epsilon_sweep.json")),
    reason="Sweep not yet run for current dataset — rerun with: python -m src.sweep_epsilon",
)
def test_epsilon_sweep_json_exists_and_formatted():
    """Verify that epsilon_sweep.json exists and is formatted correctly."""
    manifest_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep.json")
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


def test_mia_roc_and_attack_metrics():
    """Verify ROC curve, AUC, and balanced attack metrics calculations."""
    import numpy as np
    from src.threshold_attack import attack_metrics, roc_curve_np, tpr_at_fpr

    # Perfect separation
    mem = np.array([10.0, 9.0, 8.0, 7.0])
    non = np.array([3.0, 2.0, 1.0, 0.0])
    metrics = attack_metrics(mem, non)
    assert metrics["auc"] == 1.0
    assert metrics["attack_accuracy"] == 1.0
    assert metrics["tpr_at_fpr_0.01"] == 1.0

    # Random guess separation
    mem_rand = np.array([1.0, 1.0, 1.0, 1.0])
    non_rand = np.array([1.0, 1.0, 1.0, 1.0])
    rand_metrics = attack_metrics(mem_rand, non_rand)
    assert rand_metrics["auc"] == 0.5
    assert rand_metrics["attack_accuracy"] == 0.5


def test_mia_score_dataset_synthetic():
    """Verify score_dataset calculates loss, confidence, and modified entropy signals."""
    import numpy as np
    from src.threshold_attack import score_dataset

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    model.eval()

    dummy_x = torch.randn(10, 4)
    dummy_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    ds = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(ds, batch_size=4)

    scores = score_dataset(model, loader, torch.device("cpu"))
    assert "loss" in scores and "confidence" in scores and "mentr" in scores
    assert len(scores["loss"]) == 10
    assert 0.0 <= scores["accuracy"] <= 1.0
    assert np.all(np.isfinite(scores["loss"]))
    assert np.all(np.isfinite(scores["confidence"]))
    assert np.all(np.isfinite(scores["mentr"]))


def test_mia_aggregation():
    """Verify multi-seed / multi-configuration grouping for MIA plotting."""
    from src.plot_mia_curves import aggregate

    rows = [
        {
            "noise_multiplier": 0.0,
            "epsilon": None,
            "test_accuracy": 99.0,
            "attacks": {"loss": {"auc": 0.52}},
        },
        {
            "noise_multiplier": 1.1,
            "epsilon": 0.3,
            "test_accuracy": 90.0,
            "attacks": {"loss": {"auc": 0.50}},
        },
        {
            "noise_multiplier": 1.1,
            "epsilon": 0.3,
            "test_accuracy": 91.0,
            "attacks": {"loss": {"auc": 0.51}},
        },
    ]

    baseline, dp = aggregate(rows, "loss")
    assert baseline is not None
    assert baseline["auc_mean"] == pytest.approx(0.52)
    assert len(dp) == 1
    assert dp[0]["epsilon"] == 0.3
    assert dp[0]["auc_mean"] == pytest.approx(0.505)
    assert dp[0]["acc_mean"] == pytest.approx(90.5)
