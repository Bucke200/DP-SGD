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


def test_clipping_sweep_json_entries():
    """Verify that clipping_sweep.json exists and has one entry per (C, seed) pair."""
    manifest_path = os.path.join(config.RESULTS_DIR, "clipping_sweep.json")
    assert os.path.exists(manifest_path), f"clipping_sweep.json not found at {manifest_path}"
    with open(manifest_path, "r") as f:
        results = json.load(f)
    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    expected_clips = [0.5, 1.0, 5.0, 10.0, 50.0]
    expected_seeds = [42, 43, 44]
    expected_pairs = {(float(c), int(s)) for c in expected_clips for s in expected_seeds}

    found_pairs = set()
    for entry in results:
        c = float(entry["clip_norm"])
        s = int(entry["seed"])
        found_pairs.add((c, s))

    assert found_pairs == expected_pairs, (
        f"Missing/extra (C, seed) pairs. Missing: {expected_pairs - found_pairs}, "
        f"Extra: {found_pairs - expected_pairs}"
    )
    assert len(results) == len(expected_pairs)


def test_clipping_sweep_checkpoints_load():
    """Verify that every checkpoint path exists and loads into a clean CifarCNN."""
    manifest_path = os.path.join(config.RESULTS_DIR, "clipping_sweep.json")
    assert os.path.exists(manifest_path), f"clipping_sweep.json not found at {manifest_path}"
    with open(manifest_path, "r") as f:
        results = json.load(f)
    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    device = torch.device("cpu")
    for entry in results:
        ckpt_path = entry.get("checkpoint_path") or entry.get("checkpoint")
        assert ckpt_path is not None, f"No checkpoint key in entry: {entry}"
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(ckpt_path)
        assert os.path.exists(ckpt_path), f"Checkpoint does not exist: {ckpt_path}"

        model = CifarCNN().to(device)
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
        state_dict = {k.replace("_module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        dummy_x = torch.randn(2, 3, 32, 32)
        out = model(dummy_x)
        assert out.shape == (2, 10)


def test_clipping_sweep_epsilon_is_null():
    """Verify that epsilon is null for all clipping-sweep runs."""
    manifest_path = os.path.join(config.RESULTS_DIR, "clipping_sweep.json")
    assert os.path.exists(manifest_path), f"clipping_sweep.json not found at {manifest_path}"
    with open(manifest_path, "r") as f:
        results = json.load(f)
    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    for entry in results:
        assert entry.get("epsilon") is None, (
            f"Expected epsilon to be None, got {entry.get('epsilon')} for entry: {entry}"
        )
        assert "infinite" in str(entry.get("epsilon_note", "")).lower()


def test_clipping_sweep_member_indices_match_epsilon_sweep():
    """Verify that the member index set used by clipping sweep is identical to epsilon sweep at the same seed."""
    import numpy as np
    splits_dir = getattr(config, "SPLITS_DIR", "experiments/cifar10/splits")
    n_train = getattr(config, "N_TRAIN", 5000)

    for seed in [42, 43, 44]:
        _, _, clipping_indices = get_data_loaders(seed=seed, save_indices=False)

        json_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.json")
        npy_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.npy")

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                saved_indices = np.array(json.load(f))
            assert np.array_equal(clipping_indices, saved_indices), (
                f"Seed {seed} member indices do not match {json_path}"
            )
        elif os.path.exists(npy_path):
            saved_indices = np.load(npy_path)
            assert np.array_equal(clipping_indices, saved_indices), (
                f"Seed {seed} member indices do not match {npy_path}"
            )
        else:
            pytest.fail(f"No saved member index file found for seed {seed}")


def test_epsilon_sweep_multiseed_json_entries():
    """Verify that epsilon_sweep_multiseed.json exists and has an entry per (noise_multiplier, seed)."""
    manifest_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep_multiseed.json")
    if not os.path.exists(manifest_path):
        pytest.skip(f"epsilon_sweep_multiseed.json not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        data = json.load(f)

    expected_nms = [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0]
    expected_seeds = [42, 43, 44]
    expected_pairs = {(nm, s) for nm in expected_nms for s in expected_seeds}

    found_pairs = set()
    for nm_key, entry in data.items():
        nm = float(entry.get("noise_multiplier", nm_key))
        runs = entry.get("runs", [])
        for r in runs:
            s = int(r["seed"])
            found_pairs.add((nm, s))

    assert found_pairs == expected_pairs, (
        f"Missing/extra (noise_multiplier, seed) pairs. Missing: {expected_pairs - found_pairs}, "
        f"Extra: {found_pairs - expected_pairs}"
    )


def test_epsilon_sweep_multiseed_checkpoints_load():
    """Verify that every checkpoint path exists and loads into a clean CifarCNN."""
    manifest_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep_multiseed.json")
    if not os.path.exists(manifest_path):
        pytest.skip(f"epsilon_sweep_multiseed.json not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        data = json.load(f)

    device = torch.device("cpu")
    for nm_key, entry in data.items():
        runs = entry.get("runs", [])
        for r in runs:
            ckpt_path = r.get("checkpoint_path") or r.get("checkpoint")
            assert ckpt_path is not None, f"Missing checkpoint in {r}"
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.abspath(ckpt_path)
            assert os.path.exists(ckpt_path), f"Checkpoint does not exist: {ckpt_path}"

            model = CifarCNN().to(device)
            ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
            state_dict = {k.replace("_module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()

            dummy_x = torch.randn(2, 3, 32, 32)
            out = model(dummy_x)
            assert out.shape == (2, 10)


def test_epsilon_consistency_across_seeds():
    """Verify that epsilon is identical across seeds within each noise multiplier (tolerance 1e-6)."""
    manifest_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep_multiseed.json")
    if not os.path.exists(manifest_path):
        pytest.skip(f"epsilon_sweep_multiseed.json not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        data = json.load(f)

    for nm_key, entry in data.items():
        nm = float(entry.get("noise_multiplier", nm_key))
        if nm == 0.0:
            continue
        runs = entry.get("runs", [])
        epsilons = [float(r["epsilon"]) for r in runs if r.get("epsilon") is not None]
        assert len(epsilons) >= 2, f"Expected at least 2 seeds for nm={nm}, got {len(epsilons)}"
        for eps in epsilons:
            assert abs(eps - epsilons[0]) < 1e-6, (
                f"Epsilon discrepancy for nm={nm}: {eps} vs {epsilons[0]}"
            )


def test_distinct_member_index_splits_per_seed():
    """Verify each seed has a distinct member index file, and no two seeds share a split."""
    import numpy as np
    splits_dir = getattr(config, "SPLITS_DIR", "experiments/cifar10/splits")
    n_train = getattr(config, "N_TRAIN", 5000)
    seeds = [42, 43, 44]

    splits = {}
    for seed in seeds:
        json_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.json")
        npy_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.npy")
        assert os.path.exists(json_path) or os.path.exists(npy_path), (
            f"Missing member index file for seed {seed}"
        )
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                idx = np.array(json.load(f))
        else:
            idx = np.load(npy_path)
        assert len(idx) == n_train
        splits[seed] = set(idx.tolist())

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            s1, s2 = seeds[i], seeds[j]
            overlap = len(splits[s1] & splits[s2])
            assert splits[s1] != splits[s2], f"Seed {s1} and {s2} have identical splits!"
            # Subsampling 5000 from 50000 has expected overlap ~ 500, never 5000
            assert overlap < n_train, f"Seed {s1} and {s2} completely share a split!"


def test_seed42_values_match_original_single_seed_run():
    """Verify seed-42 values in multiseed JSON match the original single-seed run."""
    single_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep.json")
    multi_path = os.path.join(config.RESULTS_DIR, "epsilon_sweep_multiseed.json")
    assert os.path.exists(single_path), f"epsilon_sweep.json not found at {single_path}"
    if not os.path.exists(multi_path):
        pytest.skip(f"epsilon_sweep_multiseed.json not found at {multi_path}")

    with open(single_path, "r") as f:
        single_data = json.load(f)
    with open(multi_path, "r") as f:
        multi_data = json.load(f)

    single_by_nm = {}
    for r in single_data:
        nm = float(r["noise_multiplier"])
        single_by_nm[nm] = r

    for nm_key, entry in multi_data.items():
        nm = float(entry.get("noise_multiplier", nm_key))
        runs = entry.get("runs", [])
        run_42 = next((r for r in runs if r["seed"] == 42), None)
        assert run_42 is not None, f"Seed 42 not found for nm={nm}"
        orig = single_by_nm[nm]
        assert abs(run_42["test_accuracy"] - orig["test_accuracy"]) < 0.1, (
            f"Accuracy mismatch for nm={nm}, seed 42: {run_42['test_accuracy']} vs {orig['test_accuracy']}"
        )
        if nm > 0.0:
            assert abs(float(run_42["epsilon"]) - float(orig["epsilon"])) < 1e-4, (
                f"Epsilon mismatch for nm={nm}, seed 42: {run_42['epsilon']} vs {orig['epsilon']}"
            )


def test_seed42_baseline_regression_guard():
    """
    REGRESSION GUARD: The refactored loss signal must reproduce published numbers exactly.
    Asserts seed-42 non-private baseline gives attack_auc = 0.8535 and
    tpr_at_1pct_fpr = 0.0287 (tolerance 1e-4).
    """
    from pathlib import Path
    from types import SimpleNamespace
    from src.threshold_attack import (
        load_model,
        build_loaders,
        score_dataset,
        attack_metrics,
        _load_config,
    )
    from src.utils import get_device

    cfg = _load_config()
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = repo_root / "experiments" / "cifar10" / "checkpoints" / "baseline_seed42.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Baseline checkpoint not found at {ckpt_path}")

    member_file = repo_root / "experiments" / "cifar10" / "splits" / "member_indices_n5000_seed42.npy"
    if not member_file.exists():
        member_file = repo_root / "experiments" / "cifar10" / "splits" / "member_indices.npy"
    assert member_file.exists(), f"Member index file not found at {member_file}"

    device = get_device()
    args = SimpleNamespace(
        data_root=cfg["DATA_ROOT"],
        member_index_file=str(member_file),
        n_samples=5000,
        batch_size=getattr(config, "TEST_BATCH_SIZE", 1000),
        num_workers=0,
        seed=42,
    )
    m_ldr, nm_ldr, n = build_loaders(args, cfg)
    model = load_model(ckpt_path, device)
    mem = score_dataset(model, m_ldr, device)
    non = score_dataset(model, nm_ldr, device, log_probs=mem["_log_probs"])

    metrics = attack_metrics(mem["loss"], non["loss"])
    auc = metrics["auc"]
    tpr = metrics["tpr_at_fpr_0.01"]

    # Tolerance 1e-4 regression guard
    assert abs(auc - 0.8535) <= 1e-4, f"Regression guard failed: AUC {auc:.6f} != 0.8535 (tol 1e-4)"
    assert abs(tpr - 0.0287) <= 1e-4, f"Regression guard failed: TPR@1%FPR {tpr:.6f} != 0.0287 (tol 1e-4)"


def test_pluggable_scoring_functions_standalone():
    """Verify pluggable scoring functions take (logits, labels) and return finite 1D arrays."""
    import numpy as np
    import torch
    from src.threshold_attack import score_loss, score_confidence, score_mentr, SCORING_FUNCTIONS

    # Test with torch.Tensor
    logits = torch.randn(20, 10)
    labels = torch.randint(0, 10, (20,))

    for name, fn in SCORING_FUNCTIONS.items():
        scores = fn(logits, labels)
        assert isinstance(scores, np.ndarray), f"{name} should return numpy ndarray"
        assert scores.shape == (20,), f"{name} expected shape (20,), got {scores.shape}"
        assert np.all(np.isfinite(scores)), f"{name} produced non-finite values"

    # Test with np.ndarray
    logits_np = np.random.randn(15, 10).astype(np.float32)
    labels_np = np.random.randint(0, 10, size=15)

    for name, fn in [("loss", score_loss), ("confidence", score_confidence), ("mentr", score_mentr)]:
        scores = fn(logits_np, labels_np)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (15,)
        assert np.all(np.isfinite(scores))



