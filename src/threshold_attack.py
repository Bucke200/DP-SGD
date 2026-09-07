"""
Objective 3 / Phase A — Threshold (loss-based) membership inference attack.

Runs a per-sample scoring attack against the baseline model and every DP-SGD
checkpoint produced by `src/sweep_epsilon.py`, and writes AUC-ROC plus low-FPR
metrics for each epsilon to `threshold_attack.json`.

Assumptions
-----------
* `epsilon_sweep.json` exists and each entry has the keys
  `epsilon`, `noise_multiplier`, `test_accuracy`, `checkpoint` (optionally `seed`).
* Checkpoints load into a raw model instance (Opacus `_module.` prefixes already
  stripped by `verify_checkpoints.py`; stripped again here defensively).
* Members are drawn from the dataset *train* split, non-members from the *test*
  split, using the same transform the models were trained with. If your sweep
  trained on a subset of the train split, pass `--member-index-file` pointing at
  a .npy/.json of the indices that were actually used, otherwise some "members"
  will really be non-members and every AUC will be biased toward 0.5.

Usage
-----
    python -m src.threshold_attack
    python -m src.threshold_attack --n-samples 10000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# Project imports
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from src.model import get_model


def _load_config() -> dict:
    """Pull configuration parameters."""
    dataset = getattr(config, "DATASET", "cifar10")
    defaults = {
        "SEED": getattr(config, "SEED", 42),
        "BATCH_SIZE": getattr(config, "BATCH_SIZE", 256),
        "DATA_ROOT": getattr(config, "DATA_DIR", str(REPO_ROOT / "data")),
        "DATASET": dataset,
        "MNIST_MEAN": getattr(config, "MNIST_MEAN", (0.1307,)),
        "MNIST_STD": getattr(config, "MNIST_STD", (0.3081,)),
        "CIFAR10_MEAN": getattr(config, "CIFAR10_MEAN", [0.4914, 0.4822, 0.4465]),
        "CIFAR10_STD": getattr(config, "CIFAR10_STD", [0.2470, 0.2435, 0.2616]),
        "N_TRAIN": getattr(config, "N_TRAIN", 5000),
        "RESULTS_DIR": getattr(config, "RESULTS_DIR", str(REPO_ROOT / f"experiments/{dataset}/results")),
        "CHECKPOINT_DIR": getattr(config, "CHECKPOINT_DIR", str(REPO_ROOT / f"experiments/{dataset}/checkpoints")),
        "SPLITS_DIR": getattr(config, "SPLITS_DIR", str(REPO_ROOT / f"experiments/{dataset}/splits")),
    }
    return defaults


# --------------------------------------------------------------------------- #
# Metrics (pure numpy so sklearn stays optional)
# --------------------------------------------------------------------------- #


def roc_curve_np(y_true: np.ndarray, scores: np.ndarray):
    """Return (fpr, tpr, thresholds), higher score = predicted member."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]

    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)

    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]

    tps, fps, thr = tps[idx], fps[idx], s[idx]
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)

    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, thr]


# np.trapz was removed in numpy 2.0 in favour of np.trapezoid.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def auc_np(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(_trapz(tpr, fpr))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """TPR at a fixed low FPR — the metric Carlini et al. (2022) argue for."""
    if target < fpr[0]:
        return 0.0
    return float(np.interp(target, fpr, tpr))


def attack_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict:
    member_scores = np.asarray(member_scores, dtype=np.float64)
    nonmember_scores = np.asarray(nonmember_scores, dtype=np.float64)

    assert np.all(np.isfinite(member_scores)), (
        f"FATAL: NaN or Inf detected in member score array! "
        f"NaN count: {np.isnan(member_scores).sum()}, Inf count: {np.isinf(member_scores).sum()}"
    )
    assert np.all(np.isfinite(nonmember_scores)), (
        f"FATAL: NaN or Inf detected in nonmember score array! "
        f"NaN count: {np.isnan(nonmember_scores).sum()}, Inf count: {np.isinf(nonmember_scores).sum()}"
    )

    y = np.r_[np.ones_like(member_scores), np.zeros_like(nonmember_scores)]
    s = np.r_[member_scores, nonmember_scores]

    fpr, tpr, thr = roc_curve_np(y, s)
    auc = auc_np(fpr, tpr)

    # Best balanced accuracy over all thresholds = the "attack accuracy" that
    # the threshold attack would achieve with an oracle-chosen cutoff.
    balanced = (tpr + (1.0 - fpr)) / 2.0
    best = int(np.argmax(balanced))

    return {
        "auc": auc,
        "attack_accuracy": float(balanced[best]),
        "best_threshold": float(thr[best]) if np.isfinite(thr[best]) else None,
        "tpr_at_fpr_0.001": tpr_at_fpr(fpr, tpr, 1e-3),
        "tpr_at_fpr_0.01": tpr_at_fpr(fpr, tpr, 1e-2),
        "tpr_at_fpr_0.1": tpr_at_fpr(fpr, tpr, 1e-1),
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _output_is_log_probs(logits: torch.Tensor) -> bool:
    """Model may end in log_softmax.
    Applying log_softmax twice silently corrupts the loss, so detect it."""
    total = logits.exp().sum(dim=1)
    return bool(torch.allclose(total, torch.ones_like(total), atol=1e-3))


def _prepare_inputs(
    logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure logits and labels are PyTorch tensors with correct dtype."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
    return logits, labels.long()


def score_loss(
    logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray
) -> np.ndarray:
    """
    Negative cross-entropy loss membership score.
    Higher score indicates MORE LIKELY MEMBER (training examples have lower loss).

    Formula:
        score_loss(x, y) = -CE(logits, y) = log p_theta(y | x)

    Args:
        logits: Model output logits or log-probabilities of shape (N, C).
        labels: Ground truth class integer labels of shape (N,).

    Returns:
        1D numpy array of shape (N,) where HIGHER means MORE LIKELY MEMBER.
    """
    logits_t, labels_t = _prepare_inputs(logits, labels)
    log_probs = _output_is_log_probs(logits_t)
    logp = logits_t if log_probs else F.log_softmax(logits_t, dim=-1)

    loss = F.nll_loss(logp, labels_t, reduction="none")
    scores = (-loss).detach().cpu().numpy().astype(np.float64)

    assert np.all(np.isfinite(scores)), (
        f"score_loss: non-finite values encountered! "
        f"NaNs: {np.isnan(scores).sum()}, Infs: {np.isinf(scores).sum()}"
    )
    return scores


def score_confidence(
    logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray
) -> np.ndarray:
    """
    Softmax probability assigned to the true class.
    Higher score indicates MORE LIKELY MEMBER (training examples typically have higher confidence).

    Formula:
        score_confidence(x, y) = p_theta(y | x) = exp(log p_theta(y | x))

    Args:
        logits: Model output logits or log-probabilities of shape (N, C).
        labels: Ground truth class integer labels of shape (N,).

    Returns:
        1D numpy array of shape (N,) where HIGHER means MORE LIKELY MEMBER.
    """
    logits_t, labels_t = _prepare_inputs(logits, labels)
    log_probs = _output_is_log_probs(logits_t)
    logp = logits_t if log_probs else F.log_softmax(logits_t, dim=-1)

    p = logp.exp().clamp(min=1e-12, max=1.0)
    py = p.gather(1, labels_t.unsqueeze(1)).squeeze(1)
    scores = py.detach().cpu().numpy().astype(np.float64)

    assert np.all(np.isfinite(scores)), (
        f"score_confidence: non-finite values encountered! "
        f"NaNs: {np.isnan(scores).sum()}, Infs: {np.isinf(scores).sum()}"
    )
    return scores


def score_mentr(
    logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray
) -> np.ndarray:
    r"""
    Negative modified prediction entropy (mentr) membership scoring signal.
    Higher score indicates MORE LIKELY MEMBER.

    Song & Mittal, "Systematic Evaluation of Privacy Risks of Machine Learning
    Models" (USENIX Security 2021), Section 4 defines Modified Entropy (Mentr) as:
        Mentr(f(x), y) = -(1 - f(x)_y) log(f(x)_y) - \sum_{i \ne y} f(x)_i log(1 - f(x)_i)
    where f(x)_i is the predicted probability for class i, and y is the true label.

    Because training samples (members) exhibit lower entropy (uncertainty) and higher
    confidence on the true label, Mentr is lower for members than for non-members.
    To satisfy the convention that HIGHER score indicates MORE LIKELY MEMBER:
        score_mentr(f(x), y) = -Mentr(f(x), y)
                             = (1 - f(x)_y) log(f(x)_y) + \sum_{i \ne y} f(x)_i log(1 - f(x)_i)

    Numerical Stability:
        - Evaluated with probabilities clamped away from 0: [1e-12, 1.0].
        - Evaluated with (1 - p) clamped away from 0: [1e-12, 1.0] before logarithm.
        - Runtime assertion guarantees that all score values are strictly finite (no NaN or Inf).

    Args:
        logits: Model output logits or log-probabilities of shape (N, C).
        labels: Ground truth class integer labels of shape (N,).

    Returns:
        1D numpy array of shape (N,) where HIGHER means MORE LIKELY MEMBER.
    """
    logits_t, labels_t = _prepare_inputs(logits, labels)
    log_probs = _output_is_log_probs(logits_t)
    logp = logits_t if log_probs else F.log_softmax(logits_t, dim=-1)

    p = logp.exp().clamp(min=1e-12, max=1.0)
    py = p.gather(1, labels_t.unsqueeze(1)).squeeze(1)

    one_minus_p = (1.0 - p).clamp(min=1e-12, max=1.0)
    one_minus_py = one_minus_p.gather(1, labels_t.unsqueeze(1)).squeeze(1)

    # Mentr calculation:
    # term1 = -(1 - py) * log(py)
    # term2 = -\sum_{i != y} p_i * log(1 - p_i) = -(\sum_{all i} p_i * log(1 - p_i) - py * log(1 - py))
    # Mentr = term1 + term2 >= 0
    # score = -Mentr <= 0 (higher for members)
    term1 = -(1.0 - py) * torch.log(py)
    sum_all = (p * torch.log(one_minus_p)).sum(dim=-1)
    y_term = py * torch.log(one_minus_py)
    term2 = -(sum_all - y_term)

    m = term1 + term2
    score = -m

    scores = score.detach().cpu().numpy().astype(np.float64)
    assert np.all(np.isfinite(scores)), (
        f"score_mentr: non-finite values encountered! "
        f"NaNs: {np.isnan(scores).sum()}, Infs: {np.isinf(scores).sum()}"
    )
    return scores


SCORING_FUNCTIONS = {
    "loss": score_loss,
    "confidence": score_confidence,
    "mentr": score_mentr,
}


@torch.no_grad()
def score_dataset(model, loader, device, log_probs: bool | None = None) -> dict:
    """Per-sample attack signals using pluggable scoring functions.
    Higher score = more member-like for all three signals."""
    all_logits = []
    all_y = []
    correct = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        all_logits.append(out)
        all_y.append(y)
        correct.append((out.argmax(1) == y).float().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_y, dim=0)

    if log_probs is None:
        log_probs = _output_is_log_probs(logits)
        print(f"[info] model outputs detected as "
              f"{'log-probabilities' if log_probs else 'raw logits'}")

    loss_scores = score_loss(logits, labels)
    conf_scores = score_confidence(logits, labels)
    mentr_scores = score_mentr(logits, labels)

    assert np.all(np.isfinite(loss_scores)), "NaN/Inf in loss scores"
    assert np.all(np.isfinite(conf_scores)), "NaN/Inf in confidence scores"
    assert np.all(np.isfinite(mentr_scores)), "NaN/Inf in mentr scores"

    return {
        "loss": loss_scores,
        "confidence": conf_scores,
        "mentr": mentr_scores,
        "accuracy": float(torch.cat(correct).mean()),
        "_log_probs": log_probs,
    }


def load_model(checkpoint_path: Path, device: torch.device):
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0
        state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    # Defensive: strip any surviving Opacus GradSampleModule prefix.
    state = {k.replace("_module.", "", 1): v for k, v in state.items()}

    model = get_model().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _get_dataset_transforms(dataset_name: str, cfg: dict):
    ds = dataset_name.lower().replace("-", "").replace("_", "")
    if ds == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.get("CIFAR10_MEAN", [0.4914, 0.4822, 0.4465]),
                                 cfg.get("CIFAR10_STD", [0.2470, 0.2435, 0.2616])),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.get("MNIST_MEAN", (0.1307,)), cfg.get("MNIST_STD", (0.3081,))),
        ])


def _load_raw_datasets(dataset_name: str, data_root: str, tfm):
    ds = dataset_name.lower().replace("-", "").replace("_", "")
    if ds == "cifar10":
        train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
    else:
        train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def build_loaders(args, cfg):
    dataset_name = cfg.get("DATASET", "cifar10")
    tfm = _get_dataset_transforms(dataset_name, cfg)
    train_ds, test_ds = _load_raw_datasets(dataset_name, args.data_root, tfm)

    if args.member_index_file and Path(args.member_index_file).exists():
        path = Path(args.member_index_file)
        pool = (np.load(path) if path.suffix == ".npy"
                else np.array(json.loads(path.read_text())))
        print(f"[info] member pool restricted to {len(pool)} indices from {path.name}")
    else:
        pool = np.arange(len(train_ds))

    n = min(args.n_samples, len(pool), len(test_ds))
    if n < args.n_samples:
        print(f"[warn] requested {args.n_samples} per class, using {n} (dataset limit)")

    rng = np.random.default_rng(args.seed)
    member_idx = rng.choice(pool, size=n, replace=False)
    nonmember_idx = rng.choice(len(test_ds), size=n, replace=False)

    kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return (
        DataLoader(Subset(train_ds, member_idx.tolist()), **kwargs),
        DataLoader(Subset(test_ds, nonmember_idx.tolist()), **kwargs),
        n,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_epsilon(value) -> float:
    """Manifest stores the baseline as the string "inf"."""
    if value is None:
        return float("inf")
    if isinstance(value, str):
        return float("inf") if value.lower() in {"inf", "infinity"} else float(value)
    return float(value)


def main():
    cfg = _load_config()

    results_dir = Path(cfg.get("RESULTS_DIR", REPO_ROOT / f"experiments/{cfg['DATASET']}/results"))
    splits_dir = Path(cfg.get("SPLITS_DIR", REPO_ROOT / f"experiments/{cfg['DATASET']}/splits"))

    default_manifest = results_dir / "epsilon_sweep.json"
    if not default_manifest.exists():
        legacy_manifest = REPO_ROOT / "experiments/results/epsilon_sweep.json"
        if legacy_manifest.exists():
            default_manifest = legacy_manifest

    default_out = results_dir / "threshold_attack.json"
    default_scores = results_dir / "mia_scores"

    member_npy = splits_dir / "member_indices.npy"
    member_seed_npy = splits_dir / f"member_indices_n{cfg.get('N_TRAIN', 5000)}_seed{cfg['SEED']}.npy"
    member_json = splits_dir / f"member_indices_n{cfg.get('N_TRAIN', 5000)}_seed{cfg['SEED']}.json"

    if member_npy.exists():
        default_member_file = str(member_npy)
    elif member_seed_npy.exists():
        default_member_file = str(member_seed_npy)
    elif member_json.exists():
        default_member_file = str(member_json)
    else:
        legacy_member_json = REPO_ROOT / "experiments/splits" / f"member_indices_n{cfg.get('N_TRAIN', 5000)}_seed{cfg['SEED']}.json"
        default_member_file = str(legacy_member_json if legacy_member_json.exists() else member_npy)

    ap = argparse.ArgumentParser(description="Threshold MIA against the epsilon sweep")
    ap.add_argument("--manifest", default=str(default_manifest))
    ap.add_argument("--out", default=str(default_out))
    ap.add_argument("--scores-dir", default=str(default_scores))
    ap.add_argument("--data-root", default=cfg["DATA_ROOT"])
    ap.add_argument("--member-index-file", default=default_member_file,
                    help=".npy/.json of train indices actually used for training")
    ap.add_argument("--n-samples", type=int, default=10000,
                    help="Members and non-members each (balanced attack set)")
    ap.add_argument("--batch-size", type=int, default=cfg["BATCH_SIZE"])
    ap.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    ap.add_argument("--seed", type=int, default=cfg["SEED"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run src.sweep_epsilon first.")

    import re
    from types import SimpleNamespace

    manifest = json.loads(manifest_path.read_text())
    if isinstance(manifest, dict):
        if "results" in manifest:
            entries = manifest["results"]
        else:
            entries = []
            for nm_key, val in manifest.items():
                if isinstance(val, dict) and "runs" in val:
                    entries.extend(val["runs"])
                elif isinstance(val, dict):
                    entries.append(val)
    else:
        entries = manifest

    device = torch.device(args.device)
    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Cache loaders per seed
    loaders_by_seed = {}
    n_train = cfg.get("N_TRAIN", 5000)

    results = []
    for entry in entries:
        ckpt = Path(entry.get("checkpoint_path") or entry["checkpoint"])
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        if not ckpt.exists():
            candidate = Path(cfg["CHECKPOINT_DIR"]) / ckpt.name
            if candidate.exists():
                ckpt = candidate
        if not ckpt.exists():
            print(f"[skip] missing checkpoint {ckpt}")
            continue

        seed = entry.get("seed")
        if seed is None:
            m = re.search(r"seed(\d+)", ckpt.name)
            seed = int(m.group(1)) if m else int(args.seed)
        else:
            seed = int(seed)

        # Determine member index file for this checkpoint's seed
        seed_json = splits_dir / f"member_indices_n{n_train}_seed{seed}.json"
        seed_npy = splits_dir / f"member_indices_n{n_train}_seed{seed}.npy"

        if seed_json.exists():
            member_file = str(seed_json)
        elif seed_npy.exists():
            member_file = str(seed_npy)
        elif seed == cfg["SEED"] and (splits_dir / "member_indices.npy").exists():
            member_file = str(splits_dir / "member_indices.npy")
        else:
            raise FileNotFoundError(
                f"FATAL: No member index file found for seed {seed} under {splits_dir}!"
            )

        # STRICT ASSERTION: verify loaded member index file corresponds to checkpoint's seed
        assert f"seed{seed}" in Path(member_file).name, (
            f"FATAL: Member index file '{member_file}' does not match checkpoint seed '{seed}'! "
            f"Evaluating model against wrong seed indices produces false AUC ~0.50."
        )

        if seed not in loaders_by_seed:
            args_seed = SimpleNamespace(
                data_root=args.data_root,
                member_index_file=member_file,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=seed,
            )
            m_ldr, nm_ldr, n_actual = build_loaders(args_seed, cfg)
            loaders_by_seed[seed] = (m_ldr, nm_ldr, n_actual, member_file)

        member_loader, nonmember_loader, n, loaded_split = loaders_by_seed[seed]
        assert f"seed{seed}" in Path(loaded_split).name, (
            f"FATAL: Loaded split '{loaded_split}' does not match checkpoint seed '{seed}'!"
        )

        eps = parse_epsilon(entry.get("epsilon"))
        label = "baseline (no DP)" if not np.isfinite(eps) else f"eps={eps:.2f}"

        model = load_model(ckpt, device)
        mem = score_dataset(model, member_loader, device)
        non = score_dataset(model, nonmember_loader, device, log_probs=mem["_log_probs"])

        row = {
            "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
            "label": label,
            "epsilon": None if not np.isfinite(eps) else eps,
            "noise_multiplier": entry.get("noise_multiplier"),
            "seed": entry.get("seed"),
            "test_accuracy": entry.get("test_accuracy"),
            "member_accuracy": mem["accuracy"],
            "nonmember_accuracy": non["accuracy"],
            "generalization_gap": mem["accuracy"] - non["accuracy"],
            "attacks": {
                signal: attack_metrics(mem[signal], non[signal])
                for signal in ("loss", "confidence", "mentr")
            },
        }
        results.append(row)

        np.savez_compressed(
            scores_dir / f"{ckpt.stem}.npz",
            **{f"member_{k}": mem[k] for k in ("loss", "confidence", "mentr")},
            **{f"nonmember_{k}": non[k] for k in ("loss", "confidence", "mentr")},
        )

        a = row["attacks"]["loss"]
        print(f"{label:<20} gap={row['generalization_gap']:+.4f}  "
              f"AUC={a['auc']:.4f}  acc={a['attack_accuracy']:.4f}  "
              f"TPR@1%FPR={a['tpr_at_fpr_0.01']:.4f}")

    if not results:
        raise SystemExit("No checkpoints scored — nothing written.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "dataset": cfg["DATASET"],
            "n_members": n, "n_nonmembers": n, "seed": args.seed,
            "member_index_file": args.member_index_file,
            "attack": "threshold", "signals": ["loss", "confidence", "mentr"],
        },
        "results": results,
    }, indent=2))

    print(f"\nWrote {out_path}")
    print(f"Per-sample scores in {scores_dir}/  ->  run src.plot_mia_curves next.")

    baseline = next((r for r in results if r["epsilon"] is None), None)
    if baseline and baseline["attacks"]["loss"]["auc"] < 0.55:
        print("\n[!] Baseline AUC is near 0.50: the non-private model barely "
              "overfits, so there is no attack signal for DP to remove. See the "
              "notes in the docstring of src/plot_mia_curves.py before writing up.")


if __name__ == "__main__":
    main()
