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


@torch.no_grad()
def score_dataset(model, loader, device, log_probs: bool | None = None) -> dict:
    """Per-sample attack signals. Higher score = more member-like for all three."""
    losses, confidences, mentr, correct = [], [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)

        if log_probs is None:
            log_probs = _output_is_log_probs(out)
            print(f"[info] model outputs detected as "
                  f"{'log-probabilities' if log_probs else 'raw logits'}")

        logp = out if log_probs else F.log_softmax(out, dim=1)
        p = logp.exp().clamp(1e-12, 1.0)

        loss = F.nll_loss(logp, y, reduction="none")
        py = p.gather(1, y[:, None]).squeeze(1)

        # Song & Mittal (2021) modified entropy: label-aware, lower = member.
        one_minus_p = (1.0 - p).clamp(1e-12, 1.0)
        m = -(1.0 - py) * torch.log(py) - (p * torch.log(one_minus_p)).sum(1) \
            + py * torch.log(one_minus_p.gather(1, y[:, None]).squeeze(1))

        losses.append(loss.cpu())
        confidences.append(py.cpu())
        mentr.append(m.cpu())
        correct.append((out.argmax(1) == y).float().cpu())

    return {
        "loss": (-torch.cat(losses)).numpy(),        # low loss  -> member
        "confidence": torch.cat(confidences).numpy(),  # high conf -> member
        "mentr": (-torch.cat(mentr)).numpy(),          # low Mentr -> member
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
