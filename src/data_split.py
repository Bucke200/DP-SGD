"""
Fixed member/non-member data split for MIA evaluation on MNIST and CIFAR-10.

Subsamples the training set so the model is forced to memorize,
producing a measurable train/test gap that the threshold attack can exploit.
The exact member indices are persisted so threshold attacks can
distinguish true members from non-members at evaluation time.

Usage:
    from src.data_split import get_data_loaders

    train_loader, test_loader, member_indices = get_data_loaders(
        n_train=5000, seed=42, batch_size=64,
    )
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config

SPLIT_DIR = getattr(config, "SPLITS_DIR", "experiments/splits")


def _mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def _cifar10_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])


def member_index_path(n_train: int = config.N_TRAIN, seed: int = config.SEED) -> str:
    splits_dir = getattr(config, "SPLITS_DIR", SPLIT_DIR)
    return os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.json")


def get_split_loaders(
    n_train: int = config.N_TRAIN,
    seed: int = config.SEED,
    batch_size: int = config.BATCH_SIZE,
    test_batch_size: int = config.TEST_BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
    save_indices: bool = True,
):
    """
    Return (train_loader, test_loader, member_indices) for MNIST.

    train_loader draws from a fixed random subset of the MNIST train split.
    test_loader is the full 10k MNIST test split (non-members).
    member_indices is a sorted int array of the selected training indices.
    """
    tfm = _mnist_transform()

    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    splits_dir = getattr(config, "SPLITS_DIR", SPLIT_DIR)
    seed_json_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.json")
    seed_npy_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.npy")

    if os.path.exists(seed_json_path):
        with open(seed_json_path, "r") as f:
            member_indices = np.array(json.load(f), dtype=np.int64)
    elif os.path.exists(seed_npy_path):
        member_indices = np.load(seed_npy_path).astype(np.int64)
    else:
        rng = np.random.default_rng(seed)
        member_indices = np.sort(rng.choice(len(full_train), size=n_train, replace=False))
        if save_indices:
            os.makedirs(splits_dir, exist_ok=True)
            np.save(seed_npy_path, member_indices)
            with open(seed_json_path, "w") as f:
                json.dump(member_indices.tolist(), f)
            if seed == getattr(config, "SEED", 42):
                npy_path = os.path.join(splits_dir, "member_indices.npy")
                np.save(npy_path, member_indices)

    train_loader = DataLoader(
        Subset(full_train, member_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, member_indices


def get_cifar10_split_loaders(
    n_train: int = config.N_TRAIN,
    seed: int = config.SEED,
    batch_size: int = config.BATCH_SIZE,
    test_batch_size: int = config.TEST_BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
    save_indices: bool = True,
):
    """
    Return (train_loader, test_loader, member_indices) for CIFAR-10.

    Downloads CIFAR-10 via torchvision.
    Applies normalization using CIFAR-10 constants from config.
    Subsamples the training set to config.N_TRAIN using a seeded RNG.
    Saves member indices to config.SPLITS_DIR/member_indices.npy.
    Returns (train_loader, test_loader, member_indices).
    """
    tfm = _cifar10_transform()

    full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)

    splits_dir = getattr(config, "SPLITS_DIR", SPLIT_DIR)
    seed_json_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.json")
    seed_npy_path = os.path.join(splits_dir, f"member_indices_n{n_train}_seed{seed}.npy")

    if os.path.exists(seed_json_path):
        with open(seed_json_path, "r") as f:
            member_indices = np.array(json.load(f), dtype=np.int64)
    elif os.path.exists(seed_npy_path):
        member_indices = np.load(seed_npy_path).astype(np.int64)
    else:
        rng = np.random.default_rng(seed)
        member_indices = np.sort(rng.choice(len(full_train), size=n_train, replace=False))
        if save_indices:
            os.makedirs(splits_dir, exist_ok=True)
            np.save(seed_npy_path, member_indices)
            with open(seed_json_path, "w") as f:
                json.dump(member_indices.tolist(), f)
            if seed == getattr(config, "SEED", 42):
                npy_path = os.path.join(splits_dir, "member_indices.npy")
                np.save(npy_path, member_indices)

    train_loader = DataLoader(
        Subset(full_train, member_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, member_indices


def get_data_loaders(
    dataset: str | None = None,
    n_train: int = config.N_TRAIN,
    seed: int = config.SEED,
    batch_size: int = config.BATCH_SIZE,
    test_batch_size: int = config.TEST_BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
    save_indices: bool = True,
):
    """
    Dispatcher function to return (train_loader, test_loader, member_indices)
    based on config.DATASET or the explicit dataset argument ('mnist' or 'cifar10').
    """
    if dataset is None:
        dataset = getattr(config, "DATASET", "cifar10")

    ds = dataset.lower().replace("-", "").replace("_", "")
    if ds == "mnist":
        return get_split_loaders(
            n_train=n_train,
            seed=seed,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            data_dir=data_dir,
            save_indices=save_indices,
        )
    elif ds == "cifar10":
        return get_cifar10_split_loaders(
            n_train=n_train,
            seed=seed,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            data_dir=data_dir,
            save_indices=save_indices,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected 'mnist' or 'cifar10'.")
