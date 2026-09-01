import torch
import torch.nn as nn
from opacus.validators import ModuleValidator

import config


class SampleCNN(nn.Module):
    """
    A simple CNN architecture suitable for MNIST classification.
    Consists of 2 Conv layers, ReLU activations, MaxPool layers, and 2 Fully Connected layers.
    """

    def __init__(self):
        super(SampleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class CifarCNN(nn.Module):
    """
    CNN architecture for CIFAR-10 classification (Opacus DP-SGD compatible).
    Architecture:
      Conv2d(3, 32, 3, padding=1) -> ReLU -> Conv2d(32, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
      -> Conv2d(32, 64, 3, padding=1) -> ReLU -> Conv2d(64, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
      -> Flatten -> Linear(64*8*8, 256) -> ReLU -> Linear(256, 10)
    No BatchNorm or Dropout to ensure Opacus compatibility and allow memorization for MIA.
    """

    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.pool1(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.pool2(self.relu4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(dataset: str | None = None) -> nn.Module:
    """
    Return appropriate model architecture based on dataset.

    Args:
        dataset: "mnist" or "cifar10". If None, reads from config.DATASET.

    Returns:
        SampleCNN instance for MNIST or CifarCNN instance for CIFAR-10.
    """
    if dataset is None:
        dataset = getattr(config, "DATASET", "cifar10")

    ds = dataset.lower().replace("-", "").replace("_", "")
    if ds == "mnist":
        return SampleCNN()
    elif ds == "cifar10":
        return CifarCNN()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected 'mnist' or 'cifar10'.")


def validate_model_for_opacus(model: nn.Module):
    """
    Validates model compatibility with Opacus per-sample gradient computation.

    Returns:
        is_valid (bool): True if model is valid, False otherwise.
        errors (list): List of error messages returned by ModuleValidator.validate.
    """
    errors = ModuleValidator.validate(model)
    is_valid = ModuleValidator.is_valid(model)
    return is_valid, errors
