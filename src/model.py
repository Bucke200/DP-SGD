import torch
import torch.nn as nn
from opacus.validators import ModuleValidator


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
