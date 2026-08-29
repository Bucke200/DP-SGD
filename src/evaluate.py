import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, arg2, arg3=None, criterion: nn.Module = None):
    """
    Evaluate the model on test set.

    Supports both signatures:
        evaluate(model, device, test_loader, criterion)
        evaluate(model, test_loader, device, criterion=None)

    Returns:
        test_loss (float): Average cross entropy loss across test dataset.
        accuracy (float): Top-1 accuracy percentage (0-100%).
    """
    if isinstance(arg2, DataLoader):
        test_loader = arg2
        device = arg3
    else:
        device = arg2
        test_loader = arg3

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    test_loss /= total
    accuracy = 100.0 * correct / total
    return test_loss, accuracy
