from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloaders(data_dir: str, batch_size: int, test_batch_size: int):
    """
    Load MNIST dataset and return PyTorch DataLoaders for train and test splits.

    Args:
        data_dir: Directory where dataset will be stored/downloaded.
        batch_size: Batch size for training data.
        test_batch_size: Batch size for testing data.

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Helpful for uniform batch sizing in DP-SGD
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_mnist_loaders(batch_size: int = 64, test_batch_size: int = 1000, data_dir: str = "./data"):
    """
    Convenience wrapper for get_mnist_dataloaders with default parameters.
    """
    return get_mnist_dataloaders(data_dir=data_dir, batch_size=batch_size, test_batch_size=test_batch_size)
