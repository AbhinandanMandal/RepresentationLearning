
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# transform = transforms.ToTensor()
transform = transforms.ToTensor()
DATA_DIR = Path(__file__).resolve().parent / "data"

# Initially we'll be working with MNIST dataset


def get_dataloader(batch_size: int = 16, num_workers: int = 0):
    """Return MNIST training and test loaders with a project-local data directory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    train_dataset = datasets.MNIST(
        # root="./", transform=transform, download=True, train=True)
        root=DATA_DIR, transform=transform, download=True, train=True)
    test_dataset = datasets.MNIST(
        # root="./", transform=transform, train=False, download=True)
        root=DATA_DIR, transform=transform, train=False, download=True)

    train_dataloader = DataLoader(
        # dataset=train_dataset, batch_size=batch_size, shuffle=True)
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(
        # dataset=test_dataset, batch_size=batch_size, shuffle=False)
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    # Additionaly pin_memory = True fasten data transfer from CPU to GPU memory

    return train_dataloader, test_dataloader
