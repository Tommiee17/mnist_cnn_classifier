import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

def get_data(root="./data", batch_size=32, val_size=10_000, seed=1):
    mean, std = 0.1307, 0.3081
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_full = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=dataset_transform
    )

    test_set = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=dataset_transform
    )

    if val_size <= 0 or val_size >= len(train_full):
        raise ValueError(f"val_size must be between 1 and {len(train_full) - 1}, got {val_size}")

    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], torch.Generator().manual_seed(seed))

    # Data Loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, (mean, std)
