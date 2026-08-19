"""
loader.py

Production-ready dataset loader for M1.

Creates PyTorch Dataset and DataLoaders.
"""

from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.dataset.constants import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
)


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_datasets():
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(True))
    val_ds = datasets.ImageFolder(VAL_DIR, transform=get_transforms(False))
    test_ds = datasets.ImageFolder(TEST_DIR, transform=get_transforms(False))
    return train_ds, val_ds, test_ds


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds, val_ds, test_ds = create_datasets()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"Train batches : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")
