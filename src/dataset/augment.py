"""
augment.py

Data augmentation transforms for Cats vs Dogs training.

This module provides reusable torchvision transform pipelines for
training, validation and inference.
"""

from torchvision import transforms

from src.dataset.constants import IMAGE_SIZE


def get_train_transforms():
    """Return augmentation pipeline used during training."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.8, 1.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms():
    """Return deterministic transforms for validation/testing."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_inference_transforms():
    """Return transforms used during inference."""
    return get_eval_transforms()


if __name__ == "__main__":
    print("Training Transform")
    print(get_train_transforms())

    print("\nValidation Transform")
    print(get_eval_transforms())
