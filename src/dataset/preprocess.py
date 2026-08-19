"""
preprocess.py

Image preprocessing utilities for the M1 Cats vs Dogs pipeline.

Responsibilities
----------------
- Build training/evaluation transforms
- Provide reusable preprocessing API
"""

from torchvision import transforms

from src.dataset.constants import IMAGE_SIZE


def get_train_transform():
    """Return transforms for training images."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform():
    """Return transforms for validation/test images."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_transforms(train: bool = True):
    """
    Factory method returning the appropriate transform.

    Parameters
    ----------
    train : bool
        True for training transforms, False for evaluation transforms.
    """
    return get_train_transform() if train else get_eval_transform()


if __name__ == "__main__":
    print("Training Transform:")
    print(get_train_transform())

    print("\nEvaluation Transform:")
    print(get_eval_transform())
