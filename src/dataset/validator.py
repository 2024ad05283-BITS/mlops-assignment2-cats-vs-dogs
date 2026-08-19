"""
validator.py

Dataset validation utilities for the MLOps project.

Responsibilities
----------------
1. Verify dataset directory exists.
2. Verify required class folders exist.
3. Validate supported image extensions.
4. Detect corrupted images.
5. Count images per class.
6. Print dataset statistics.
7. Raise meaningful exceptions on failure.
"""

from pathlib import Path
from typing import Dict
from .constants import RAW_DATA_DIR
from PIL import Image

from src.dataset.constants import (
    RAW_DATA_DIR,
    SUPPORTED_EXTENSIONS,
)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""


def _count_images(directory: Path) -> int:
    """Count valid image files inside a directory."""

    count = 0

    for ext in SUPPORTED_EXTENSIONS:
        count += len(list(directory.rglob(f"*{ext}")))

    return count


def _validate_image(image_path: Path) -> bool:
    """
    Validate image integrity.

    Returns
    -------
    bool
        True if image is readable.
    """

    try:
        with Image.open(image_path) as img:
            img.verify()
        return True

    except Exception:
        return False


def validate_dataset(
    dataset_dir: Path = RAW_DATA_DIR,
) -> Dict[str, int]:
    """
    Validate dataset structure.

    Expected structure

    data/raw/

        cats/

        dogs/

    Parameters
    ----------
    dataset_dir : Path

    Returns
    -------
    Dict[str,int]

        Image count per class.
    """

    if not dataset_dir.exists():
        raise DatasetValidationError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    class_dirs = [
        d for d in dataset_dir.iterdir()
        if d.is_dir()
    ]

    if len(class_dirs) == 0:
        raise DatasetValidationError(
            "No class folders found."
        )

    stats = {}

    total_images = 0

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    for class_dir in sorted(class_dirs):

        image_count = 0

        for file in class_dir.rglob("*"):

            if not file.is_file():
                continue

            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            if not _validate_image(file):
                raise DatasetValidationError(
                    f"Corrupted image detected:\n{file}"
                )

            image_count += 1

        if image_count == 0:
            raise DatasetValidationError(
                f"No valid images found in:\n{class_dir}"
            )

        stats[class_dir.name] = image_count

        total_images += image_count

        print(f"{class_dir.name:<20}: {image_count}")

    print("-" * 60)
    print(f"Total Images        : {total_images}")
    print(f"Classes             : {len(stats)}")
    print("=" * 60)

    return stats


if __name__ == "__main__":

    validate_dataset()