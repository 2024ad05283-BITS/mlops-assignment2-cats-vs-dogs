"""
dataset/manager.py

Production-style DatasetManager for M1.

Responsibilities
----------------
1. Check dataset availability
2. Download dataset if missing (hook)
3. Extract archive
4. Remove corrupted images
5. Validate dataset
6. Split dataset
7. Create PyTorch DataLoaders

Replace download_dataset() with your preferred implementation
(Kaggle API or direct URL).
"""

from pathlib import Path
import shutil
import zipfile
from typing import Tuple

from src.dataset.constants import RAW_DATA_DIR
from src.dataset.validator import validate_dataset
from src.dataset.splitter import split_dataset
from src.dataset.loader import create_dataloaders


class DatasetManager:

    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config.get("raw_dir", RAW_DATA_DIR))
        self.dataset_zip = Path(config.get("dataset_zip", ""))

    def dataset_exists(self) -> bool:
        if not self.raw_dir.exists():
            return False

        class_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        return len(class_dirs) >= 2

    def download_dataset(self):
        """
        Implement your downloader here.

        Example:
            - Kaggle API
            - Google Drive
            - Internal artifact server
        """
        raise NotImplementedError(
            "Implement dataset download logic."
        )

    def extract_dataset(self):
        if not self.dataset_zip.exists():
            return

        with zipfile.ZipFile(self.dataset_zip, "r") as zf:
            zf.extractall(self.raw_dir)

    def remove_corrupted_images(self):
        from PIL import Image

        for img in self.raw_dir.rglob("*"):
            if not img.is_file():
                continue
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                print(f"Removing corrupted image: {img}")
                img.unlink(missing_ok=True)

    def prepare(self) -> Tuple:
        if not self.dataset_exists():
            print("Dataset not found.")
            self.download_dataset()
            self.extract_dataset()

        self.remove_corrupted_images()

        validate_dataset()

        split_dataset()

        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=self.config.get("batch_size", 32),
            num_workers=self.config.get("num_workers", 4),
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":

    config = {
        "batch_size": 32,
        "num_workers": 4,
    }

    manager = DatasetManager(config)

    train_loader, val_loader, test_loader = manager.prepare()

    print(train_loader)
    print(val_loader)
    print(test_loader)
