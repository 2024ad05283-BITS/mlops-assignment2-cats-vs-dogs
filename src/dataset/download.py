"""
download.py

Local dataset importer.

Expected source:
src/dataset/DogCatDataset/
    Cat/
    Dog/

Copies the dataset into data/raw only if it is not already present.
"""

from pathlib import Path
import shutil

from src.dataset.constants import RAW_DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_DATASET = PROJECT_ROOT / "src" / "dataset" / "DogCatDataset"

CLASS_FOLDERS = ("Cat", "Dog")


def dataset_exists() -> bool:
    """Return True if the dataset is already available in data/raw."""
    return all((RAW_DATA_DIR / cls).is_dir() for cls in CLASS_FOLDERS)


def download_dataset() -> Path:
    """
    Import the local dataset into data/raw.

    Returns
    -------
    Path
        Path to data/raw
    """

    if dataset_exists():
        print("[INFO] Dataset already exists in data/raw. Skipping copy.")
        return RAW_DATA_DIR

    if not LOCAL_DATASET.exists():
        raise FileNotFoundError(
            f"Local dataset folder not found:\n{LOCAL_DATASET}"
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Importing dataset from:\n{LOCAL_DATASET}")

    for cls in CLASS_FOLDERS:

        src = LOCAL_DATASET / cls
        dst = RAW_DATA_DIR / cls

        if not src.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {src}"
            )

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

        print(f"[INFO] Copied {cls}")

    print("[INFO] Dataset import completed.")

    return RAW_DATA_DIR


if __name__ == "__main__":
    download_dataset()