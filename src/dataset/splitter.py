"""
splitter.py

Create reproducible train/validation/test directory structure
from data/raw using stratified random splitting.

Expected input:
data/raw/
    Cat/
    Dog/

Output:
data/processed/
    train/
    val/
    test/
"""

from pathlib import Path
import random
import shutil

from src.dataset.constants import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    SUPPORTED_EXTENSIONS,
)


def _image_files(folder: Path):
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(files)


def split_dataset(seed: int = 42):
    random.seed(seed)

    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    for split in ["train", "val", "test"]:
        (PROCESSED_DATA_DIR / split).mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]

    if not class_dirs:
        raise RuntimeError("No class folders found in data/raw")

    for cls in class_dirs:
        images = _image_files(cls)
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        train = images[:n_train]
        val = images[n_train:n_train+n_val]
        test = images[n_train+n_val:]

        for split_name, subset in [
            ("train", train),
            ("val", val),
            ("test", test),
        ]:
            dst = PROCESSED_DATA_DIR / split_name / cls.name
            dst.mkdir(parents=True, exist_ok=True)

            for img in subset:
                shutil.copy2(img, dst / img.name)

        print(f"{cls.name}: train={len(train)} val={len(val)} test={len(test)}")

    print("Dataset split completed.")


if __name__ == "__main__":
    split_dataset()
