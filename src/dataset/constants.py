"""
constants.py

Centralized project constants for the Dataset module.

This file contains:
- Project paths
- Dataset paths
- Image settings
- Dataset split configuration
- DataLoader configuration
- Image normalization parameters
- Validation settings
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

SOURCE_DATA_DIR = DATA_DIR / "source"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# =============================================================================
# Artifact Paths
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = ARTIFACTS_DIR / "logs"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# =============================================================================
# Dataset Configuration
# =============================================================================

CLASS_NAMES = ["Cat", "Dog"]
NUM_CLASSES = len(CLASS_NAMES)

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
)

# =============================================================================
# Image Configuration
# =============================================================================

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS = 3

# =============================================================================
# Dataset Split Configuration
# =============================================================================

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

RANDOM_SEED = 42

# =============================================================================
# DataLoader Configuration
# =============================================================================

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4

PIN_MEMORY = True
SHUFFLE = True

# =============================================================================
# Image Normalization (ImageNet)
# =============================================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

# =============================================================================
# Dataset Validation
# =============================================================================

MIN_IMAGES_PER_CLASS = 1

ALLOW_EMPTY_DIRECTORIES = False

# =============================================================================
# Logging
# =============================================================================

LOGGER_NAME = "dataset"

# =============================================================================
# Device Configuration
# =============================================================================

DEFAULT_DEVICE = "cuda"
CPU_DEVICE = "cpu"