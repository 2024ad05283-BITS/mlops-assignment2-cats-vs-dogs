"""
logger.py

Production-ready logging utility.

Features
--------
- Console logging
- File logging
- Automatic log directory creation
- Rotating log files
- Configurable log level
"""

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# ==========================================================
# Configuration
# ==========================================================

LOG_DIR = Path("artifacts/logs")

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOG_DIR / "api.log"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

# ==========================================================
# Logger Factory
# ==========================================================


def get_logger(
    name: str = "mlops",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Returns configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)

    # ------------------------------------------------------
    # Console Handler
    # ------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(level)

    console_handler.setFormatter(formatter)

    # ------------------------------------------------------
    # File Handler
    # ------------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )

    file_handler.setLevel(level)

    file_handler.setFormatter(formatter)

    # ------------------------------------------------------
    # Attach Handlers
    # ------------------------------------------------------

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


# ==========================================================
# Smoke Test
# ==========================================================

if __name__ == "__main__":

    logger = get_logger("monitoring")

    logger.info("Logger initialized successfully.")

    logger.warning("Sample warning message.")

    logger.error("Sample error message.")

    print("=" * 60)
    print("Logger Smoke Test Completed")
    print(f"Log File : {LOG_FILE}")
    print("=" * 60)
    