import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")


def configure_logger() -> None:
    """
    Configure root logger with file and console handlers.
    """

    root_logger = logging.getLogger()

    # Prevent duplicate handlers when module is imported multiple times
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# Configure logging when module is imported
configure_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance.

    Example:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
