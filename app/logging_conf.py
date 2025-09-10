import logging
from logging.handlers import RotatingFileHandler
import os

# --- Log directory ---

# Directory for log files (override with LOG_DIR env var).
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# --- Setup function ---

def setup_logging() -> logging.Logger:
    """Configure root logger with console + rotating file handler."""
    logger = logging.getLogger()

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )

    # File handler (rotates at 1 MiB, keeps 3 backups)
    log_file_path = os.path.join(LOG_DIR, "app.log")
    file_handler = RotatingFileHandler(log_file_path, maxBytes=1_048_576, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
