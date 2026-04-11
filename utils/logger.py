"""
=============================================================================
 Logger — Structured logging for the authentication system.
=============================================================================
"""

import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_logger(name="FaceAuth", log_file=None):
    """
    Set up a structured logger with file and console handlers.

    Args:
        name (str): Logger name.
        log_file (str): Path to log file. Defaults to logs directory.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # ── Console handler ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # ── File handler ──
    if log_file is None:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        log_file = os.path.join(
            config.LOGS_DIR,
            f"auth_{datetime.now().strftime('%Y%m%d')}.log"
        )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


# Create default logger
logger = setup_logger()
