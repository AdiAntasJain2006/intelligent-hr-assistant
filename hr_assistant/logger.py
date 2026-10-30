"""
logger.py

Centralized logging configuration for the chatbot application.

Responsibilities:
    - Create and configure a named application logger.
    - Save log messages to a file.
    - Display log messages in the terminal.
    - Automatically rotate log files when they reach a specified size.
    - Prevent duplicate handlers from being added.
    - Maintain a consistent logging format across all modules.

Usage:
    Import the logger wherever logging is required:

        from chatbot.logger import logger

    Example:
        logger.info("Chatbot initialized")
        logger.warning("No documents found")
        logger.error("Failed to generate response")

Log Levels:
    DEBUG    : Detailed debugging information.
    INFO     : Normal application events.
    WARNING  : Unexpected situations.
    ERROR    : Application errors.
    CRITICAL : Serious application failures.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

# Create logs directory
LOG_DIR = "logs"
now = datetime.now()

day = now.strftime("%d")
month = months[now.month]

f_name = f"{day}-{month}-{now.year}"
LOG_DIR = os.path.join(LOG_DIR, f_name)

os.makedirs(LOG_DIR, exist_ok=True)

# Logger configuration
file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE = os.path.join(LOG_DIR, f"run_{file_name}.log")

logging.basicConfig(
    level = logging.INFO, # Sets the minimum logging level.
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s", # defines how the log message will look
    handlers=[
            RotatingFileHandler(
                LOG_FILE,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=5,
                encoding="utf-8"
            ),
            logging.StreamHandler()
    ]
)

def get_logger(name : str) -> logging.Logger :
    return logging.getLogger(name)

