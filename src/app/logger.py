# src/app/logger.py
import logging
import os

# Ensure the log directory exists
log_dir = ".llxprt/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "app.log")

# Setup formatting
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("app")
