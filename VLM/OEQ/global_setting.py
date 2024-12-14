import os
import sys
import logging

def initialize_logging(log_file='datalog.log', log_level=logging.INFO):
    """Initialize logging only if not already initialized."""

    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, log_file)

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='w'
        )

def add_project_root_to_path(base_path=__file__):
    """Add the project root directory to Python path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(base_path), "../.."))
    sys.path.insert(0, project_root)
