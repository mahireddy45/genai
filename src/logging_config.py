from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO", module_levels: Optional[Dict[str, str]] = None, rotate_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    """Configure root logging for the application.

    - Creates `log_dir` if missing
    - Adds a file handler with datetime in filename and a console handler
    - Allows overriding log level per-module via `module_levels` dict

    Args:
        log_dir: directory to write log files
        log_level: default root log level (e.g. "DEBUG", "INFO")
        module_levels: optional mapping {"module.name": "DEBUG"} to set specific levels
        rotate_bytes: max size per log file before rotation
        backup_count: number of rotated files to keep
    """
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Check if handlers are already configured (avoid duplicates)
    has_file_handler = any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)
    has_console_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)

    # Add file handler if not present
    if not has_file_handler:
        # Create log filename with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"app_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(p / log_filename), maxBytes=rotate_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Add console handler if not present
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # Apply module-specific levels
    if module_levels:
        for mod, lvl in module_levels.items():
            logging.getLogger(mod).setLevel(getattr(logging, lvl.upper(), logging.INFO))


def get_logger(name: str | None = None) -> logging.Logger:
    """Convenience wrapper to obtain a logger for a module or component."""
    return logging.getLogger(name)
