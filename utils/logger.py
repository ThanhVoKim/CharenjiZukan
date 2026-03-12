# -*- coding: utf-8 -*-
"""
utils/logger.py — Logging configuration for CharenjiZukan project

Usage:
    from utils.logger import get_logger, setup_logging
    logger = get_logger(__name__)
    logger.info("Processing...")

For Google Colab:
    from utils.logger import setup_colab_logging
    setup_colab_logging(verbose=True)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Cấu hình mặc định
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file (tùy chọn)
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "charenjizukan.log"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: str = DEFAULT_FORMAT,
) -> None:
    """
    Cấu hình logging cho toàn bộ project.
    Gọi một lần duy nhất tại entry point (main script).
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Đường dẫn file log (tùy chọn)
        format_string: Format string cho log messages
    
    Example:
        >>> setup_logging(level=logging.DEBUG, log_file=Path("logs/app.log"))
    """
    # Tạo thư mục logs nếu cần
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Cấu hình handlers
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),  # Console output
    ]
    
    if log_file:
        handlers.append(
            logging.FileHandler(log_file, encoding="utf-8")
        )
    
    # Cấu hình root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=DEFAULT_DATE_FORMAT,
        handlers=handlers,
        force=True,  # Override existing config
    )


def get_logger(name: str = "srt_translator") -> logging.Logger:
    """
    Lấy logger với tên module.
    
    Args:
        name: Tên logger, thường dùng __name__
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing...")
    """
    return logging.getLogger(name)


def setup_colab_logging(verbose: bool = False) -> None:
    """
    Cấu hình logging cho Google Colab.
    Đơn giản, chỉ output ra console.
    
    Args:
        verbose: Nếu True, bật DEBUG level
    
    Example:
        >>> setup_colab_logging(verbose=True)
        [Logger] Logging configured: level=DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Reset handlers (quan trọng trong Colab/Jupyter)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    setup_logging(level=level)
    
    # Print để confirm
    print(f"[Logger] Logging configured: level={logging.getLevelName(level)}")