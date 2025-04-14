"""
Logging utilities for WDBX.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, Union


def configure_logging(
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
    file_log_level: Optional[Union[int, str]] = None,
) -> None:
    """
    Configure logging for WDBX.

    Args:
        log_level: Logging level for console output
        log_file: Path to log file (None for no file logging)
        log_format: Format string for log messages
        date_format: Format string for log timestamps
        max_file_size: Maximum size of log file before rotation (bytes)
        backup_count: Number of backup log files to keep
        console: Whether to log to console
        file_log_level: Logging level for file output (defaults to log_level if None)
    """
    # Convert string log level to numeric if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    if file_log_level is None:
        file_log_level = log_level
    elif isinstance(file_log_level, str):
        file_log_level = getattr(logging, file_log_level.upper())

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(log_level, file_log_level))

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(
        name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Args:
        name: Logger name
        level: Logging level (None to use root logger level)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log messages.
    """

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """
        Initialize the logger adapter.

        Args:
            logger: Base logger
            extra: Extra context information
        """
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the log message by adding context information.

        Args:
            msg: Log message
            kwargs: Logging keyword arguments

        Returns:
            Tuple of (processed_message, kwargs)
        """
        # Add context information to message
        context_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
        if context_str:
            msg = f"{msg} [{context_str}]"

        return msg, kwargs


def get_contextual_logger(
    name: str,
    context: Dict[str, Any],
    level: Optional[Union[int, str]] = None
) -> LoggerAdapter:
    """
    Get a logger with context information.

    Args:
        name: Logger name
        context: Context information to add to log messages
        level: Logging level (None to use root logger level)

    Returns:
        LoggerAdapter instance
    """
    logger = get_logger(name, level)
    return LoggerAdapter(logger, context)


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.
    """

    def __init__(
        self,
        logger: Union[logging.Logger, LoggerAdapter],
        total: int,
        description: str = "Progress",
        log_interval: int = 10,
        level: int = logging.INFO
    ):
        """
        Initialize the progress logger.

        Args:
            logger: Base logger
            total: Total number of items to process
            description: Description of the operation
            log_interval: How often to log progress (in percentage points)
            level: Logging level for progress messages
        """
        self.logger = logger
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.level = level

        self.count = 0
        self.last_logged_percent = 0
        self.start_time = None

    def __enter__(self):
        """Start tracking progress."""
        import time
        self.start_time = time.time()
        self.logger.log(self.level, f"{self.description} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log final progress."""
        import time
        if exc_type is None:
            duration = time.time() - self.start_time
            self.logger.log(
                self.level, f"{self.description} completed: {self.count}/{self.total} items in {duration:.2f}s")
        else:
            self.logger.error(f"{self.description} failed: {exc_val}")

    def update(self, increment: int = 1):
        """
        Update progress.

        Args:
            increment: Number of items processed
        """
        import time

        self.count += increment

        if self.total <= 0:
            # No total count, just log the current count
            self.logger.log(
                self.level, f"{self.description}: {self.count} items")
            return

        percent = (self.count * 100) // self.total

        # Log if we've passed another interval or reached 100%
        if percent >= self.last_logged_percent + self.log_interval or self.count >= self.total:
            self.last_logged_percent = percent

            # Calculate time remaining
            if self.count > 0 and self.start_time is not None:
                elapsed = time.time() - self.start_time
                items_per_sec = self.count / elapsed

                if items_per_sec > 0:
                    remaining_items = self.total - self.count
                    remaining_secs = remaining_items / items_per_sec

                    self.logger.log(
                        self.level,
                        f"{self.description}: {self.count}/{self.total} ({percent}%) - {items_per_sec:.2f} items/s, {remaining_secs:.2f}s remaining"
                    )
                else:
                    self.logger.log(
                        self.level,
                        f"{self.description}: {self.count}/{self.total} ({percent}%)"
                    )
            else:
                self.logger.log(
                    self.level,
                    f"{self.description}: {self.count}/{self.total} ({percent}%)"
                )
