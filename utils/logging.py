"""Logging wrapper with stdout formatter and Sentry placeholder.

This module provides a centralized logging configuration that outputs
to stdout with consistent formatting and includes a placeholder for
Sentry integration in production environments.
"""

import logging
import sys
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    sentry_dsn: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        level: Logging level (default: INFO).
        format_string: Custom format string. If None, uses default.
        sentry_dsn: Optional Sentry DSN for error tracking (placeholder).

    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger = logging.getLogger("assignment1")
    logger.setLevel(level)
    logger.addHandler(handler)

    # Placeholder for Sentry integration
    if sentry_dsn:
        # TODO: Integrate Sentry SDK when needed
        # import sentry_sdk
        # sentry_sdk.init(dsn=sentry_dsn)
        logger.info("Sentry integration placeholder (not configured)")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name. If None, returns root assignment1 logger.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"assignment1.{name}")
    return logging.getLogger("assignment1")


