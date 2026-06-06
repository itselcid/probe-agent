"""Structured JSON logging for ProbeAgent.

Uses ``structlog`` to produce machine-readable JSON log lines with
timestamps, log level, caller information, and arbitrary key/value context.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure ``structlog`` for JSON-formatted, production-ready logging.

    Call this once at application startup — typically from the CLI entry-point
    in :mod:`probe_agent.main`.

    Args:
        level: Python logging level name (``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``, ``CRITICAL``).  Case-insensitive.

    Example::

        from probe_agent.logging_setup import setup_logging

        setup_logging("DEBUG")
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the standard library root logger so third-party libraries
    # (httpx, docker, etc.) are captured as well.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Shared processors for both structlog-native and stdlib loggers.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ],
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Format exceptions as strings inside the JSON payload.
            structlog.processors.format_exc_info,
            # Render the final event as a JSON line.
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound ``structlog`` logger.

    Args:
        name: Optional logger name.  If ``None``, structlog will infer it
            from the calling module.

    Returns:
        A :class:`structlog.stdlib.BoundLogger` that outputs JSON.
    """
    return structlog.get_logger(name)
