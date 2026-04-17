"""Structured JSON logging with correlation id support."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any, Literal

import structlog

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def bind_correlation_id(value: str | None) -> None:
    """Bind a correlation id for the current async/thread context."""
    _correlation_id.set(value)


def _inject_correlation_id(
    _logger: logging.Logger,
    _name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    cid = _correlation_id.get()
    if cid is not None:
        event_dict.setdefault("correlation_id", cid)
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    fmt: Literal["json", "console"] = "json",
) -> None:
    """Configure structlog + stdlib logging with a single renderer.

    Safe to call multiple times; idempotent.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        force=True,
    )

    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _inject_correlation_id,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger."""
    return structlog.get_logger(name) if name else structlog.get_logger()
