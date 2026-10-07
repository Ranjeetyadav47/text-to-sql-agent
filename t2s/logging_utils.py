"""Structured JSON logging. One line per event, machine-readable, ask_id on every line.

Pipe it into anything (Cloud Logging, Datadog, `jq`). Example line:
{"ts": "...", "level": "INFO", "event": "guard.blocked", "ask_id": "...", "rule": "statement_not_allowed"}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class EventLogger:
    """Thin wrapper so call sites read as `log.info("step.ok", run_id=..., step=...)`."""

    def __init__(self, name: str = "t2s"):
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        self._log.log(level, event, extra={"fields": fields})

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)


def configure_logging(level: str = "INFO", stream=None, quiet: bool = False) -> None:
    root = logging.getLogger("t2s")
    root.handlers.clear()
    root.setLevel(logging.CRITICAL if quiet else getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False
