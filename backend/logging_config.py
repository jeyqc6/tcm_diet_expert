#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-wide logging setup.

ENGINEERING §6.2: JSON lines with `trace_id` / `stage` / … . `trace_id` is
read from the ContextVar in `tracing.py` by a logging.Filter — call sites
write `logger.info("…")` and the field appears automatically.

Call `configure_logging()` from the FastAPI lifespan (not at import time).
`LOG_FORMAT=json` (default, production/containers) or `text` (local tty).
`LOG_LEVEL` defaults to INFO.

Health-bearing extra fields go through the same `redact.py` helpers used
for Langfuse, so a forgotten call site cannot leak diet-log text into
docker logs.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

from backend.env import load_env
from backend.observability.redact import redact_log_payload
from backend.observability.tracing import current_trace_id

_STANDARD_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "thread",
        "threadName",
        "exc_info",
        "exc_text",
        "stack_info",
        "message",
        "asctime",
        "taskName",
        "trace_id",
    }
)


class _TraceIdFilter(logging.Filter):
    """Stamp every record with the request `trace_id` (or None outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = current_trace_id()
        return True


def _record_extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
    }


class _JsonFormatter(logging.Formatter):
    """One JSON object per line. Merges `stage_log` payloads so fields stay flat."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", None),
        }
        message = record.getMessage()
        merged = False
        if message.startswith("{") and message.endswith("}"):
            try:
                parsed = json.loads(message)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict) and "stage" in parsed:
                payload.update(parsed)
                merged = True
        if not merged:
            payload["message"] = message
        extras = _record_extras(record)
        for key, value in extras.items():
            payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact_log_payload(payload), ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, "trace_id", None) or "-"
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        line = (
            f"{timestamp} {record.levelname} {record.name} "
            f"[trace_id={trace_id}] {record.getMessage()}"
        )
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(*, stream: TextIO | None = None, force: bool = True) -> None:
    """Attach one handler to the `diet_expert` logger. Idempotent when force=False."""
    load_env()
    fmt = (os.environ.get("LOG_FORMAT") or "json").strip().lower()
    level_name = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("diet_expert")
    if logger.handlers and not force:
        return

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.addFilter(_TraceIdFilter())
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _TextFormatter())
    logger.handlers = [handler]
    logger.setLevel(level)
    # Avoid duplicating through the root logger (uvicorn / basicConfig).
    logger.propagate = False
