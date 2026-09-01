#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Langfuse tracing facade.

ENGINEERING.md §6: every request has a `trace_id` that flows through HTTP
headers, structured logs, and Langfuse spans. BUILD_PLAN phase 6 success
criterion: opening one trace shows the route, per-stage latency, and cost.

This module is the only place that talks to the Langfuse SDK. Call sites use
`observation()` / `start_request_trace()` / `update_current()`.

Behaviour:
  - No keys / LANGFUSE_ENABLED=0 / missing package → no-op spans (tests, local).
  - Tests inject `use_memory_backend()` and assert the span tree without network.
  - Health content is redacted unless LANGFUSE_CAPTURE_IO=1 (see redact.py).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from backend.env import load_env
from backend.observability.redact import redact_log_payload

logger = logging.getLogger("diet_expert.observability")

SpanType = Literal[
    "span",
    "generation",
    "agent",
    "tool",
    "retriever",
    "chain",
    "guardrail",
]

_trace_id: ContextVar[str | None] = ContextVar("diet_expert_trace_id", default=None)
_span_stack: ContextVar[tuple[str, ...]] = ContextVar("diet_expert_span_stack", default=())

# Backend override is process-global on purpose: FastAPI TestClient runs the
# ASGI app in a portal whose ContextVar copy would not see a test-local
# override. trace_id / span stack stay ContextVars because they are set
# *inside* the request.
_override_backend: Any = None
_cached_backend: Any = None
_warned_missing_sdk = False


@dataclass
class RecordedSpan:
    """In-memory span used by tests. Mirrors the subset we send to Langfuse."""

    name: str
    as_type: str
    parent: str | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage_details: dict[str, int] | None = None
    cost_details: dict[str, float] | None = None
    model: str | None = None
    level: str = "DEFAULT"
    status_message: str | None = None
    error: str | None = None
    duration_ms: float = 0.0
    trace_id: str | None = None
    closed: bool = False


class _Span:
    """Minimal span protocol shared by no-op / memory / Langfuse adapters."""

    def update(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
        model: str | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        return None


class _NoOpSpan(_Span):
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _MemorySpan(_Span):
    def __init__(self, record: RecordedSpan, backend: "MemoryBackend") -> None:
        self.record = record
        self._backend = backend
        self._t0 = 0.0
        self._stack_token = None

    def __enter__(self) -> _MemorySpan:
        self._t0 = time.perf_counter()
        stack = _span_stack.get()
        self.record.parent = stack[-1] if stack else None
        self.record.trace_id = current_trace_id()
        self._stack_token = _span_stack.set(stack + (self.record.name,))
        self._backend.spans.append(self.record)
        return self

    def update(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
        model: str | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if input is not None:
            self.record.input = input
        if output is not None:
            self.record.output = output
        if metadata:
            self.record.metadata.update(metadata)
        if usage_details is not None:
            self.record.usage_details = usage_details
        if cost_details is not None:
            self.record.cost_details = cost_details
        if model is not None:
            self.record.model = model
        if level is not None:
            self.record.level = level
        if status_message is not None:
            self.record.status_message = status_message

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.record.duration_ms = (time.perf_counter() - self._t0) * 1000.0
        self.record.closed = True
        if exc is not None:
            self.record.level = "ERROR"
            self.record.error = str(exc)
        if self._stack_token is not None:
            _span_stack.reset(self._stack_token)
        return False


class MemoryBackend:
    """Records spans in process for assertions. No network."""

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    def observation(self, name: str, *, as_type: str = "span", **kwargs: Any) -> _MemorySpan:
        record = RecordedSpan(
            name=name,
            as_type=as_type,
            input=kwargs.get("input"),
            output=kwargs.get("output"),
            metadata=dict(kwargs.get("metadata") or {}),
            model=kwargs.get("model"),
            usage_details=kwargs.get("usage_details"),
            cost_details=kwargs.get("cost_details"),
        )
        return _MemorySpan(record, self)


class _LangfuseSpan(_Span):
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def update(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
        model: str | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if input is not None:
            payload["input"] = input
        if output is not None:
            payload["output"] = output
        if metadata:
            payload["metadata"] = metadata
        if usage_details is not None:
            payload["usage_details"] = usage_details
        if cost_details is not None:
            payload["cost_details"] = cost_details
        if model is not None:
            payload["model"] = model
        if level is not None:
            payload["level"] = level
        if status_message is not None:
            payload["status_message"] = status_message
        if payload:
            self._inner.update(**payload)


class LangfuseBackend:
    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def observation(self, name: str, *, as_type: str = "span", **kwargs: Any):
        trace_id = kwargs.pop("trace_id", None)
        lf_kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
        for key in (
            "input",
            "output",
            "metadata",
            "model",
            "model_parameters",
            "usage_details",
            "cost_details",
            "level",
            "status_message",
        ):
            if key in kwargs and kwargs[key] is not None:
                lf_kwargs[key] = kwargs[key]
        if trace_id:
            lf_kwargs["trace_context"] = {"trace_id": trace_id}
        with self._client.start_as_current_observation(**lf_kwargs) as inner:
            yield _LangfuseSpan(inner)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:  # noqa: BLE001 — shutdown must not fail the process
            logger.warning("Langfuse flush failed", exc_info=True)


class _NoOpBackend:
    def observation(self, name: str, *, as_type: str = "span", **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def flush(self) -> None:
        return None


def _env_flag(name: str) -> str | None:
    load_env()
    raw = os.environ.get(name)
    return raw.strip() if raw else None


def _explicitly_disabled() -> bool:
    raw = (_env_flag("LANGFUSE_ENABLED") or "").lower()
    return raw in {"0", "false", "no", "off"}


def _keys_present() -> bool:
    return bool(_env_flag("LANGFUSE_PUBLIC_KEY") and _env_flag("LANGFUSE_SECRET_KEY"))


def is_tracing_enabled() -> bool:
    if _override_backend is not None:
        return not isinstance(_override_backend, _NoOpBackend)
    if _explicitly_disabled():
        return False
    return _keys_present()


def current_trace_id() -> str | None:
    return _trace_id.get()


def use_memory_backend() -> MemoryBackend:
    """Test helper: capture spans in memory, never contact Langfuse."""
    global _override_backend, _cached_backend
    backend = MemoryBackend()
    _override_backend = backend
    _cached_backend = backend
    return backend


def reset_tracing_backend() -> None:
    """Drop injected backends and the cached Langfuse client. Used by tests."""
    global _override_backend, _cached_backend
    _override_backend = None
    _cached_backend = None


def _build_langfuse_backend() -> LangfuseBackend | None:
    global _warned_missing_sdk
    try:
        from langfuse import Langfuse
    except ImportError:
        if not _warned_missing_sdk:
            logger.warning(
                "LANGFUSE_PUBLIC_KEY is set but the langfuse package is not installed; "
                "tracing is disabled. pip install langfuse"
            )
            _warned_missing_sdk = True
        return None
    kwargs: dict[str, Any] = {
        "public_key": _env_flag("LANGFUSE_PUBLIC_KEY"),
        "secret_key": _env_flag("LANGFUSE_SECRET_KEY"),
    }
    base_url = _env_flag("LANGFUSE_BASE_URL") or _env_flag("LANGFUSE_HOST")
    if base_url:
        kwargs["base_url"] = base_url
    environment = _env_flag("LANGFUSE_TRACING_ENVIRONMENT") or _env_flag("LANGFUSE_ENVIRONMENT")
    if environment:
        kwargs["environment"] = environment
    return LangfuseBackend(Langfuse(**kwargs))


def _get_backend() -> Any:
    global _cached_backend
    if _override_backend is not None:
        return _override_backend
    if _cached_backend is not None:
        return _cached_backend
    if _explicitly_disabled() or not _keys_present():
        _cached_backend = _NoOpBackend()
        return _cached_backend
    built = _build_langfuse_backend()
    _cached_backend = built if built is not None else _NoOpBackend()
    return _cached_backend


@contextmanager
def observation(
    name: str,
    *,
    as_type: SpanType = "span",
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
    trace_id: str | None = None,
    level: str | None = None,
) -> Iterator[_Span]:
    """Open a child span (or generation / tool / …) under the current trace."""
    backend = _get_backend()
    kwargs: dict[str, Any] = {
        "as_type": as_type,
        "input": input,
        "output": output,
        "metadata": metadata,
        "model": model,
        "usage_details": usage_details,
        "cost_details": cost_details,
        "level": level,
    }
    if trace_id:
        kwargs["trace_id"] = trace_id
    with backend.observation(name, **kwargs) as span:
        yield span


@contextmanager
def start_request_trace(
    trace_id: str,
    *,
    name: str = "chat",
    session_id: str | None = None,
    user_id: str | None = None,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[_Span]:
    """Root span for one HTTP request. Sets `current_trace_id()` even if no-op."""
    token = _trace_id.set(trace_id)
    merged = dict(metadata or {})
    merged.setdefault("trace_id", trace_id)
    if session_id:
        merged.setdefault("session_id", session_id)
    if user_id:
        merged.setdefault("user_id", user_id)
    try:
        with observation(
            name,
            as_type="span",
            input=input,
            metadata=merged,
            trace_id=trace_id,
        ) as span:
            backend = _get_backend()
            if isinstance(backend, LangfuseBackend):
                try:
                    from langfuse import propagate_attributes

                    with propagate_attributes(
                        user_id=user_id,
                        session_id=session_id,
                        metadata={"trace_id": trace_id},
                    ):
                        yield span
                        return
                except Exception:  # noqa: BLE001 — never fail the request for tracing
                    logger.warning("Langfuse propagate_attributes failed", exc_info=True)
            yield span
    finally:
        _trace_id.reset(token)


def update_current(**kwargs: Any) -> None:
    """Update the span opened by the innermost `observation()` / request trace.

    Memory backend: look at the last still-open span on the stack by name.
    Langfuse: `update_current_span` / `update_current_generation`.
    No-op: drop.
    """
    as_type = kwargs.pop("as_type", None)
    backend = _get_backend()
    if isinstance(backend, MemoryBackend):
        stack = _span_stack.get()
        if not stack:
            return
        current_name = stack[-1]
        for record in reversed(backend.spans):
            if record.name == current_name and not record.closed:
                _MemorySpan(record, backend).update(**kwargs)
                return
        return
    if isinstance(backend, LangfuseBackend):
        try:
            if as_type == "generation":
                backend._client.update_current_generation(**kwargs)
            else:
                backend._client.update_current_span(**kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("Langfuse update_current failed", exc_info=True)


def flush_tracing() -> None:
    backend = _get_backend()
    flush = getattr(backend, "flush", None)
    if callable(flush):
        flush()


def stage_log(log: logging.Logger, stage: str, **fields: Any) -> None:
    """ENGINEERING §6.2 JSON line: trace_id / stage / latency_ms / tokens / cost_est / …"""
    payload = redact_log_payload(
        {
            "trace_id": current_trace_id(),
            "stage": stage,
            **{k: v for k, v in fields.items() if v is not None},
        }
    )
    log.info("%s", json.dumps(payload, ensure_ascii=False, default=str))
