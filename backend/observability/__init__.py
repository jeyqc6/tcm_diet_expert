"""Langfuse tracing facade (ENGINEERING.md §6).

Public surface is the wrappers in `tracing.py`. Call sites should not import
the Langfuse SDK directly — that keeps tests and local runs working when
Langfuse is not configured.
"""
from backend.observability.cost import estimate_cost_usd
from backend.observability.tracing import (
    current_trace_id,
    flush_tracing,
    is_tracing_enabled,
    observation,
    reset_tracing_backend,
    stage_log,
    start_request_trace,
    update_current,
    use_memory_backend,
)

__all__ = [
    "current_trace_id",
    "estimate_cost_usd",
    "flush_tracing",
    "is_tracing_enabled",
    "observation",
    "reset_tracing_backend",
    "stage_log",
    "start_request_trace",
    "update_current",
    "use_memory_backend",
]
