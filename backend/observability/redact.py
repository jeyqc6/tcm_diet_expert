#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redact health-related content before it is sent to Langfuse.

PRD: observability data is self-hosted, redacted before ingest, retained 30 days.
ENGINEERING §5: health data must not appear in logs in plaintext.

Default is metadata-only (lengths, keys, ids). Full I/O is opt-in via
`LANGFUSE_CAPTURE_IO=1` for a self-hosted Langfuse that the operator trusts.
Even then, diet_log entries and write_memory payloads stay stripped — those
are the highest-sensitivity records this app handles.
"""
from __future__ import annotations

import os
from typing import Any

from backend.env import load_env

# Tool results that are personal health records even when capture_io is on.
_ALWAYS_STRIP_RESULT = frozenset({"query_diet_log", "write_memory"})
# Argument fields that are identifiers / routing, not free-text health notes.
_SAFE_ARG_KEYS = frozenset(
    {
        "top_k",
        "limit",
        "aggregation",
        "time_range",
        "match",
        "city",
        "date",
        "include_recent_days",
        "category",
    }
)
_RETRIEVAL_TOOLS = frozenset({"retrieve_tcm", "retrieve_nutrition"})


def capture_io_enabled() -> bool:
    load_env()
    raw = os.environ.get("LANGFUSE_CAPTURE_IO", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def redact_text(text: str | None, *, preview_chars: int = 0) -> dict[str, Any]:
    """Replace a user/model string with length (+ optional short preview)."""
    if text is None:
        return {"chars": 0, "redacted": True}
    payload: dict[str, Any] = {"chars": len(text), "redacted": not capture_io_enabled()}
    if capture_io_enabled():
        payload["text"] = text
    elif preview_chars > 0 and text:
        payload["preview"] = text[:preview_chars]
    return payload


def redact_messages(messages: list[dict] | None) -> list[dict[str, Any]]:
    """Keep roles / tool names; drop message bodies unless capture_io is on."""
    out: list[dict[str, Any]] = []
    for m in messages or []:
        item: dict[str, Any] = {"role": m.get("role")}
        if "name" in m:
            item["name"] = m.get("name")
        if capture_io_enabled():
            item["content"] = m.get("content")
            if m.get("tool_calls"):
                item["tool_calls"] = m["tool_calls"]
        else:
            content = m.get("content")
            item["content"] = redact_text(content if isinstance(content, str) else None)
            if m.get("tool_calls"):
                item["tool_call_names"] = [
                    c.get("name") for c in m["tool_calls"] if isinstance(c, dict)
                ]
        out.append(item)
    return out


def redact_tool_args(tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    arguments = arguments or {}
    if capture_io_enabled() and tool_name not in _ALWAYS_STRIP_RESULT:
        return dict(arguments)
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in _SAFE_ARG_KEYS:
            safe[key] = value
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = {"count": len(value), "redacted": True}
        else:
            safe[key] = redact_text(str(value) if value is not None else None)
    return safe


def summarize_tool_result(tool_name: str, result: Any) -> dict[str, Any]:
    """Describe a tool result without copying retrieved text or diet_log rows."""
    if tool_name in _ALWAYS_STRIP_RESULT:
        if isinstance(result, dict):
            summary: dict[str, Any] = {
                "type": "dict",
                "keys": sorted(result.keys()),
                "redacted": True,
            }
            if "entries" in result and isinstance(result["entries"], list):
                summary["entry_count"] = len(result["entries"])
            if "time_range" in result:
                summary["time_range"] = result.get("time_range")
            if "ok" in result:
                summary["ok"] = result.get("ok")
            return summary
        return {"type": type(result).__name__, "redacted": True}

    if isinstance(result, list):
        summary = {"type": "list", "count": len(result)}
        if tool_name in _RETRIEVAL_TOOLS:
            ids = [
                item.get("source_id")
                for item in result
                if isinstance(item, dict) and item.get("source_id")
            ]
            summary["source_ids"] = ids
        elif capture_io_enabled():
            summary["value"] = result
        return summary

    if isinstance(result, dict):
        return {"type": "dict", "keys": sorted(result.keys())}

    return {"type": type(result).__name__}


# Keys that may carry user/model free text. Operational fields (stage,
# latency_ms, tokens, cost_est, …) are not in this set and pass through.
_SENSITIVE_LOG_KEYS = frozenset(
    {
        "content",
        "messages",
        "input",
        "payload",
        "text",
        "query",
        "raw_input",
        "matched",
        "preview",
        "arguments",
        "result",
        "secret_note",
        "profile_updates",
        "final_text",
    }
)


def redact_log_payload(value: Any) -> Any:
    """Walk a structured log payload and redact known health-content keys.

    Same helpers as Langfuse spans (`redact_text` / `redact_messages`), so
    a log line cannot leak diet-log / user-message text just because a
    call site forgot to redact. Already-redacted dicts (`{"redacted": True}`)
    are left alone.
    """
    if isinstance(value, dict):
        if value.get("redacted") is True:
            return value
        out: dict[str, Any] = {}
        for key, inner in value.items():
            if key in _SENSITIVE_LOG_KEYS:
                if key == "messages" and isinstance(inner, list):
                    out[key] = redact_messages(inner)
                elif isinstance(inner, dict) and inner.get("redacted") is True:
                    out[key] = inner
                elif isinstance(inner, str) or inner is None:
                    out[key] = redact_text(inner)
                else:
                    out[key] = redact_text(str(inner) if inner is not None else None)
            else:
                out[key] = redact_log_payload(inner)
        return out
    if isinstance(value, list):
        return [redact_log_payload(item) for item in value]
    return value

