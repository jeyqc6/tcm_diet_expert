#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe wrapper for non-agent MCP tool calls (log_review, profile writes, …)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.mcp_server.server import McpSession

logger = logging.getLogger("diet_expert.mcp_server.safe_call")


@dataclass(frozen=True)
class SafeToolResult:
    ok: bool
    result: Any = None
    error_type: str | None = None
    detail: str | None = None


def safe_call_tool(
    session: McpSession,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> SafeToolResult:
    """Call an MCP tool; on failure log and return a structured error instead of raising."""
    try:
        result = session.call_tool(name, arguments)
        return SafeToolResult(ok=True, result=result)
    except Exception as exc:
        logger.warning(
            "MCP tool call failed · tool=%s · error=%s",
            name,
            exc,
            exc_info=True,
        )
        return SafeToolResult(
            ok=False,
            error_type=type(exc).__name__,
            detail=str(exc),
        )
