#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP server skeleton with role-scoped tool visibility (ARCHITECTURE.md §2.3).

Single server process; each caller opens a session bound to one CallerRole.
list_tools / call_tool enforce the whitelist before any tool handler runs —
unauthorized tools are not listed and cannot be invoked (D7 protocol boundary).

Usage (in-process, for Agent Loop / tests):
    server = DietExpertMcpServer()
    session = server.open_session(CallerRole.TCM_SUBAGENT)
    session.list_tools()
    session.call_tool("retrieve_tcm", {"query": "气虚质春季饮食"})
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from backend.mcp_server.exceptions import ToolNotDeclaredError
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.roles import ROLE_TOOL_WHITELIST, CallerRole
from backend.observability.redact import redact_tool_args, summarize_tool_result
from backend.observability.tracing import observation, stage_log, update_current

logger = logging.getLogger("diet_expert.mcp_server")


@dataclass
class UnauthorizedToolAttempt:
    role: CallerRole
    tool_name: str


# 这两个工具的 handler 接受 `user_id` 关键字参数(query_diet_log.py/write_memory.py)，
# 但 `user_id` 不进它们的公开 JSON Schema(registry.py 里注释过：V1 单用户时不需要，
# 现在多用户了也不能让 LLM 自己填这个参数——它填的话就是在猜/编一个用户身份，
# 不是"哪个真实用户在跟它对话"这件事该由模型判断的)。真实值从 session 打开时
# 传入的 `user_id` 上取，`call_tool` 在派发前替调用方补上，SubAgent 自己感知不到。
_USER_SCOPED_TOOLS = frozenset({"query_diet_log", "write_memory"})


@dataclass
class McpSession:
    """One client session = one role's declared tool subset, bound to one user_id."""

    _server: DietExpertMcpServer
    role: CallerRole
    user_id: str = "default_user"
    _allowed: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self._allowed = ROLE_TOOL_WHITELIST[self.role]

    def list_tools(self) -> list[ToolDefinition]:
        """Return only tools declared for this role (others do not exist here)."""
        return [
            self._server.tools[name]
            for name in sorted(self._allowed)
            if name in self._server.tools
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Dispatch to handler after protocol-layer whitelist check."""
        as_type = "retriever" if name in {"retrieve_tcm", "retrieve_nutrition"} else "tool"
        t0 = time.perf_counter()
        with observation(
            f"tool.{name}",
            as_type=as_type,
            input=redact_tool_args(name, arguments),
            metadata={"role": self.role.value, "tool": name},
        ):
            if name not in self._allowed:
                self._server.record_unauthorized(self.role, name)
                update_current(level="WARNING", output={"unauthorized": True})
                stage_log(
                    logger,
                    "tool",
                    tool=name,
                    role=self.role.value,
                    unauthorized=True,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
                raise ToolNotDeclaredError(role=self.role, tool_name=name)
            tool = self._server.tools.get(name)
            if tool is None:
                raise KeyError(f"Unknown tool {name!r} (registry bug)")
            call_args = dict(arguments or {})
            if name in _USER_SCOPED_TOOLS:
                call_args.setdefault("user_id", self.user_id)
            result = tool.handler(**call_args)
            update_current(output=summarize_tool_result(name, result))
            stage_log(
                logger,
                "tool",
                tool=name,
                role=self.role.value,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
            return result


class DietExpertMcpServer:
    """Local MCP server: role-scoped sessions, shared tool registry."""

    def __init__(
        self,
        tools: dict[str, ToolDefinition] | None = None,
    ) -> None:
        self.tools = tools if tools is not None else default_tool_definitions()
        self.unauthorized_attempts: list[UnauthorizedToolAttempt] = []

    def open_session(self, role: CallerRole, user_id: str = "default_user") -> McpSession:
        return McpSession(_server=self, role=role, user_id=user_id)

    def record_unauthorized(self, role: CallerRole, tool_name: str) -> None:
        attempt = UnauthorizedToolAttempt(role=role, tool_name=tool_name)
        self.unauthorized_attempts.append(attempt)
        logger.warning(
            "MCP unauthorized tool call: role=%s tool=%s",
            role.value,
            tool_name,
        )

    def tool_names_for_role(self, role: CallerRole) -> list[str]:
        return sorted(ROLE_TOOL_WHITELIST[role])
