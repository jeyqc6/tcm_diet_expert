#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP server protocol errors — distinct from tool business logic failures."""
from __future__ import annotations

from dataclasses import dataclass

from backend.exceptions import AuthorizationError
from backend.mcp_server.roles import CallerRole


@dataclass
class ToolNotDeclaredError(AuthorizationError):
    """Raised when a role invokes a tool that was not declared for its session."""

    role: CallerRole
    tool_name: str

    def __str__(self) -> str:
        return (
            f"Tool {self.tool_name!r} is not declared for role {self.role.value!r} "
            f"(protocol-layer rejection)"
        )
