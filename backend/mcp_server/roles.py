#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caller roles and per-role tool whitelists (ARCHITECTURE.md §2.3).

Enforcement lives in server.py / McpSession — individual tool modules must NOT
check caller role; unauthorized tools are absent from list_tools and rejected on call.
"""
from __future__ import annotations

from enum import Enum

# Canonical tool names (ARCHITECTURE.md §2.2)
TOOL_RETRIEVE_TCM = "retrieve_tcm"
TOOL_RETRIEVE_NUTRITION = "retrieve_nutrition"
TOOL_QUERY_RECIPES = "query_recipes_by_ingredients"
TOOL_QUERY_WEATHER = "query_weather"
TOOL_QUERY_DIET_LOG = "query_diet_log"
TOOL_WRITE_MEMORY = "write_memory"

ALL_TOOLS: frozenset[str] = frozenset(
    {
        TOOL_RETRIEVE_TCM,
        TOOL_RETRIEVE_NUTRITION,
        TOOL_QUERY_RECIPES,
        TOOL_QUERY_WEATHER,
        TOOL_QUERY_DIET_LOG,
        TOOL_WRITE_MEMORY,
    }
)


class CallerRole(str, Enum):
    """Session identity — one MCP client session per role (ARCHITECTURE §2.3 方式 A)."""

    ROUTER = "router"
    TCM_SUBAGENT = "tcm_subagent"
    NUTRITION_SUBAGENT = "nutrition_subagent"
    RECONCILIATION = "reconciliation"
    VERIFICATION = "verification"
    # D1 的 B2 ablation baseline(docs/DECISIONS.md D1"验证方式")：单 agent、
    # 两个检索工具同时暴露在同一上下文里，不拆两个 SubAgent、不经过独立的
    # 调和层——只用于 evals/run_b2_ablation.py 的实验对比，不接进 `/api/chat`
    # 生产路径(见 backend/agents/single_agent_baseline.py 模块文档)。
    SINGLE_AGENT_B2 = "single_agent_b2"


ROLE_TOOL_WHITELIST: dict[CallerRole, frozenset[str]] = {
    CallerRole.ROUTER: ALL_TOOLS,
    CallerRole.TCM_SUBAGENT: frozenset(
        {TOOL_RETRIEVE_TCM, TOOL_QUERY_WEATHER, TOOL_QUERY_DIET_LOG}
    ),
    CallerRole.NUTRITION_SUBAGENT: frozenset(
        {TOOL_RETRIEVE_NUTRITION, TOOL_QUERY_DIET_LOG, TOOL_QUERY_RECIPES}
    ),
    CallerRole.RECONCILIATION: frozenset(),
    CallerRole.VERIFICATION: frozenset(),
    # TCM + Nutrition 两个 SubAgent 各自工具的并集，去掉 write_memory(只读，
    # 同 SubAgent 一贯不持有写权限的原则，§2.3)。
    CallerRole.SINGLE_AGENT_B2: frozenset(
        {
            TOOL_RETRIEVE_TCM,
            TOOL_RETRIEVE_NUTRITION,
            TOOL_QUERY_WEATHER,
            TOOL_QUERY_DIET_LOG,
            TOOL_QUERY_RECIPES,
        }
    ),
}
