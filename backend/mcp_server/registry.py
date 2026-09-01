#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool catalog: JSON Schema + handler registration for the MCP server."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.mcp_server.roles import (
    ALL_TOOLS,
    TOOL_QUERY_DIET_LOG,
    TOOL_QUERY_RECIPES,
    TOOL_QUERY_WEATHER,
    TOOL_RETRIEVE_NUTRITION,
    TOOL_RETRIEVE_TCM,
    TOOL_WRITE_MEMORY,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def default_tool_definitions() -> dict[str, ToolDefinition]:
    """Build the full catalog. Handlers are the tools/*.py entry points."""
    from backend.mcp_server.tools import (
        query_diet_log,
        query_recipes,
        query_weather,
        retrieve_nutrition,
        retrieve_tcm,
        write_memory,
    )

    defs: dict[str, ToolDefinition] = {
        TOOL_RETRIEVE_TCM: ToolDefinition(
            name=TOOL_RETRIEVE_TCM,
            description="Vector search over TCM knowledge_chunks (domain=tcm).",
            input_schema=_object_schema(
                {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "filters": {"type": ["object", "null"]},
                },
                required=["query"],
            ),
            handler=retrieve_tcm.retrieve_tcm,
        ),
        TOOL_RETRIEVE_NUTRITION: ToolDefinition(
            name=TOOL_RETRIEVE_NUTRITION,
            description="Vector search over nutrition knowledge_chunks (domain=nutrition).",
            input_schema=_object_schema(
                {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "filters": {"type": ["object", "null"]},
                },
                required=["query"],
            ),
            handler=retrieve_nutrition.retrieve_nutrition,
        ),
        TOOL_QUERY_RECIPES: ToolDefinition(
            name=TOOL_QUERY_RECIPES,
            description="Find recipes by ingredient list (recipes table GIN query).",
            input_schema=_object_schema(
                {
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "match": {"type": "string", "enum": ["any", "all"], "default": "all"},
                    "limit": {"type": "integer", "default": 20},
                },
                required=["ingredients"],
            ),
            handler=query_recipes.query_recipes_by_ingredients,
        ),
        TOOL_QUERY_WEATHER: ToolDefinition(
            name=TOOL_QUERY_WEATHER,
            description="Weather via Open-Meteo (cached); fallback to solar-term table.",
            input_schema=_object_schema(
                {
                    "city": {"type": "string"},
                    "date": {"type": ["string", "null"]},
                    "include_recent_days": {"type": "integer", "default": 3},
                },
                required=["city"],
            ),
            handler=query_weather.query_weather,
        ),
        TOOL_QUERY_DIET_LOG: ToolDefinition(
            name=TOOL_QUERY_DIET_LOG,
            description="Read-only diet_log query (relative dates supported).",
            input_schema=_object_schema(
                {
                    "time_range": {"type": "string"},
                    "aggregation": {
                        "type": "string",
                        "enum": ["by_ingredient", "by_property", "by_meal_type", "raw"],
                        "default": "raw",
                    },
                    "limit": {"type": ["integer", "null"]},
                },
                required=["time_range"],
            ),
            handler=query_diet_log.query_diet_log,
        ),
        TOOL_WRITE_MEMORY: ToolDefinition(
            name=TOOL_WRITE_MEMORY,
            description="Write user_profile (critical) or diet_log (daily_log); router only.",
            input_schema=_object_schema(
                {
                    "category": {"type": "string", "enum": ["critical", "daily_log"]},
                    "payload": {"type": "object"},
                    "idempotency_key": {"type": ["string", "null"]},
                },
                required=["category", "payload"],
            ),
            handler=write_memory.write_memory,
        ),
    }
    missing = ALL_TOOLS - defs.keys()
    if missing:
        raise RuntimeError(f"Tool registry incomplete: {sorted(missing)}")
    return defs
