#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_recipes_by_ingredients：recipes 表 GIN &&/@> 查询。

设计依据：docs/ARCHITECTURE.md §2.2
决策依据：docs/DECISIONS.md D24
"""
from __future__ import annotations

from typing import Any

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn


def query_recipes_by_ingredients(
    ingredients: list[str],
    match: str = "all",
    limit: int = 20,
    *,
    dsn: str | None = None,
) -> list[dict[str, Any]]:
    """Find recipes whose ingredient array matches the requested ingredients."""
    if match not in {"any", "all"}:
        raise ValueError("match must be 'any' or 'all'")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    cleaned: list[str] = []
    seen: set[str] = set()
    for ingredient in ingredients:
        if not isinstance(ingredient, str):
            raise ValueError("ingredients must contain only strings")
        value = ingredient.strip()
        if value and value not in seen:
            cleaned.append(value)
            seen.add(value)
    if not cleaned:
        raise ValueError("ingredients must contain at least one non-empty value")

    if psycopg2 is None:
        raise RuntimeError("需要 psycopg2：pip install psycopg2-binary")
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        raise RuntimeError(
            "没有连接串。请设置 DIET_EXPERT_PG_DSN，或通过部署环境配置数据库连接。"
        )

    operator = "&&" if match == "any" else "@>"
    sql = f"""
        SELECT id, name, dish, description, ingredients, instructions, author, source
        FROM recipes
        WHERE ingredients {operator} %s
        ORDER BY id ASC
        LIMIT %s
    """

    conn = psycopg2.connect(resolved_dsn)
    try:
        cur = conn.cursor()
        cur.execute(sql, (cleaned, limit))
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    return [
        {
            "id": row[0],
            "name": row[1],
            "dish": row[2],
            "description": row[3],
            "ingredients": row[4],
            "instructions": row[5],
            "author": row[6],
            "source": row[7],
        }
        for row in rows
    ]
