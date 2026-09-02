#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按用户体质/目标从 `conflict_rules` 表查出相关规则，喂给调和层(D14 的输入之一)。

设计依据：docs/ARCHITECTURE.md §5.2 步骤5("命中的 conflict_rules(≤2k)")
决策依据：docs/DECISIONS.md D23(关系表建模)

⚠️ 这个模块存在的理由：`backend/agents/reconciliation.py` 的 `reconcile()` 一直
支持 `matched_rules` 参数(D14 设计如此)，但 `api/main.py` 调 `reconcile_subagent_results()`
时从没传过这个参数——40 条真实规则已经灌进 Postgres(`db/load_conflict_rules.py`)，
真实请求路径却完全没有"按体质/目标查出相关规则"这一步。本模块补上查询本身，
`api/main.py` 里对应的改动补上调用。

匹配口径按 §5.2 步骤5原文——"命中的 conflict_rules"，也是当前 BUILD_PLAN 现状
说明里"按体质/目标从表里查出相关规则"这句话本身的字面意思:一条规则只要
`applicable_constitutions`/`applicable_goals` 和用户画像有任意重叠就算命中，
**不**按当前对话具体提到的菜品/话题做语义匹配——那需要先从自由文本抽取"用户在
问什么食物"，是 §4.2 菜品拆解那层还没做的能力(`dish_decomposition.py` 仍是
`NotImplementedError`)，本次不越界去做。

只有 40 行数据(V1 规模)，直接全表拉回来在 Python 里过滤/排序/截断，不做
"WHERE 数组重叠"这类下推到 SQL 的优化——省去在 SQL 和 Python 两处各写一遍
"什么叫命中"的重复定义，40 行的全表扫描不构成性能问题。
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn

logger = logging.getLogger("diet_expert.agents.conflict_rules_lookup")

DEFAULT_LIMIT = 5

# §5.2 步骤5 给调和层的 budget 是 ≤2k token，命中规则按置信度优先截断，
# 不是随手取前 N 条——confidence 越高的规则更值得挤进这个预算。
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _confidence_sort_key(rule: dict[str, Any]) -> tuple[int, str]:
    return (_CONFIDENCE_RANK.get(rule.get("confidence"), 3), str(rule.get("rule_id", "")))


def select_matched_rules(
    rows: list[dict[str, Any]],
    constitutions: list[str] | None,
    goal_tags: list[str] | None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """纯函数：给定候选规则行，按用户体质/目标过滤 + 按置信度排序 + 截断。
    不连数据库，独立出来是为了这条"什么算命中"的业务逻辑能被直接测试。
    """
    constitution_set = set(constitutions or ())
    goal_set = set(goal_tags or ())
    if not constitution_set and not goal_set:
        return []
    matched = [
        r
        for r in rows
        if set(r.get("applicable_constitutions") or ()) & constitution_set
        or set(r.get("applicable_goals") or ()) & goal_set
    ]
    matched.sort(key=_confidence_sort_key)
    return matched[:limit]


def fetch_matched_conflict_rules(
    constitutions: list[str] | None,
    goal_tags: list[str] | None,
    dsn: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """查不到/连不上库返回空列表——调和层本来就能接受 `matched_rules=None`/`[]`
    (`reconciliation.py` `_format_matched_rules` 的既有降级文案)，读规则失败
    不该让 `/api/chat` 请求失败。"""
    if not constitutions and not goal_tags:
        return []
    if psycopg2 is None:
        return []
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        return []
    try:
        conn = psycopg2.connect(resolved_dsn)
    except Exception:
        logger.warning("fetch_matched_conflict_rules: connect failed", exc_info=True)
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT rule_id, topic, relation, resolution, confidence,
                   applicable_constitutions, applicable_goals
            FROM conflict_rules
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
    except Exception:
        logger.warning("fetch_matched_conflict_rules: query failed", exc_info=True)
        return []
    finally:
        conn.close()
    return select_matched_rules(rows, constitutions, goal_tags, limit=limit)
