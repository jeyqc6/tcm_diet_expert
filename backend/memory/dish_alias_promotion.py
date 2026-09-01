#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个人菜品简称的计数与晋升逻辑，与查找逻辑分文件，便于单独测阈值行为。

设计依据：docs/ARCHITECTURE.md §4.2
决策依据：docs/DECISIONS.md D27 修订一
⚠️ 晋升阈值（建议3次）是待实测调整的常数，留成配置项

⚠️ **一处简化，已如实标注**：设计原文是"若人在环确认(§4.2 步骤4)时用户未修改
这次的拆解结果 → 计数+1"——`/api/chat` 是单发 SSE 请求，没有"先展示拆解结果、
等用户回复确认/修改"这样的二次往返(那需要一个独立的编辑/纠正端点，目前不存在)，
所以本模块把"这次 LLM 兜底调用成功返回了结果"本身当作计数条件，不等价于
"用户确认没有修改过"。如果以后做"编辑一条已记录的饮食"功能，这里应该改成只在
用户没有编辑那条记录时才计数——目前这个更强的条件无法判断，用更宽松的条件
先让晋升机制跑起来，比完全不计数更接近设计意图。
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn
from backend.memory.dish_decomposition import DishMatch, normalize_phrase

DEFAULT_PROMOTION_THRESHOLD = 3


@dataclass(frozen=True)
class PromotionResult:
    ok: bool
    hit_count: int
    promoted: bool
    detail: str = ""


def record_llm_fallback_hit(
    user_id: str,
    normalized_phrase_input: str,
    matches: tuple[DishMatch, ...],
    *,
    threshold: int = DEFAULT_PROMOTION_THRESHOLD,
    dsn: str | None = None,
) -> PromotionResult:
    """LLM 兜底成功解析出结果后调用——UPSERT 计数，达到阈值时晋升
    (`promoted_at = now()`)。空匹配(LLM 说"这段话里其实没有食物")不计数，
    没有信息量的候选没必要占一行。"""
    phrase = normalize_phrase(normalized_phrase_input)
    if not phrase or not matches:
        return PromotionResult(ok=False, hit_count=0, promoted=False, detail="empty phrase or no matches")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold!r}")

    if psycopg2 is None:
        raise RuntimeError("psycopg2 not installed")
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        raise RuntimeError("DIET_EXPERT_PG_DSN not configured")

    dishes_json = [
        {
            "dish": m.dish,
            "ingredients": list(m.ingredients),
            "tcm_nature": m.tcm_nature,
            "allergens": list(m.allergens),
        }
        for m in matches
    ]
    ingredients = sorted({ing for m in matches for ing in m.ingredients})

    conn = psycopg2.connect(resolved_dsn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_dish_aliases (user_id, normalized_phrase, dishes, ingredients, hit_count)
            VALUES (%s, %s, %s, %s, 1)
            ON CONFLICT (user_id, normalized_phrase) DO UPDATE
                SET hit_count = user_dish_aliases.hit_count + 1,
                    dishes = EXCLUDED.dishes,
                    ingredients = EXCLUDED.ingredients
            WHERE user_dish_aliases.promoted_at IS NULL
            RETURNING hit_count, promoted_at
            """,
            (user_id, phrase, psycopg2.extras.Json(dishes_json), ingredients),
        )
        row = cur.fetchone()
        if row is None:
            # 已经晋升过的行：ON CONFLICT ... WHERE 子句让这次更新不生效，
            # 说明这条别名早就晋升了，不需要再计数——查一下当前状态给调用方看。
            cur.execute(
                "SELECT hit_count, promoted_at FROM user_dish_aliases "
                "WHERE user_id = %s AND normalized_phrase = %s",
                (user_id, phrase),
            )
            row = cur.fetchone()
        hit_count, promoted_at = row
        promoted_now = False
        if promoted_at is None and hit_count >= threshold:
            cur.execute(
                "UPDATE user_dish_aliases SET promoted_at = now() "
                "WHERE user_id = %s AND normalized_phrase = %s",
                (user_id, phrase),
            )
            promoted_now = True
        conn.commit()
        cur.close()
    finally:
        conn.close()

    return PromotionResult(
        ok=True,
        hit_count=hit_count,
        promoted=promoted_now or promoted_at is not None,
    )
