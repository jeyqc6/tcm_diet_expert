#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
category=critical → 写 user_profile（需人在环确认）；category=daily_log → 写 diet_log（需 idempotency_key）。
仅中枢 agent 持有（ARCHITECTURE.md §2.3）——权限由 MCP session 保证，本模块不做 role 检查。

设计依据：docs/ARCHITECTURE.md §2.2/§2.3

⚠️ **状态(2026-08-26)**：`category="critical"`(写 `user_profile`)已实现——这是为了
让首次引导(`backend/onboarding/flow.py`)的结果真的能落库，不是"引导流程写完了
反而没地方存"。这里不再重复加一层"人在环确认"：引导流程本身每一步都是直接
问用户、拿用户的原话作答(体质那一步还有显式的确认/修改/跳过子步骤，§11.2)，
这就是 ARCHITECTURE 说的"人在环确认"本身，不是这个函数还要再做一次。

⚠️ **`category="daily_log"`(写 `diet_log`)已实现**(2026-08-26)——依赖
`backend/memory/dish_decomposition.py`(自由文本拆解成菜品/配料/中医食性,已完成)。
调用方(`api/main.py` 的 log_write 分支)负责组装 `payload`(`raw_input`/`dishes`/
`ingredients`/`food_properties`/`meal_type`/`logged_at`)并算好 `idempotency_key`
传进来——ENGINEERING §1.2 写路径幂等键规格是 `hash(user_id, logged_at, raw_input_hash)`，
"怎么算这个 hash"是调用方的职责，本函数只负责"给了 idempotency_key 就按它去重"。
用 `INSERT ... ON CONFLICT (idempotency_key) DO NOTHING`：如果这次调用和之前某次
产生了同一个 idempotency_key(比如客户端重试了同一条消息)，不会插入第二行，
`WriteResult.detail` 里会标注这是一次去重命中，不是失败。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn

DEFAULT_USER_ID = "default_user"

_VALID_MEAL_TYPES = frozenset({"早餐", "午餐", "晚餐", "夜宵", "下午茶", "加餐", "未知"})
_REQUIRED_DAILY_LOG_FIELDS = frozenset(
    {"raw_input", "dishes", "ingredients", "food_properties", "meal_type", "logged_at"}
)

# user_profile 里"关键事实"允许通过 write_memory(critical) 写入的列——
# 有意排除 id/updated_at 这类不该由调用方直接指定的列。
_CRITICAL_COLUMNS = frozenset(
    {
        "constitution",
        "constitution_secondary",
        "constitution_source",
        "constitution_confirmed_at",
        "allergens",
        "supplements",
        "goal_tags",
        "preferences",
        "city",
        "timezone",
        "onboarding_done",
        "locale",
    }
)

# 需要用 psycopg2.extras.Json() 包一层的 JSONB 列——TEXT[]/TEXT 列直接传值，
# 数组/复合结构的列(supplements 是 [{"name":...,"dose":...}] 这样的列表，
# preferences 是 dict)需要显式序列化，否则 psycopg2 会按数组字面量处理。
_JSON_COLUMNS = frozenset({"preferences", "supplements"})


@dataclass(frozen=True)
class WriteResult:
    ok: bool
    table: str
    user_id: str
    fields_written: tuple[str, ...]
    detail: str = ""
    duplicate: bool = False
    row_id: int | None = None


def _write_critical(payload: dict[str, Any], user_id: str, dsn: str | None) -> WriteResult:
    unknown = set(payload) - _CRITICAL_COLUMNS
    if unknown:
        raise ValueError(f"write_memory(critical): unknown user_profile field(s) {sorted(unknown)}")
    # None 值不写(表示"这一步引导没有收集到值"，不是"显式要清空这一列")——
    # 部分更新只覆盖真的拿到了新值的列，同一行别的已有字段不受影响。
    fields = {k: v for k, v in payload.items() if v is not None}
    if not fields:
        return WriteResult(ok=False, table="user_profile", user_id=user_id, fields_written=(), detail="empty payload")

    if psycopg2 is None:
        raise RuntimeError("psycopg2 not installed")
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        raise RuntimeError("DIET_EXPERT_PG_DSN not configured")

    columns = sorted(fields)
    values: list[Any] = []
    for col in columns:
        v = fields[col]
        values.append(psycopg2.extras.Json(v) if col in _JSON_COLUMNS else v)

    insert_cols = ["user_id", *columns]
    placeholders = ", ".join(["%s"] * len(insert_cols))
    update_assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
    sql = (
        f"INSERT INTO user_profile ({', '.join(insert_cols)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (user_id) DO UPDATE SET {update_assignments}, updated_at = now()"
    )

    conn = psycopg2.connect(resolved_dsn)
    try:
        cur = conn.cursor()
        cur.execute(sql, [user_id, *values])
        conn.commit()
        cur.close()
    finally:
        conn.close()

    return WriteResult(ok=True, table="user_profile", user_id=user_id, fields_written=tuple(columns))


def _write_daily_log(
    payload: dict[str, Any], idempotency_key: str | None, user_id: str, dsn: str | None
) -> WriteResult:
    if not idempotency_key:
        raise ValueError("write_memory(daily_log): idempotency_key is required")
    missing = _REQUIRED_DAILY_LOG_FIELDS - set(payload)
    if missing:
        raise ValueError(f"write_memory(daily_log): missing field(s) {sorted(missing)}")
    unknown = set(payload) - _REQUIRED_DAILY_LOG_FIELDS
    if unknown:
        raise ValueError(f"write_memory(daily_log): unknown field(s) {sorted(unknown)}")
    meal_type = payload["meal_type"]
    if meal_type not in _VALID_MEAL_TYPES:
        raise ValueError(f"write_memory(daily_log): invalid meal_type {meal_type!r}")
    logged_at = payload["logged_at"]
    if not isinstance(logged_at, datetime):
        raise ValueError("write_memory(daily_log): logged_at must be a datetime")

    if psycopg2 is None:
        raise RuntimeError("psycopg2 not installed")
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        raise RuntimeError("DIET_EXPERT_PG_DSN not configured")

    conn = psycopg2.connect(resolved_dsn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO diet_log
                (user_id, logged_at, meal_type, raw_input, dishes, ingredients,
                 food_properties, idempotency_key)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (
                user_id,
                logged_at,
                meal_type,
                payload["raw_input"],
                psycopg2.extras.Json(payload["dishes"]),
                list(payload["ingredients"]),
                list(payload["food_properties"]),
                idempotency_key,
            ),
        )
        row = cur.fetchone()
        if row is not None:
            conn.commit()
            cur.close()
            return WriteResult(
                ok=True, table="diet_log", user_id=user_id,
                fields_written=tuple(sorted(_REQUIRED_DAILY_LOG_FIELDS)), row_id=row[0],
            )
        # ON CONFLICT DO NOTHING 没有返回行 = 命中了已存在的 idempotency_key，
        # 不是失败——查出那一行的 id 报给调用方，说明"这次记录之前已经写过了"。
        cur.execute("SELECT id FROM diet_log WHERE idempotency_key = %s", (idempotency_key,))
        existing = cur.fetchone()
        cur.close()
        return WriteResult(
            ok=True, table="diet_log", user_id=user_id, fields_written=(),
            detail="duplicate idempotency_key, no new row written", duplicate=True,
            row_id=existing[0] if existing else None,
        )
    finally:
        conn.close()


def write_memory(
    category: str,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
    *,
    user_id: str = DEFAULT_USER_ID,
    dsn: str | None = None,
) -> WriteResult:
    if category == "critical":
        return _write_critical(payload, user_id, dsn)
    if category == "daily_log":
        return _write_daily_log(payload, idempotency_key, user_id, dsn)
    raise ValueError(f"unknown write_memory category: {category!r}")
