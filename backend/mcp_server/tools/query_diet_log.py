#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按时间范围/聚合维度查询 diet_log，含相对日期解析。

设计依据：docs/ARCHITECTURE.md §2.2
roadmap：阶段 4.2 任务 3

⚠️ 时区基准（DECISIONS.md 待决问题表里的那一条，2026-08-26 已解决）：`user_profile`
现在有 `timezone` 字段（IANA 时区名，如 "Asia/Shanghai"，db/schema.sql），`_get_tz()`
按优先级解析：user_profile.timezone > `DIET_EXPERT_TZ` 环境变量 > 硬编码默认值
Asia/Shanghai。三层都是为了同一件事有退路：用户没填资料时不报错、部署环境没配
环境变量时也不报错，但只要用户填过 timezone 就以它为准——不从 `city` 字段自动
反查时区（地理编码在时区边界/夏令时上容易出错，且 V1 用户量小，直接问比猜更可靠，
呼应 PRD §10.2 人在环原则）。

聚合维度覆盖 ARCHITECTURE §2.2 现有规格四个（by_ingredient/by_property/by_nutrient/
raw）外加 by_meal_type（2026-08-26 经用户确认后新增，ARCHITECTURE.md §2.2 已同步
更新签名，不是本文件单方面扩展接口）。
by_nutrient 明确没做——需要把 ingredients 关联到营养素数据（FDC 或知识库），这条
关联现在不存在，装作算出一个数字比直接报错更危险，所以是 NotImplementedError
而不是返回一个编造的空结果。
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backend.env import get_pg_dsn, load_env  # noqa: E402

DEFAULT_TZ_NAME = "Asia/Shanghai"

_AGGREGATIONS = {"raw", "by_ingredient", "by_property", "by_meal_type", "by_nutrient"}

_RELATIVE_DAY_OFFSETS = {
    "今天": 0, "今日": 0, "today": 0,
    "昨天": 1, "昨日": 1, "yesterday": 1,
    "前天": 2, "the day before yesterday": 2,
}
_RECENT_DAYS_PATTERN = re.compile(
    r"(?:最近\s*(\d+)\s*天|(?:the )?last\s*(\d+)\s*days?)",
    re.IGNORECASE,
)
_THIS_WEEK_ALIASES = frozenset({"本周", "这周", "this week"})
_LAST_WEEK_ALIASES = frozenset({"上周", "last week"})


def _fetch_user_timezone(user_id: str, dsn: str | None) -> str | None:
    """查 user_profile.timezone；表/行不存在、字段为空、甚至连不上库，
    都当"这一层没有答案"处理，交给调用方(_get_tz)按优先级继续往下退，
    不在这里抛错——查时区兜底链路本身不应该因为画像还没建好就让整个工具用不了。
    """
    if psycopg2 is None:
        return None
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        return None
    try:
        conn = psycopg2.connect(resolved_dsn)
    except Exception:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT timezone FROM user_profile WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
    except Exception:
        return None
    finally:
        conn.close()
    return row[0] if row else None


def _get_tz(user_id: str = "default_user", dsn: str | None = None) -> ZoneInfo:
    """优先级：user_profile.timezone > DIET_EXPERT_TZ 环境变量 > Asia/Shanghai。
    任何一层的值不是合法 IANA 时区名（比如画像里手滑存了个错字）就跳过它，
    不整个报错——错误的时区兜底应该是"退到下一层"，不是"这个工具彻底用不了"。
    """
    candidates = [_fetch_user_timezone(user_id, dsn)]
    load_env()
    candidates.append(os.environ.get("DIET_EXPERT_TZ"))
    candidates.append(DEFAULT_TZ_NAME)

    for name in candidates:
        if not name:
            continue
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            continue
    # 理论上到不了这里——DEFAULT_TZ_NAME 本身必须是合法时区名，是最后一道保险
    raise RuntimeError(f"内置默认时区 {DEFAULT_TZ_NAME!r} 本身不合法，这是代码 bug 不是配置问题")


@dataclass
class TimeSpan:
    start: datetime  # tz-aware，含
    end: datetime  # tz-aware，不含（半开区间）


def _day_start(d: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(d, dtime.min, tzinfo=tz)


def _day_span(d: date, tz: ZoneInfo) -> TimeSpan:
    start = _day_start(d, tz)
    return TimeSpan(start=start, end=start + timedelta(days=1))


def parse_time_range(
    expr: str,
    *,
    now: datetime | None = None,
    tz: ZoneInfo | None = None,
    user_id: str = "default_user",
    dsn: str | None = None,
) -> TimeSpan:
    """把相对/绝对日期表达式解析成一个半开时间区间 [start, end)。

    支持：今天/昨天/前天（及 today/yesterday/the day before yesterday）、
    最近N天 / last N days（含今天，N天前 00:00 到现在这一刻的次日 00:00）、
    本周/这周/this week（本周一到现在）、上周/last week（上周一到本周一）、
    显式 YYYY-MM-DD 单日。
    解析不了就明确报错，不悄悄猜一个默认范围——"猜错的日期范围"比"报错"更危险，
    调和层/SubAgent 拿到错的时间窗口做出的判断是错的但看起来正常。

    `tz` 未显式传入时才去查 user_profile/环境变量兜底（`_get_tz(user_id, dsn)`）——
    调用方已经知道要用哪个时区时（比如上层已经查过一次）不用重复查库。
    """
    tz = tz or _get_tz(user_id, dsn)
    now = (now or datetime.now(tz)).astimezone(tz)
    today = now.date()
    expr = expr.strip()
    expr_key = expr.lower()

    if expr_key in _RELATIVE_DAY_OFFSETS:
        day = today - timedelta(days=_RELATIVE_DAY_OFFSETS[expr_key])
        return _day_span(day, tz)

    m = _RECENT_DAYS_PATTERN.fullmatch(expr)
    if m:
        n = int(m.group(1) or m.group(2))
        if n <= 0:
            raise ValueError(f"time_range={expr!r}：天数必须是正整数")
        start_day = today - timedelta(days=n - 1)
        return TimeSpan(start=_day_start(start_day, tz), end=_day_start(today + timedelta(days=1), tz))

    if expr_key in _THIS_WEEK_ALIASES:
        monday = today - timedelta(days=today.weekday())
        return TimeSpan(start=_day_start(monday, tz), end=_day_start(today + timedelta(days=1), tz))

    if expr_key in _LAST_WEEK_ALIASES:
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(days=7)
        return TimeSpan(start=_day_start(last_monday, tz), end=_day_start(this_monday, tz))

    try:
        day = date.fromisoformat(expr)
    except ValueError:
        pass
    else:
        return _day_span(day, tz)

    raise ValueError(
        f"无法解析 time_range={expr!r}。支持：今天/昨天/前天/本周/上周/最近N天/"
        "today/yesterday/this week/last week/last N days/YYYY-MM-DD"
    )


def _fetch_entries(user_id: str, span: TimeSpan, dsn: str | None) -> list[dict[str, Any]]:
    if psycopg2 is None:
        raise RuntimeError("需要 psycopg2：pip install psycopg2-binary")
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        raise RuntimeError("没有连接串。传 dsn 参数，或在 .env / 环境变量里设置 DIET_EXPERT_PG_DSN。")

    conn = psycopg2.connect(resolved_dsn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT logged_at, meal_type, raw_input, dishes, ingredients, food_properties
            FROM diet_log
            WHERE user_id = %s AND logged_at >= %s AND logged_at < %s
            ORDER BY logged_at DESC
            """,
            (user_id, span.start, span.end),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    return [
        {
            "logged_at": row[0].isoformat(),
            "meal_type": row[1],
            "raw_input": row[2],
            "dishes": row[3],
            "ingredients": row[4],
            "food_properties": row[5],
        }
        for row in rows
    ]


def _top_counts(items, limit: int | None) -> list[dict[str, Any]]:
    counter = Counter(items)
    return [{"value": k, "count": v} for k, v in counter.most_common(limit)]


def query_diet_log(
    time_range: str,
    aggregation: str = "raw",
    limit: int | None = None,
    *,
    user_id: str = "default_user",
    dsn: str | None = None,
    now: datetime | None = None,
    tz: ZoneInfo | None = None,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """MCP 工具签名对齐 ARCHITECTURE §2.2：`(time_range, aggregation, limit)`。
    `user_id` 不进公开 JSON Schema（V1 单用户，见 backend/mcp_server/registry.py），
    `dsn`/`now`/`tz`/`entries` 是测试注入点，不是业务代码会传的参数。

    ⚠️ `tz` 不传时才会去查 `user_profile`/环境变量（`_get_tz()` 会真的连数据库）——
    单测传了 `now`/`entries` 但没传 `tz` 的话，仍然会触发一次真实 DB 查询，
    这不是"注入了假数据就完全隔离"，务必也传 `tz` 才是真正离线的单测。
    """
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"aggregation={aggregation!r} 不在支持范围内：{sorted(_AGGREGATIONS)}")
    if aggregation == "by_nutrient":
        raise NotImplementedError(
            "by_nutrient 需要把 ingredients 关联到营养素数据（FDC 或知识库），这条关联"
            "目前不存在，不是本次实现范围——见本文件模块文档。装作算出一个数字比"
            "直接报错更危险，不做假实现。"
        )

    tz = tz or _get_tz(user_id, dsn)
    span = parse_time_range(time_range, now=now, tz=tz)

    if entries is None:
        entries = _fetch_entries(user_id, span, dsn)

    result: dict[str, Any] = {
        "time_range": time_range,
        "resolved_range": {"start": span.start.isoformat(), "end": span.end.isoformat()},
        "aggregation": aggregation,
        "count": len(entries),
    }

    if aggregation == "raw":
        # `if limit` 会把 `limit=0` 误判成"没传 limit"（0 是 falsy）,导致
        # limit=0 时反而返回全部条目——和 `_top_counts()` 用 `most_common(limit)`
        # 处理 limit=0 时正确返回空列表的行为不一致。用 `is not None` 才是
        # "调用方到底有没有传 limit"的正确判断。
        result["entries"] = entries[:limit] if limit is not None else entries
    elif aggregation == "by_ingredient":
        result["breakdown"] = _top_counts(
            (ing for e in entries for ing in (e.get("ingredients") or [])), limit
        )
    elif aggregation == "by_property":
        result["breakdown"] = _top_counts(
            (p for e in entries for p in (e.get("food_properties") or [])), limit
        )
    elif aggregation == "by_meal_type":
        result["breakdown"] = _top_counts((e.get("meal_type") for e in entries), limit)

    return result
