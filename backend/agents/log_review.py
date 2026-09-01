#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`log_review`(记录回顾)分支——不经过 SubAgent/LLM，直接查 `query_diet_log`、
确定性格式化。

设计依据：docs/ARCHITECTURE.md §5.3
2026-08-28：从 api/main.py 拆出，纯粹搬文件，不改变任何函数签名/行为。
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import AsyncIterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from api.schemas import ChatRequest
from backend.agents.sse import chunk_text, sse_event
from backend.agents.user_context import UserProfileContext
from backend.mcp_server.roles import CallerRole
from backend.mcp_server.server import DietExpertMcpServer
from backend.i18n import meal_type_label, t
from backend.observability.tracing import observation, update_current
from backend.observability.redact import redact_text

# Longer English phrases first so "the day before yesterday" is not eaten by "yesterday".
_TIME_RANGE_KEYWORDS = (
    "the day before yesterday",
    "前天",
    "yesterday",
    "昨天",
    "昨日",
    "today",
    "今天",
    "今日",
    "this week",
    "本周",
    "这周",
    "last week",
    "上周",
)
_RECENT_DAYS_RE = re.compile(
    r"(最近\s*\d+\s*天|(?:the )?last\s*\d+\s*days?)",
    re.IGNORECASE,
)


def _extract_time_range(message: str, locale: str = "zh") -> str:
    """从自由文本里抠一个 `query_diet_log` 认识的 time_range 表达——粗糙但对
    "最小闭环"够用；真正做好这件事需要更完整的时间表达解析（比如"上周三"这种），
    不在本任务范围内，默认兜底到"今天"。中英双语关键词；返回值与
    query_diet_log.parse_time_range 的别名表对齐。"""
    m = _RECENT_DAYS_RE.search(message)
    if m:
        return m.group(0)
    lowered = message.lower()
    for kw in _TIME_RANGE_KEYWORDS:
        if kw.lower() in lowered:
            return kw
    return "today" if locale == "en" else "今天"


def _resolve_review_tz(profile: UserProfileContext | None) -> ZoneInfo:
    """和 `log_write.py` 的 `_resolve_log_tz()` 同样的三层兜底(`user_profile.timezone`
    > `DIET_EXPERT_TZ` 环境变量 > 硬编码默认值)，这里重新实现一遍而不是共用一个
    函数——同 `_resolve_log_tz()` 自己的既有理由：调用方已经查过一次 `profile`，
    没必要为同一个请求多打一次数据库去问 `query_diet_log.py` 的 `_get_tz()`。"""
    for candidate in (
        profile.timezone if profile else None,
        os.environ.get("DIET_EXPERT_TZ"),
        "Asia/Shanghai",
    ):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except ZoneInfoNotFoundError:
            continue
    return ZoneInfo("Asia/Shanghai")


def _format_logged_at(logged_at: str, tz: ZoneInfo) -> str:
    """`query_diet_log` 返回的 `logged_at` 是 `row[0].isoformat()`(见
    `query_diet_log.py` `_fetch_entries()`)——数据库驱动带回来的 tzinfo 只是
    Postgres 连接会话的显示时区，跟这个用户自己的 `timezone` 是两回事，直接
    原样展示会出现一个和用户所在地对不上、还带 6 位微秒的时间戳(2026-08-31
    用户反馈发现"感觉有点奇怪")。这里转换到查询用户自己的时区再格式化——存的
    绝对时刻不受影响，只是换一种人看得懂、跟自己时区对得上的方式呈现。解析
    失败(理论上不该发生，防御性兜底)时原样返回，不让格式化本身变成新的故障点。"""
    try:
        parsed = datetime.fromisoformat(logged_at)
    except ValueError:
        return logged_at
    return parsed.astimezone(tz).strftime("%Y-%m-%d %H:%M")


def _format_entry_food_summary(entry: dict, locale: str = "zh") -> str:
    """回顾一条记录时应该直接展示"吃了什么"(菜名列表)，不是原样吐用户当时
    那句完整的话——`raw_input` 是"帮我记录一下，中午吃了麻婆豆腐"这种带指令性
    前缀的原始输入，是给 idempotency/审计用的存档字段，不是给人看的"吃了什么"
    摘要(2026-08-31 用户反馈发现：回顾时看到的是一整句原话，而不是菜名)。
    `dishes`(见 log_write.py 写入时的结构)才是结构化的菜名列表，这里优先用它；
    没有 dishes(比如历史上更早、这个字段还不存在时写入的行)才退回 raw_input，
    好过展示空字符串。"""
    dishes = entry.get("dishes") or []
    list_separator = ", " if locale == "en" else "、"
    names = list_separator.join(d.get("dish", "") for d in dishes if d.get("dish"))
    return names or entry.get("raw_input", "")


def _format_diet_log_summary(raw: dict, tz: ZoneInfo, locale: str = "zh") -> str:
    """确定性格式化，不经过任何 LLM 调用——§5.3 原文"核查pass简化为回答里出现
    的记录条目确实来自 diet_log 查询结果"，直接从查询结果本身拼文本，天然满足
    这条简化版核查,不需要额外跑一次 verify()。"""
    entries = raw.get("entries") or []
    time_range = raw.get("time_range", "")
    if not entries:
        return t("log_review.empty", locale, time_range=time_range)
    lines = [t("log_review.header", locale, time_range=time_range, count=len(entries))]
    for e in entries:
        meal_type = meal_type_label(e.get("meal_type"), locale)
        separator = ": " if locale == "en" else "："
        logged_at = _format_logged_at(e.get("logged_at", ""), tz)
        lines.append(f"- {logged_at} {meal_type}{separator}{_format_entry_food_summary(e, locale)}")
    return "\n".join(lines)


async def stream_log_review(
    request: ChatRequest,
    server: DietExpertMcpServer,
    trace_id: str,
    profile: UserProfileContext | None = None,
) -> AsyncIterator[str]:
    with observation("log_review", as_type="span", input={"message": redact_text(request.message)}):
        session = server.open_session(CallerRole.ROUTER, user_id=request.user_id)
        time_range = _extract_time_range(request.message, request.locale)
        raw = session.call_tool("query_diet_log", {"time_range": time_range, "aggregation": "raw"})
        tz = _resolve_review_tz(profile)
        text = _format_diet_log_summary(raw, tz, locale=request.locale)
        update_current(output={"time_range": time_range, "text": redact_text(text)})
    for chunk in chunk_text(text):
        yield sse_event("token", {"text": chunk})
    yield sse_event("done", {"trace_id": trace_id})
