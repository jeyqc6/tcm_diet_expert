#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`log_write`(记录)分支——§4.2 三级查找(dish_ingredient_map → user_dish_aliases
→ LLM 兜底) + 过敏原即时警示(不阻断写入,记录的是已经发生的事) +
write_memory(daily_log) 幂等写入。

没有单独的"人在环确认"往返：§4.2 步骤4原文"展示拆解结果,用户确认或修改"，
但 `/api/chat` 是单发 SSE、没有编辑一条已记录条目的端点(不在这次范围)，
所以这里是"先幂等写入,再把写入结果原样展示给用户"，展示本身就是确认的
呈现面；如果以后要支持"用户看到拆解错了、要求修改"，需要新增一个编辑端点，
到时候这里的写入时机可能要跟着调整，不是本次能一并解决的。

2026-08-28：从 api/main.py 拆出，纯粹搬文件，不改变任何函数签名/行为。
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from api.schemas import ChatRequest
from backend.agents.clarification import ClarificationStore, PendingClarification
from backend.agents.routing import CompleteFn, RouteBranch
from backend.agents.sse import chunk_text, sse_event
from backend.agents.user_context import UserProfileContext
from backend.i18n import meal_type_label, t
from backend.mcp_server.roles import CallerRole
from backend.mcp_server.server import DietExpertMcpServer
from backend.memory.dish_alias_promotion import record_llm_fallback_hit
from backend.memory.dish_decomposition import (
    SOURCE_LLM_FALLBACK,
    MealDecomposition,
    decompose_meal,
    match_global_table,
    normalize_phrase,
)

logger = logging.getLogger("diet_expert.agents.log_write")

# ⚠️ 真实跑通时发现的问题：最初版本要求"早上吃"这种时间词+动词连在一起才算
# 命中，"早上刚喝了燕麦牛奶"这类中间插了别的字的说法就漏检，退到按服务器当前
# 时刻兜底——如果用户是"补记"(比如下午才想起来把早上的记一下)，会被错误地
# 记成"下午茶"。改成单独的时间词就命中，不要求紧跟在动词前面。
_MEAL_TYPE_KEYWORDS = (
    ("早餐", ("早餐", "早饭", "早上", "早晨", "清晨", "breakfast")),
    ("午餐", ("午餐", "午饭", "中午", "lunch")),
    ("晚餐", ("晚餐", "晚饭", "晚上", "傍晚", "dinner", "supper")),
    ("夜宵", ("夜宵", "宵夜", "半夜", "凌晨", "late night", "midnight snack")),
    ("下午茶", ("下午茶", "下午", "afternoon tea")),
    ("加餐", ("加餐", "零食", "snack")),
)


def _infer_meal_type(message: str, now_local: datetime) -> str:
    """§4.2:"早餐/午餐/晚餐/夜宵/下午茶/加餐/未知,按关键词/时段确定性推断,
    不额外调模型"。关键词优先；用户没提到是哪顿饭时按当前时段兜底——大概率
    就是"现在吃的这顿"，比留"未知"更有信息量。"""
    lowered = (message or "").lower()
    for meal_type, keywords in _MEAL_TYPE_KEYWORDS:
        for kw in keywords:
            haystack = lowered if kw.isascii() else message
            if kw in haystack:
                return meal_type
    hour = now_local.hour
    if 5 <= hour < 10:
        return "早餐"
    if 10 <= hour < 14:
        return "午餐"
    if 14 <= hour < 17:
        return "下午茶"
    if 17 <= hour < 21:
        return "晚餐"
    if hour >= 21 or hour < 2:
        return "夜宵"
    return "未知"


_LOG_DAY_OFFSETS = {
    "今天": 0, "今日": 0, "today": 0,
    "昨天": 1, "昨日": 1, "yesterday": 1,
    "前天": 2, "the day before yesterday": 2,
}


def _resolve_log_tz(profile: UserProfileContext | None) -> ZoneInfo:
    """和 `query_diet_log.py` 的 `_get_tz()` 同样的三层兜底(`user_profile.timezone`
    > `DIET_EXPERT_TZ` 环境变量 > 硬编码默认值)，这里重新实现一遍而不是直接调用
    `_get_tz()`——那个函数自己会再查一次 `user_profile`，这里已经有 `profile`
    在手（调用方已经查过一次），没必要为同一个请求多打一次数据库。"""
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


def _resolve_logged_at(message: str, tz: ZoneInfo) -> datetime:
    """粗糙但对"记录"分支够用——识别"今天/昨天/前天"这几个最常见的补记表达，
    按天数回退当前时刻；没提到就是"现在"。比 `query_diet_log.py` 的
    `parse_time_range` 简单得多：那个要解析出一个查询用的时间*范围*，这里只
    需要给新记录一个具体的时间*点*，同一简化原则不重复展开。"""
    now = datetime.now(tz)
    lowered = message.lower()
    for kw, offset in _LOG_DAY_OFFSETS.items():
        haystack = lowered if kw.isascii() else message
        if kw in haystack:
            return now - timedelta(days=offset)
    return now


def _compute_idempotency_key(user_id: str, logged_at: datetime, raw_input: str) -> str:
    """ENGINEERING §1.2:`idempotency_key = hash(user_id, logged_at, raw_input_hash)`。
    `logged_at` 截断到分钟级再算 hash——同一条消息在同一分钟内被重复提交(比如
    客户端网络重试)应该判定为同一次记录，不产生第二行；这层截断是 `/api/chat`
    本身没有客户端幂等 token 时唯一能做到的"事实上大概率不重复"，不是精确的
    请求级幂等(那需要 `ChatRequest` 带一个稳定的请求 id，目前没有，不在这次
    范围内解决)。"""
    minute_bucket = logged_at.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
    raw_hash = hashlib.sha256(raw_input.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{user_id}|{minute_bucket}|{raw_hash}".encode()).hexdigest()


def _format_log_write_confirmation(
    decomposition: MealDecomposition, meal_type: str, logged_at: datetime, duplicate: bool,
    locale: str = "zh",
) -> str:
    list_separator = ", " if locale == "en" else "、"
    dish_names = list_separator.join(m.dish for m in decomposition.matches) or t("log_write.unknown_dish", locale)
    prefix = t("log_write.already_recorded", locale) if duplicate else t("log_write.recorded", locale)
    separator = " — " if locale == "en" else " —— "
    lines = [
        f"{prefix}: {logged_at.strftime('%Y-%m-%d %H:%M')} "
        f"{meal_type_label(meal_type, locale)}{separator}{dish_names}"
    ]
    for m in decomposition.matches:
        ingredients_text = list_separator.join(m.ingredients) or t("log_write.unknown_ingredient", locale)
        note = t("log_write.llm_note", locale) if m.source_tier == SOURCE_LLM_FALLBACK else ""
        lines.append(f"- {m.dish}：{ingredients_text}{note}")
    return "\n".join(lines)


async def stream_log_write(
    request: ChatRequest,
    server: DietExpertMcpServer,
    trace_id: str,
    profile: UserProfileContext | None,
    complete: CompleteFn,
    clarification_store: ClarificationStore,
    allow_clarification: bool = True,
) -> AsyncIterator[str]:
    user_id = profile.user_id if profile else request.user_id
    tz = _resolve_log_tz(profile)
    locale = request.locale

    try:
        decomposition = await decompose_meal(
            request.message, user_id=user_id, complete=complete, locale=locale
        )
    except Exception:
        logger.exception("dish decomposition failed · trace_id=%s", trace_id)
        yield sse_event("guardrail", {"type": "internal_error", "detail": t("log_write.decompose_failed", locale)})
        yield sse_event("done", {"trace_id": trace_id})
        return

    if not decomposition.matches:
        # D20 五处 agent 行为点第3条"记录解析追问"(✅V1做，2026-08-27 实现)：
        # 没识别出具体食物时先追问一次(PRD §11"输入模糊→追问一次")，而不是
        # 直接放弃；已经是重试轮(allow_clarification=False)时才判定为
        # unspecified，不再问第二次。
        if allow_clarification:
            question = t("log_write.clarification", locale)
            clarification_store.put(
                request.session_id,
                PendingClarification(original_text=request.message, branch=RouteBranch.LOG_WRITE),
            )
            yield sse_event("clarification", {"question": question})
            for chunk in chunk_text(question):
                yield sse_event("token", {"text": chunk})
            yield sse_event("done", {"trace_id": trace_id})
            return
        yield sse_event(
            "guardrail",
            {"type": "dish_not_recognized", "detail": t("log_write.not_recognized", locale)},
        )
        yield sse_event("done", {"trace_id": trace_id})
        return

    # §4.2 步骤2:过敏原即时检查，确定性集合比对，不经过模型。这里只警示、
    # 不阻止写入——记录的是"已经发生的事"，拦下写入既做不到(饭已经吃了)也
    # 没有意义，硬阻断该用在生成新建议的路径上(阶段5 guardrails)。
    allergens = set(profile.allergens) if profile else set()
    hit_allergens = allergens & set(decomposition.all_allergens())
    for allergen in sorted(hit_allergens):
        logger.warning("allergen hit in log_write · trace_id=%s · allergen=%s", trace_id, allergen)
        yield sse_event(
            "guardrail",
            {"type": "allergen_warning", "detail": t("log_write.allergen_warning", locale, allergen=allergen)},
        )

    now_local = datetime.now(tz)
    meal_type = _infer_meal_type(request.message, now_local)
    logged_at = _resolve_logged_at(request.message, tz)
    idempotency_key = _compute_idempotency_key(user_id, logged_at, request.message)

    session = server.open_session(CallerRole.ROUTER, user_id=user_id)
    payload = {
        "raw_input": request.message,
        # 每个 dish 对象带自己的 ingredients/tcm_nature/allergens——同
        # `dish_alias_promotion.py` record_llm_fallback_hit() 写 user_dish_aliases
        # 时用的既有形状一致(那边一直是对的)，这里之前只留了 dish/confidence/
        # source_tier 三个字段，把每道菜自己的食材/性味/过敏原全丢了，多道菜
        # 时只能看到顶层拍平后的 ingredients，分不清哪个食材属于哪道菜
        # (2026-08-31 用户反馈发现)。顶层 "ingredients"/"food_properties" 仍然
        # 保留拍平版本，是 query_diet_log.py by_ingredient/by_property 聚合
        # 依赖的既有字段，不能删。
        "dishes": [
            {
                "dish": m.dish,
                "confidence": m.confidence,
                "source_tier": m.source_tier,
                "ingredients": list(m.ingredients),
                "tcm_nature": m.tcm_nature,
                "allergens": list(m.allergens),
            }
            for m in decomposition.matches
        ],
        "ingredients": list(decomposition.all_ingredients()),
        "food_properties": list(decomposition.all_food_properties()),
        "meal_type": meal_type,
        "logged_at": logged_at,
    }
    try:
        write_result = session.call_tool(
            "write_memory",
            {"category": "daily_log", "payload": payload, "idempotency_key": idempotency_key},
        )
    except Exception:
        logger.exception("write_memory(daily_log) failed · trace_id=%s", trace_id)
        yield sse_event("guardrail", {"type": "internal_error", "detail": t("log_write.write_failed", locale)})
        yield sse_event("done", {"trace_id": trace_id})
        return

    # D27 修订一:LLM 兜底成功的结果计入个人别名晋升计数——按"剩余(未被全局表
    # 覆盖的)文本"归一化后的短语作为 key，和 decompose_meal() 内部第二级查找
    # 用的 key 必须一致，否则晋升了也永远查不到命中(见 dish_decomposition.py
    # 模块文档关于三级查找 key 的说明)。重试(duplicate=True)不计数，避免网络
    # 重试被误当成"用户又说了一遍同样的话"而重复推进晋升计数。
    llm_matches = tuple(m for m in decomposition.matches if m.source_tier == SOURCE_LLM_FALLBACK)
    if llm_matches and not write_result.duplicate:
        _, remaining = match_global_table(request.message)
        residual_phrase = normalize_phrase(remaining)
        try:
            record_llm_fallback_hit(user_id, residual_phrase, llm_matches)
        except Exception:
            logger.warning("dish alias promotion failed · trace_id=%s", trace_id, exc_info=True)

    text = _format_log_write_confirmation(
        decomposition, meal_type, logged_at, write_result.duplicate, locale=locale
    )
    for chunk in chunk_text(text):
        yield sse_event("token", {"text": chunk})
    yield sse_event("done", {"trace_id": trace_id})
