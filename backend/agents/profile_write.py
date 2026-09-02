#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`profile_write` 分支——把过敏原/不耐受/补剂/饮食偏好等**画像关键事实**写入
`user_profile`（经人在环确认），不是 `log_write` 的「记一顿吃了什么」。

设计依据：docs/ARCHITECTURE.md §4.3（跨分支扫描）+ 本分支补 LLM 提取层：
  - `critical_fact_scanner`：确定性词表/句式，任意分支前都会扫（顺带提及）
  - `profile_write`：用户**明确要记录**健康限制时走此分支；LLM 抽取词表
    覆盖不到的过敏原（如猕猴桃、冷门食材）并与扫描结果合并，仍走
    `pending_critical_facts` → 用户确认 → `write_memory(critical)`。

2026-09-02 新增。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping

from api.schemas import ChatRequest
from backend.agents.routing import CompleteFn
from backend.agents.sse import chunk_text, sse_event
from backend.agents.user_context import UserProfileContext
from backend.i18n import apply_language_instruction, current_locale, t
from backend.memory.critical_fact_scanner import (
    CriticalFactScanResult,
    _preferences_delta,
    scan_critical_facts,
)
from backend.memory.pending_critical_facts import (
    PendingCriticalFact,
    PendingCriticalFactStore,
    new_pending_id,
)

logger = logging.getLogger("diet_expert.agents.profile_write")

_PROFILE_EXTRACT_SYSTEM = (
    "你是饮食画像录入助手。用户想把过敏、不耐受、正在服用的补剂或饮食偏好"
    "（口味、忌口如不吃某菜）记入个人画像。"
    "从用户原话中提取要保存的项目，只输出一个 JSON 对象，前后不要任何解释：\n"
    '{"allergens":["..."], "supplements":["..."], "preferences":{...}}\n'
    "规则：\n"
    "- allergens：食物过敏或不耐受（尽量用中文类别名：甲壳类/鱼类/乳制品/麸质/"
    "芝麻/花生/坚果/大豆/蛋类；具体食物用中文名如芒果、猕猴桃；不要编造用户"
    "没提到的项）\n"
    "- supplements：膳食补剂名称，保留用户说法\n"
    "- preferences：饮食偏好对象。常见键：忌口（字符串数组，如不吃香菜）、"
    "notes（自由文本）。只写用户明确表达的偏好，不要编造\n"
    "- 若用户只是在问问题、或没有任何要写入画像的新事实，对应字段用空数组/空对象"
)


@dataclass(frozen=True)
class LlmProfileExtract:
    allergens: tuple[str, ...] = ()
    supplements: tuple[str, ...] = ()
    preferences: dict[str, Any] = field(default_factory=dict)


def _normalize_preferences_patch(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        k = str(key).strip()
        if not k:
            continue
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                out[k] = items
        elif isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                out[k] = cleaned
        elif value is not None:
            out[k] = value
    return out


def _strip_json_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def _parse_profile_extract_json(raw_text: str) -> LlmProfileExtract:
    try:
        data = json.loads(_strip_json_fences(raw_text))
    except (json.JSONDecodeError, TypeError):
        return LlmProfileExtract()
    if not isinstance(data, dict):
        return LlmProfileExtract()
    allergens = data.get("allergens")
    supplements = data.get("supplements")
    preferences = data.get("preferences")
    a_out: list[str] = []
    s_out: list[str] = []
    if isinstance(allergens, list):
        for item in allergens:
            cleaned = str(item).strip()
            if cleaned:
                a_out.append(cleaned)
    if isinstance(supplements, list):
        for item in supplements:
            cleaned = str(item).strip()
            if cleaned:
                s_out.append(cleaned)
    return LlmProfileExtract(
        allergens=tuple(dict.fromkeys(a_out)),
        supplements=tuple(dict.fromkeys(s_out)),
        preferences=_normalize_preferences_patch(preferences),
    )


def merge_profile_facts(
    message: str,
    profile: UserProfileContext | None,
    *,
    deterministic: CriticalFactScanResult | None = None,
    llm_extract: LlmProfileExtract | None = None,
) -> CriticalFactScanResult:
    """Merge deterministic scan + LLM extraction, relative to existing profile."""
    det = deterministic if deterministic is not None else scan_critical_facts(message, profile)
    llm = llm_extract or LlmProfileExtract()
    existing_allergens = set(profile.allergens) if profile else set()
    existing_supplements = {s.get("name") for s in (profile.supplements if profile else ())}
    existing_preferences = profile.preferences if profile else {}

    candidate_allergens = set(det.new_allergens) | set(llm.allergens)
    candidate_supplements = set(det.new_supplements) | set(llm.supplements)

    new_allergens = tuple(sorted(candidate_allergens - existing_allergens))
    new_supplements = tuple(sorted(candidate_supplements - existing_supplements))
    new_preferences = _preferences_delta(existing_preferences, llm.preferences)
    return CriticalFactScanResult(
        new_allergens=new_allergens,
        new_supplements=new_supplements,
        new_preferences=new_preferences,
    )


async def llm_extract_profile_facts(
    message: str,
    *,
    complete: CompleteFn,
    locale: str | None = None,
) -> LlmProfileExtract:
    result = await complete(
        [
            {
                "role": "system",
                "content": apply_language_instruction(
                    _PROFILE_EXTRACT_SYSTEM,
                    locale if locale is not None else current_locale(),
                ),
            },
            {"role": "user", "content": message},
        ],
        force_prod_tier=False,
    )
    return _parse_profile_extract_json(result.text or "")


def _pending_already_covers(
    store: PendingCriticalFactStore,
    session_id: str,
    merged: CriticalFactScanResult,
) -> bool:
    """True when an existing session pending already lists the same new items."""
    pending_items = store.list_for_session(session_id)
    if not pending_items or not merged.hit:
        return False
    pending_allergens: set[str] = set()
    pending_supplements: set[str] = set()
    pending_preferences: dict[str, Any] = {}
    for item in pending_items:
        pending_allergens.update(item.allergens)
        pending_supplements.update(item.supplements)
        for key, value in item.preferences.items():
            if isinstance(value, list) and isinstance(pending_preferences.get(key), list):
                pending_preferences[key] = list(
                    dict.fromkeys([*pending_preferences[key], *value])
                )
            else:
                pending_preferences[key] = value
    prefs_covered = _pending_preferences_covered(pending_preferences, merged.new_preferences)
    return (
        set(merged.new_allergens) <= pending_allergens
        and set(merged.new_supplements) <= pending_supplements
        and prefs_covered
    )


def _pending_preferences_covered(
    pending_preferences: dict[str, Any], new_preferences: Mapping[str, Any]
) -> bool:
    if not new_preferences:
        return True
    for key, value in new_preferences.items():
        pending_val = pending_preferences.get(key)
        if isinstance(value, list):
            if not isinstance(pending_val, list):
                return False
            if any(item not in pending_val for item in value):
                return False
        elif pending_val != value:
            return False
    return True


async def stream_profile_write(
    request: ChatRequest,
    trace_id: str,
    profile: UserProfileContext | None,
    complete: CompleteFn,
    pending_critical_store: PendingCriticalFactStore,
    *,
    prefetched_scan: CriticalFactScanResult | None = None,
) -> AsyncIterator[str]:
    locale = request.locale
    deterministic = prefetched_scan or scan_critical_facts(request.message, profile)
    try:
        llm_extract = await llm_extract_profile_facts(
            request.message, complete=complete, locale=locale
        )
    except Exception:
        logger.exception("profile_write LLM extract failed · trace_id=%s", trace_id)
        yield sse_event(
            "guardrail",
            {"type": "internal_error", "detail": t("profile_write.extract_failed", locale)},
        )
        yield sse_event("done", {"trace_id": trace_id})
        return

    merged = merge_profile_facts(
        request.message,
        profile,
        deterministic=deterministic,
        llm_extract=llm_extract,
    )

    if merged.hit and not _pending_already_covers(
        pending_critical_store, request.session_id, merged
    ):
        pending = PendingCriticalFact(
            pending_id=new_pending_id(),
            user_id=profile.user_id if profile else request.user_id,
            session_id=request.session_id,
            allergens=merged.new_allergens,
            supplements=merged.new_supplements,
            preferences=dict(merged.new_preferences),
        )
        try:
            pending_critical_store.put(pending)
        except Exception:
            logger.exception("profile_write pending store failed · trace_id=%s", trace_id)
            yield sse_event(
                "guardrail",
                {
                    "type": "pending_critical_store_failed",
                    "detail": t("api.pending_critical_store_failed", locale),
                },
            )
        else:
            logger.info(
                "profile_write critical fact pending · trace_id=%s · pending_id=%s · "
                "allergens=%s · supplements=%s · preferences=%s",
                trace_id,
                pending.pending_id,
                merged.new_allergens,
                merged.new_supplements,
                merged.new_preferences,
            )
            yield sse_event("critical_fact_pending", pending.to_event_dict(locale=locale))
            text = t("profile_write.ack_with_pending", locale)
    elif merged.hit:
        text = t("profile_write.ack_already_pending", locale)
    else:
        text = t("profile_write.nothing_new", locale)

    for chunk in chunk_text(text):
        yield sse_event("token", {"text": chunk})
    yield sse_event("done", {"trace_id": trace_id})
