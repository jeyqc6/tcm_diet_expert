"""profile_write branch: routing regex, merge logic, stream handler."""
from __future__ import annotations

import asyncio
import json

import pytest

from backend.agents.profile_write import (
    LlmProfileExtract,
    merge_profile_facts,
    _parse_profile_extract_json,
)
from backend.agents.routing import RouteBranch, classify_route
from backend.agents.user_context import UserProfileContext
from backend.memory.critical_fact_scanner import CriticalFactScanResult


def test_profile_write_regex_beats_log_write_for_preference() -> None:
    decision = classify_route("记录一下我不吃香菜")
    assert decision.branch is RouteBranch.PROFILE_WRITE
    assert decision.rule_matched is True


def test_merge_profile_facts_includes_preferences_delta() -> None:
    profile = UserProfileContext(
        user_id="u1",
        preferences={"忌口": ["花生"]},
    )
    merged = merge_profile_facts(
        "ignored",
        profile,
        deterministic=CriticalFactScanResult(),
        llm_extract=LlmProfileExtract(
            allergens=("芒果",),
            preferences={"忌口": ["香菜", "花生"]},
        ),
    )
    assert merged.new_allergens == ("芒果",)
    assert merged.new_preferences == {"忌口": ["香菜"]}


def test_parse_profile_extract_json_includes_preferences() -> None:
    raw = '{"allergens":[],"supplements":[],"preferences":{"忌口":["香菜"]}}'
    parsed = _parse_profile_extract_json(raw)
    assert parsed.preferences == {"忌口": ["香菜"]}


def test_profile_write_regex_beats_log_write_for_health_log_request() -> None:
    msg = "Please log that I am slightly lactose intolerant and allergic to mangoes."
    decision = classify_route(msg)
    assert decision.branch is RouteBranch.PROFILE_WRITE
    assert decision.rule_matched is True


def test_log_write_still_matches_ate_pattern() -> None:
    decision = classify_route("Please log that I ate mapo tofu for lunch.")
    assert decision.branch is RouteBranch.LOG_WRITE


def test_merge_profile_facts_combines_scanner_and_llm() -> None:
    profile = UserProfileContext(user_id="u1", allergens=("花生",))
    merged = merge_profile_facts(
        "ignored",
        profile,
        deterministic=CriticalFactScanResult(new_allergens=("甲壳类",)),
        llm_extract=LlmProfileExtract(allergens=("芒果", "猕猴桃"), supplements=("鱼油",)),
    )
    assert set(merged.new_allergens) == {"甲壳类", "猕猴桃", "芒果"}
    assert merged.new_supplements == ("鱼油",)


def test_parse_profile_extract_json_strips_fences() -> None:
    raw = '```json\n{"allergens":["猕猴桃"], "supplements":[]}\n```'
    parsed = _parse_profile_extract_json(raw)
    assert parsed.allergens == ("猕猴桃",)


def test_stream_profile_write_emits_pending_and_ack() -> None:
    from api.schemas import ChatRequest
    from backend.agents.profile_write import stream_profile_write
    from backend.memory.pending_critical_facts import InMemoryPendingCriticalFactStore

    store = InMemoryPendingCriticalFactStore()

    async def fake_complete(messages, **kwargs):
        class _R:
            text = json.dumps({"allergens": ["猕猴桃"], "supplements": []})

        return _R()

    request = ChatRequest(
        session_id="s1",
        message="Remember I'm allergic to kiwi",
        user_id="u1",
        locale="en",
    )
    chunks: list[str] = []

    async def collect() -> list[str]:
        out: list[str] = []
        async for chunk in stream_profile_write(
            request,
            "trace",
            UserProfileContext(user_id="u1"),
            fake_complete,
            store,
            prefetched_scan=CriticalFactScanResult(),
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())

    assert any("critical_fact_pending" in c for c in chunks)
    assert any("event: token" in c for c in chunks)
    assert store.list_for_session("s1")
