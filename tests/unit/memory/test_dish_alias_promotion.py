"""
测试目标：晋升计数边界（恰好达到阈值、被打断的计数序列）、normalized_phrase归一化规则
对应实现：backend/memory/dish_alias_promotion.py
覆盖要求：常规

真正的 UPSERT/计数/晋升行为依赖 Postgres 的 ON CONFLICT 语义，本文件只测不需要
连库的短路路径(空输入/非法阈值)；晋升阈值边界(恰好第3次)的真实验证见对话记录，
对真实本地 Postgres 插入/重跑/核对 hit_count 和 promoted_at。
"""
from __future__ import annotations

import pytest

from backend.memory.dish_alias_promotion import record_llm_fallback_hit
from backend.memory.dish_decomposition import (
    CONFIDENCE_LOW,
    SOURCE_LLM_FALLBACK,
    DishMatch,
)

_MATCH = DishMatch(
    dish="老三样", ingredients=("鸡蛋", "番茄"), tcm_nature="平", allergens=("蛋",),
    confidence=CONFIDENCE_LOW, source_tier=SOURCE_LLM_FALLBACK,
)


def test_empty_phrase_is_a_noop_without_touching_db():
    result = record_llm_fallback_hit("default_user", "   ", (_MATCH,))
    assert result.ok is False


def test_empty_matches_is_a_noop_without_touching_db():
    result = record_llm_fallback_hit("default_user", "老三样", ())
    assert result.ok is False


def test_non_positive_threshold_raises():
    with pytest.raises(ValueError):
        record_llm_fallback_hit("default_user", "老三样", (_MATCH,), threshold=0)


def test_write_failure_raises_not_silently_swallowed():
    """和 dish_decomposition.py 的"读失败静默降级"刻意不对称——晋升计数写失败
    必须让调用方知道，不能悄悄假装计数成功了。"""
    with pytest.raises(Exception):
        record_llm_fallback_hit(
            "default_user", "老三样", (_MATCH,),
            dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist",
        )
