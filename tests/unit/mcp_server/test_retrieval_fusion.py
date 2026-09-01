"""
测试目标：backend/mcp_server/tools/_retrieval_common.py 里 2026-08-30 新增的
检索评分方法优化部分——RRF 融合、MQE 查询改写、同步/异步桥接。全部是不碰
数据库/embedding 模型的纯函数或可用假 complete 注入的逻辑。
对应实现：backend/mcp_server/tools/_retrieval_common.py
不测：`search_knowledge_chunks()` 本身（需要真实 Postgres + BGE-M3，按既有
惯例走真实数据手工验证，见 BUILD_PLAN.md 混合检索那一行）。
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from backend.mcp_server.tools._retrieval_common import (
    RRF_K,
    _run_coroutine_sync,
    generate_query_variants,
    reciprocal_rank_fusion,
    warm_embedder_enabled,
)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_rrf_single_list_preserves_relative_order():
    scores = reciprocal_rank_fusion([["a", "b", "c"]])
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_agreement_across_lists_boosts_score():
    # "a" 在两路都排第一，"b" 只在一路里出现——即便排第一，总分也该更低。
    scores = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_rrf_matches_formula_directly():
    scores = reciprocal_rank_fusion([["a", "b"]], k=RRF_K)
    assert scores["a"] == 1.0 / (RRF_K + 1)
    assert scores["b"] == 1.0 / (RRF_K + 2)


def test_rrf_custom_k_changes_scores_but_not_ordering():
    default_scores = reciprocal_rank_fusion([["a", "b", "c"]])
    tight_scores = reciprocal_rank_fusion([["a", "b", "c"]], k=1)
    assert default_scores["a"] != tight_scores["a"]
    assert tight_scores["a"] > tight_scores["b"] > tight_scores["c"]


# ---------------------------------------------------------------------------
# generate_query_variants (MQE)
# ---------------------------------------------------------------------------


def _complete_returning(text: str):
    async def _complete(messages, **kwargs):
        return SimpleNamespace(text=text)

    return _complete


def test_generate_query_variants_parses_json_array():
    complete = _complete_returning(json.dumps(["气虚质饮食建议", "春季饮食建议"]))
    variants = generate_query_variants("气虚质春季容易疲乏，饮食上怎么补？", complete=complete)
    assert variants == ["气虚质饮食建议", "春季饮食建议"]


def test_generate_query_variants_strips_markdown_code_fence():
    complete = _complete_returning('```json\n["改写1", "改写2"]\n```')
    variants = generate_query_variants("原始问题", complete=complete)
    assert variants == ["改写1", "改写2"]


def test_generate_query_variants_truncates_to_max_variants():
    complete = _complete_returning(json.dumps(["v1", "v2", "v3", "v4"]))
    variants = generate_query_variants("原始问题", complete=complete, max_variants=2)
    assert variants == ["v1", "v2"]


def test_generate_query_variants_drops_blank_entries():
    complete = _complete_returning(json.dumps(["有效改写", "  ", ""]))
    variants = generate_query_variants("原始问题", complete=complete)
    assert variants == ["有效改写"]


def test_generate_query_variants_falls_back_to_empty_on_invalid_json():
    complete = _complete_returning("这不是 JSON")
    variants = generate_query_variants("原始问题", complete=complete)
    assert variants == []


def test_generate_query_variants_falls_back_to_empty_on_non_list_json():
    complete = _complete_returning(json.dumps({"not": "a list"}))
    variants = generate_query_variants("原始问题", complete=complete)
    assert variants == []


def test_generate_query_variants_falls_back_to_empty_on_llm_exception():
    async def _raising_complete(messages, **kwargs):
        raise RuntimeError("provider unavailable")

    variants = generate_query_variants("原始问题", complete=_raising_complete)
    assert variants == []


# ---------------------------------------------------------------------------
# _run_coroutine_sync：同步/异步桥接
# ---------------------------------------------------------------------------


def test_run_coroutine_sync_without_a_running_loop():
    async def coro():
        return 42

    assert _run_coroutine_sync(coro()) == 42


def test_run_coroutine_sync_from_inside_a_running_event_loop():
    """`search_knowledge_chunks()` 被 `backend/agents/agent_loop.py`
    `run_agent_loop()`(一个运行中的 event loop)调用时，不能直接
    `asyncio.run()`——这条测试模拟那个场景：从一个已经在跑的 loop 里面
    调用 `_run_coroutine_sync()`，验证它会走线程桥接分支而不是直接抛
    `RuntimeError: asyncio.run() cannot be called from a running event loop`。
    """

    async def coro():
        return "result-from-nested-loop"

    async def call_from_inside_running_loop():
        return _run_coroutine_sync(coro())

    assert asyncio.run(call_from_inside_running_loop()) == "result-from-nested-loop"


# ---------------------------------------------------------------------------
# warm_embedder_enabled
# ---------------------------------------------------------------------------


def test_warm_embedder_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("DIET_EXPERT_WARM_EMBEDDER", raising=False)
    assert warm_embedder_enabled() is True


def test_warm_embedder_enabled_respects_zero(monkeypatch):
    monkeypatch.setenv("DIET_EXPERT_WARM_EMBEDDER", "0")
    assert warm_embedder_enabled() is False
