#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRD §11「双 SubAgent 失败 → 降级为纯 RAG 问答」的生产实现。

双侧 SubAgent 都抛错/超时后，不再直接结束请求，而是：
  1. 对中医、营养两个 domain 各做一次向量检索（不开 MQE，避免额外 LLM 调用）
  2. 单次 `complete()` 基于检索片段生成回答
  3. 返回 `SubAgentResult`，供既有核查 pass / SSE 流式路径复用

设计依据：docs/PRD.md §11 Fallback · docs/BUILD_PLAN.md 阶段 7「双侧失败 naive RAG」
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from typing import Iterable

from backend.agents._subagent_common import (
    SubAgentResult,
    build_allergen_avoidance_instruction,
)
from backend.agents.citation import (
    build_citation_instruction,
    build_score_guidance_instruction,
    format_retrieved_context,
)
from backend.i18n import apply_language_instruction, normalize_locale, set_current_locale
from backend.llm.adapter import CompleteFn
from backend.mcp_server.tools._retrieval_common import (
    RetrievedChunk,
    search_knowledge_chunks,
)
from backend.observability.tracing import observation, stage_log

logger = logging.getLogger("diet_expert.agents.naive_rag_fallback")

DOMAIN = "naive_rag_fallback"
NAIVE_RAG_SYSTEM_MARKER = "PRD §11 bilateral SubAgent failure fallback"
NAIVE_RAG_TOP_K_PER_DOMAIN = 5

_DEGRADED_SCOPE_INSTRUCTION = (
    f"【{NAIVE_RAG_SYSTEM_MARKER}】"
    "双侧专家 SubAgent 分析不可用，你正在以降级模式回答：只能依据下面预先检索好的"
    "知识库片段作答，不要假装进行过双专家分析、调和或额外检索。"
    "请综合中医食养与营养学两侧的资料，给出一份连贯、可操作的饮食建议。"
)

_CONSTITUTION_UNKNOWN_INSTRUCTION = (
    "【体质未知】用户尚未确认中医体质分型。不要猜测或套用一个默认体质——错误的"
    "体质判断比“不知道”更危险。请把建议收窄为体质无关的普适性温和原则，并在"
    "回答末尾附一句引导：完善体质信息后可以给出更精准的建议。"
)


def build_naive_rag_system_prompt(
    *,
    constitution: str | None = None,
    allergens: Iterable[str] | None = None,
    extra_profile_notes: str = "",
    locale: str = "zh",
) -> str:
    parts = [
        _DEGRADED_SCOPE_INSTRUCTION,
        build_citation_instruction(),
        build_score_guidance_instruction(),
    ]
    if constitution:
        parts.append(f"用户已确认的中医体质:{constitution}。")
    else:
        parts.append(_CONSTITUTION_UNKNOWN_INSTRUCTION)
    allergen_instruction = build_allergen_avoidance_instruction(allergens)
    if allergen_instruction:
        parts.append(allergen_instruction)
    if extra_profile_notes.strip():
        parts.append(extra_profile_notes.strip())
    return apply_language_instruction("\n\n".join(parts), locale)


def _chunks_to_tool_payload(chunks: list[RetrievedChunk]) -> str:
    return json.dumps(
        [dataclasses.asdict(chunk) for chunk in chunks],
        ensure_ascii=False,
    )


def _build_subagent_result(
    final_text: str,
    *,
    tcm_chunks: list[RetrievedChunk],
    nutrition_chunks: list[RetrievedChunk],
) -> SubAgentResult:
    messages: list[dict] = []
    if tcm_chunks:
        messages.append(
            {
                "role": "tool",
                "name": "retrieve_tcm",
                "ok": True,
                "content": _chunks_to_tool_payload(tcm_chunks),
            }
        )
    if nutrition_chunks:
        messages.append(
            {
                "role": "tool",
                "name": "retrieve_nutrition",
                "ok": True,
                "content": _chunks_to_tool_payload(nutrition_chunks),
            }
        )
    return SubAgentResult(
        domain=DOMAIN,
        final_text=final_text,
        tool_call_count=0,
        iterations=1,
        terminated_reason="naive_rag_single_shot",
        messages=messages,
        tools_called=[],
    )


def _retrieve_both_domains(
    retrieval_query: str,
    *,
    locale: str,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Sync retrieval for both domains — no MQE (keeps fallback to one LLM call)."""
    common = dict(
        top_k=NAIVE_RAG_TOP_K_PER_DOMAIN,
        use_mqe=False,
        use_hybrid=True,
        locale=locale,
    )
    tcm_chunks = search_knowledge_chunks("tcm", retrieval_query, **common)
    nutrition_chunks = search_knowledge_chunks("nutrition", retrieval_query, **common)
    return tcm_chunks, nutrition_chunks


def _format_user_prompt(
    task_input: str,
    *,
    tcm_chunks: list[RetrievedChunk],
    nutrition_chunks: list[RetrievedChunk],
) -> str:
    sections: list[str] = []
    if tcm_chunks:
        sections.append("【中医食养检索结果】\n" + format_retrieved_context(tcm_chunks))
    if nutrition_chunks:
        sections.append("【营养学检索结果】\n" + format_retrieved_context(nutrition_chunks))
    if not sections:
        sections.append("【检索结果】\n（知识库未命中直接相关的片段。）")
    sections.append(f"【用户问题】\n{task_input}")
    return "\n\n".join(sections)


async def run_naive_rag_fallback(
    task_input: str,
    complete: CompleteFn,
    *,
    retrieval_query: str | None = None,
    constitution: str | None = None,
    allergens: Iterable[str] | None = None,
    extra_profile_notes: str = "",
    locale: str = "zh",
) -> SubAgentResult:
    """Retrieve once per domain, then answer with a single LLM call."""
    locale = normalize_locale(locale)
    set_current_locale(locale)
    query = (retrieval_query or task_input).strip()
    t0 = time.perf_counter()

    with observation("naive_rag_fallback.retrieve", as_type="retriever"):
        tcm_chunks, nutrition_chunks = await asyncio.to_thread(
            _retrieve_both_domains,
            query,
            locale=locale,
        )

    retrieve_ms = (time.perf_counter() - t0) * 1000.0
    stage_log(
        logger,
        "naive_rag_retrieve",
        latency_ms=round(retrieve_ms, 1),
        tcm_hits=len(tcm_chunks),
        nutrition_hits=len(nutrition_chunks),
    )

    messages = [
        {
            "role": "system",
            "content": build_naive_rag_system_prompt(
                constitution=constitution,
                allergens=allergens,
                extra_profile_notes=extra_profile_notes,
                locale=locale,
            ),
        },
        {
            "role": "user",
            "content": _format_user_prompt(
                task_input,
                tcm_chunks=tcm_chunks,
                nutrition_chunks=nutrition_chunks,
            ),
        },
    ]

    t1 = time.perf_counter()
    with observation("naive_rag_fallback.complete", as_type="generation"):
        llm_result = await complete(messages)
    complete_ms = (time.perf_counter() - t1) * 1000.0
    stage_log(logger, "naive_rag_complete", latency_ms=round(complete_ms, 1))

    final_text = (llm_result.text or "").strip()
    if not final_text:
        raise RuntimeError("naive RAG fallback returned empty text")

    return _build_subagent_result(
        final_text,
        tcm_chunks=tcm_chunks,
        nutrition_chunks=nutrition_chunks,
    )
