#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nutrition SubAgent：独立 24k 上下文，工具=retrieve_nutrition/query_diet_log/
query_recipes_by_ingredients(只读)。

设计依据：docs/ARCHITECTURE.md §5.2 步骤 3
决策依据：docs/DECISIONS.md D1(双 SubAgent 粒度)
roadmap：阶段 4.2 任务 7

循环编排(资源限额/循环防护/领域隔离日志)复用 backend/agents/_subagent_common.py，
本文件只负责这一侧独有的东西：领域身份声明(不讨论中医体质)、引用格式要求
(复用 backend/agents/citation.py)。没有 D28 那样的降级分支——体质未知只影响
TCM SubAgent 的判断范围，营养学结论不依赖体质字段。
"""
from __future__ import annotations

from typing import Iterable

from backend.agents._subagent_common import (
    SubAgentResult,
    build_allergen_avoidance_instruction,
    run_subagent,
)
from backend.agents.citation import (
    build_citation_instruction,
    build_clarification_instruction,
    build_score_guidance_instruction,
)
from backend.i18n import apply_language_instruction
from backend.llm.adapter import CompleteFn
from backend.mcp_server.roles import CallerRole
from backend.mcp_server.server import DietExpertMcpServer

DOMAIN = "nutrition"

_DOMAIN_SCOPE_INSTRUCTION = (
    "你是 diet_expert 的营养学 SubAgent，拥有独立的上下文，只负责营养学判断。"
    "使用 retrieve_nutrition 检索知识库；需要具体菜谱或按食材查做法时调用 "
    "query_recipes_by_ingredients。不要讨论中医体质、性味、归经这类属于 TCM "
    "SubAgent 职责范围的内容——即便你恰好知道，也不要在这里输出，两侧结论会在"
    "中枢的调和层分开处理后再合并(ARCHITECTURE.md §5.2 步骤 4-5)。"
)

# Minimum recipe-path trigger (P1-3). Full 3-day plan assembly is still deferred.
_RECIPE_HINTS = (
    "菜谱",
    "食谱",
    "购物清单",
    "购物单",
    "怎么做",
    "做什么菜",
    "三天食谱",
    "一周食谱",
    "recipe",
    "recipes",
    "shopping list",
    "how to cook",
    "what to cook",
    "meal plan",
)


def is_recipe_assembly_request(message: str) -> bool:
    text = message or ""
    lowered = text.lower()
    return any(hint in (lowered if hint.isascii() else text) for hint in _RECIPE_HINTS)


def build_nutrition_system_prompt(
    *,
    allergens: Iterable[str] | None = None,
    extra_profile_notes: str = "",
    include_recipe_skill: bool = False,
    locale: str = "zh",
) -> str:
    parts = [
        _DOMAIN_SCOPE_INSTRUCTION,
        build_citation_instruction(),
        build_score_guidance_instruction(),
        build_clarification_instruction(),
    ]
    allergen_instruction = build_allergen_avoidance_instruction(allergens)
    if allergen_instruction:
        parts.append(allergen_instruction)
    if extra_profile_notes.strip():
        parts.append(extra_profile_notes.strip())
    prompt = apply_language_instruction("\n\n".join(parts), locale)
    if include_recipe_skill:
        from backend.skills.registry import compose_prompt_with_skills

        return compose_prompt_with_skills(prompt, ["recipe_and_shopping_list"])
    return prompt


async def run_nutrition_subagent(
    task_input: str,
    server: DietExpertMcpServer,
    *,
    allergens: Iterable[str] | None = None,
    extra_profile_notes: str = "",
    include_recipe_skill: bool = False,
    complete: CompleteFn | None = None,
    user_id: str = "default_user",
    locale: str = "zh",
) -> SubAgentResult:
    """`task_input` 是用户原始提问文本(D25，见 _subagent_common 模块文档)。
    `allergens` 来自 user_profile——生成阶段就提醒模型避开，核查 pass 的硬阻断
    是这条的兜底，不是唯一防线。`include_recipe_skill` 只在完整推荐且用户
    明确要菜谱/购物清单时为 True；不是独立的「步骤 7 三日计划」管线。"""
    return await run_subagent(
        domain=DOMAIN,
        role=CallerRole.NUTRITION_SUBAGENT,
        system_prompt=build_nutrition_system_prompt(
            allergens=allergens,
            extra_profile_notes=extra_profile_notes,
            include_recipe_skill=include_recipe_skill,
            locale=locale,
        ),
        task_input=task_input,
        server=server,
        complete=complete,
        user_id=user_id,
        locale=locale,
    )
