#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCM SubAgent：独立 24k 上下文，工具=retrieve_tcm/query_weather/query_diet_log(只读)。

设计依据：docs/ARCHITECTURE.md §5.2 步骤 3
决策依据：docs/DECISIONS.md D1(双 SubAgent 粒度)、D28(体质未知时的降级)
roadmap：阶段 4.2 任务 7

循环编排(资源限额/循环防护/领域隔离日志)复用 backend/agents/_subagent_common.py，
本文件只负责这一侧独有的东西：领域身份声明(不讨论营养学)、引用格式要求
(复用 backend/agents/citation.py)、D28 体质未知时的降级措辞。
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

DOMAIN = "tcm"

_DOMAIN_SCOPE_INSTRUCTION = (
    "你是 diet_expert 的中医饮食 SubAgent，拥有独立的上下文，只负责中医食养判断。"
    "只使用 retrieve_tcm 工具检索到的知识库内容作为依据；不要讨论营养成分、热量、"
    "维生素/矿物质含量这类属于 Nutrition SubAgent 职责范围的内容——即便你恰好知道，"
    "也不要在这里输出，两侧结论会在中枢的调和层分开处理后再合并(ARCHITECTURE.md "
    "§5.2 步骤 4-5)。"
)

# D28"降级规则"原文措辞：体质为空/未确认时，任务提示词显式声明"体质未知"，
# 不能猜测或套用一个默认体质——套错比不知道更危险。
_CONSTITUTION_UNKNOWN_INSTRUCTION = (
    "【体质未知】用户尚未确认中医体质分型。不要猜测或套用一个默认体质——错误的"
    "体质判断比“不知道”更危险，相当于给用户扣错体质的帽子。请把建议收窄为体质"
    "无关的普适性温和原则(比如少辛辣油腻、规律饮食这类不依赖具体体质分型的通用"
    "建议)，并在回答末尾附一句引导：完善体质信息后可以给出更精准的建议。"
)


def build_tcm_system_prompt(
    *,
    constitution: str | None = None,
    allergens: Iterable[str] | None = None,
    extra_profile_notes: str = "",
    locale: str = "zh",
) -> str:
    parts = [
        _DOMAIN_SCOPE_INSTRUCTION,
        build_citation_instruction(),
        build_score_guidance_instruction(),
        build_clarification_instruction(),
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


async def run_tcm_subagent(
    task_input: str,
    server: DietExpertMcpServer,
    *,
    constitution: str | None = None,
    allergens: Iterable[str] | None = None,
    extra_profile_notes: str = "",
    complete: CompleteFn | None = None,
    user_id: str = "default_user",
    locale: str = "zh",
) -> SubAgentResult:
    """`task_input` 是用户原始提问文本(D25，见 _subagent_common 模块文档)。
    `constitution` 由中枢从 user_profile 读出后传入(ARCHITECTURE §5.2 步骤 2)；
    为空时走 D28 的降级措辞，不在这里访问数据库。`allergens` 同样来自
    user_profile——生成阶段就提醒模型避开，核查 pass 的硬阻断是这条的兜底，
    不是唯一防线。`extra_profile_notes` 注入已确认的补剂/偏好（E8：无依据
    时声明不确定，不编造药食交互）。
    """
    return await run_subagent(
        domain=DOMAIN,
        role=CallerRole.TCM_SUBAGENT,
        system_prompt=build_tcm_system_prompt(
            constitution=constitution,
            allergens=allergens,
            extra_profile_notes=extra_profile_notes,
            locale=locale,
        ),
        task_input=task_input,
        server=server,
        complete=complete,
        user_id=user_id,
        locale=locale,
    )
