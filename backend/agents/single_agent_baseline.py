#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 ablation baseline(docs/DECISIONS.md D1"验证方式")：单 agent、中医+营养两个
检索工具同时暴露在同一上下文里，一次调用直接产出结论——不拆两个 SubAgent、
不经过独立的调和层 LLM 调用。

**只用于 `evals/run_b2_ablation.py` 的实验对比，不接进 `/api/chat` 生产路径。**
D1 的理由一(上下文隔离)、理由二(可分别评估)都假设"两个上下文分开"比"一个
上下文"更好；这个文件就是那个"如果不分开会怎样"的对照组，复用
`backend/agents/_subagent_common.py` 的同一套循环编排(资源限额/循环防护/
领域隔离日志)，只是把两侧的领域声明、引用要求、体质降级措辞拼进同一份
system prompt，工具集是 TCM+Nutrition 两个 SubAgent 各自工具的并集(去掉
write_memory，同 SubAgent 一贯的只读原则)。

失败模式假设(检验，不是预设结论)：单 agent 可能把中医的定性判断和营养的
定量数据在同一段推理里混在一起(D1 理由一"双向污染")，也可能反而因为一次
调用就能看到两侧证据、不需要靠调和层二次转述而错漏，直接给出更连贯的结论
(D1 理由一的反例)。哪种情况更常见需要真实跑分回答，不是靠论证。
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

DOMAIN = "single_agent_b2"

_DOMAIN_SCOPE_INSTRUCTION = (
    "你是 diet_expert 的饮食顾问，同时负责中医食养判断和营养学分析——这两件事"
    "在这次对话里由你一个人完成，不会有另一个专家复核你的结论。"
    "你有两个检索工具：retrieve_tcm(中医食养知识库)、retrieve_nutrition(营养学"
    "知识库)。除非问题明显只属于其中一侧，否则两个工具都要检索一遍，再把两侧"
    "证据放在一起给出一份连贯的结论——不要只查一个库就下结论，也不要把两侧"
    "证据分成两段互不相关的话简单拼接，你的任务是把它们真正综合成一个回答。"
)

_CONSTITUTION_UNKNOWN_INSTRUCTION = (
    "【体质未知】用户尚未确认中医体质分型。不要猜测或套用一个默认体质——错误的"
    "体质判断比“不知道”更危险，相当于给用户扣错体质的帽子。请把建议收窄为体质"
    "无关的普适性温和原则(比如少辛辣油腻、规律饮食这类不依赖具体体质分型的通用"
    "建议)，并在回答末尾附一句引导：完善体质信息后可以给出更精准的建议。"
)


def build_single_agent_system_prompt(
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


async def run_single_agent_b2(
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
    return await run_subagent(
        domain=DOMAIN,
        role=CallerRole.SINGLE_AGENT_B2,
        system_prompt=build_single_agent_system_prompt(
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
