#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调和层：独立 LLM 调用，只收两侧结论与依据，不接收原始检索内容。

设计依据：docs/ARCHITECTURE.md §5.2 步骤 5
决策依据：docs/DECISIONS.md D14(调和层独立成一次 LLM 调用,只收两侧结论与依据,
不接收原始检索内容——避免重新引入 D1 试图消除的上下文污染)、D19(2026-08-27
修订：调和层调用不再强制 `force_prod_tier=True`,跟随 `MODEL_TIER` 与其余调用
一致,见 D19 决策修订记录)
roadmap:阶段 4.2 任务 8

D14 的边界在这里靠**函数签名**强制,不是靠约定:`reconcile()` 只接受
`tcm_conclusion`/`nutrition_conclusion` 两个纯文本参数,不接受
`backend.agents._subagent_common.SubAgentResult` 对象本身——调用方必须自己先把
`.final_text` 取出来传进来,`.messages`(里面是原始 tool_result,含检索到的
chunk 原文)在类型层面就传不进这个函数。`reconcile_subagent_results()` 是给
上游用的一层薄封装,只替调用方做"取 `.final_text`"这一步,同样不碰
`.messages`,并且打日志记录"这一侧原本有多少条 messages 被丢弃"，作为
"调和层确实没收到原始 chunk"这条完成判据的可核查证据。

调和层本身**无工具**(ARCHITECTURE §2.3):这是一次单轮 LLM 调用,不经过
`backend/agents/router.py` 的 Agent Loop,不开 MCP session。

agent 行为点 #2(D20,§5.2 步骤 5)提到"依据不足时可回退请求 SubAgent 补充查询
(建议上限 1 次)"——中枢编排层(`api/main.py` 的 `_stream_dual_dispatch`)现在
已经实现了这条 retry 的一个具体场景：核查 pass 命中过敏原(check_number=4)时,
带着"要避开什么"的具体反馈重新调一次这里的 `reconcile()`/
`reconcile_subagent_results()`(上限 1 次,DECISIONS.md 待决问题表"核查 pass
退回调和层的最大重试次数")。本文件负责接住这条反馈——`avoid_note` 参数拼进
用户消息的"重新生成要求"小节；retry 循环本身(判断要不要重试、重试几次)是
`api/main.py` 的职责，不在这个模块里，保持 D14"调和层只做一次独立仲裁"的
单次调用性质不变——每次 `reconcile()` 调用仍然只是一次 LLM 调用，重试是调用方
多调了一次，不是这个函数自己在循环。

Skill 内容(硬优先级/仲裁原则/harm reduction/输出格式)见
`backend/skills/reconciliation_rubric.md`,正文不在这里重复;"候选评估分支走
D25 新增规则"体现为该 Skill 里的 harm-reduction 小节——Skill 按 pipeline 步骤
确定性加载(D22),不按分支条件加载,候选评估和完整推荐两条分支走的是同一份
rubric,不需要为候选评估单独分支。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from backend.agents._subagent_common import SubAgentResult
from backend.i18n import apply_language_instruction
from backend.llm import adapter as llm_adapter
from backend.llm.adapter import CompleteFn, ModelTier
from backend.observability.redact import redact_text
from backend.observability.tracing import observation, stage_log, update_current
from backend.skills.registry import compose_prompt_with_skills

logger = logging.getLogger("diet_expert.agents.reconciliation")

_SKILL_RECONCILIATION_RUBRIC = "reconciliation_rubric"

_BASE_SYSTEM_PROMPT = (
    "你是调和层，也是这个饮食助手唯一对用户说话的声音。下面给你中医食养和"
    "营养学两个角度已经得出的结论与依据，请你做一次仲裁并直接给用户答案，"
    "遵循下面的准则。你的输出会原样展示给用户，绝对不要提到「SubAgent」这个"
    "词，也不要写成两个专家在对话/转述对方意见的样子——你就是在直接回答"
    "用户，不是在转述别人说了什么。"
)


@dataclass
class ReconciliationResult:
    text: str
    model: str
    tier: ModelTier
    provider: str


def _format_user_profile(user_profile: dict[str, Any] | None) -> str:
    if not user_profile:
        return "（无用户画像，按体质/过敏原均未知处理，§5.2 步骤 5 预算 0.5k）"
    return json.dumps(user_profile, ensure_ascii=False, sort_keys=True)


def _format_matched_rules(matched_rules: list[dict[str, Any]] | None) -> str:
    if not matched_rules:
        return "（无命中 conflict_rules，按仲裁原则第 5 条：诚实说明表内无直接规则）"
    lines = []
    for rule in matched_rules:
        # 仲裁原则第 4 条只要求 relation + resolution，其余字段(source/evidence_level
        # 等)不进 prompt——预算 ≤2k(§12.3)，塞太多细节挤占两侧结论的空间。
        #
        # ⚠️ 真实跑通时两次踩到同一个坑，第二次才找到根：这里原来把 rule_id
        # (比如 "K01")显式写进这段文本里——先是 "- [K01] ..."，模型把方括号
        # 包着的短 id 当成可引用对象，套进 `[source: K01]`；改成不带方括号的
        # "规则K01：" 后，模型换了个方式复现同一个错误——直接把"规则K01"这串
        # 文本原样当成 id 塞进 `[source: 规则K01]`(这次因为带中文，绕过了
        # CITATION_PATTERN 只认 `[A-Za-z0-9_\-]+` 的确定性检查，是核查 pass
        # 的 LLM 软判定才拦下来的，不是零成本地被挡住)。两次教训是同一件事：
        # 只要这段文本里出现任何"看起来像可以指着说的短码"的 token，模型就有
        # 概率把它套进引用模板——光靠 rubric 里加一句"不要引用规则"这种说明性
        # 指令堵不住，LLM 不是 100% 遵守文字指令的确定性系统。真正的修法是
        # 不给它这个东西：rule_id 完全不出现在喂给模型的文本里，只用于内部
        # 数据结构和日志(`stage_log` 的 `matched_rule_ids` 字段读的是原始
        # `matched_rules` 列表，不依赖这里的展示文本，不受影响)。模型只需要
        # 知道"要遵循这条要求"，不需要知道"这条要求编号是几"。
        topic = rule.get("topic", "")
        relation = rule.get("relation", "?")
        resolution = rule.get("resolution", "")
        lines.append(f"- {topic} · relation={relation} · resolution={resolution}")
    return "\n".join(lines)


def _build_user_message(
    tcm_conclusion: str,
    nutrition_conclusion: str,
    user_profile: dict[str, Any] | None,
    matched_rules: list[dict[str, Any]] | None,
    avoid_note: str | None = None,
) -> str:
    parts = [
        "## 用户画像\n" f"{_format_user_profile(user_profile)}",
        "## 中医食养角度的结论\n" f"{tcm_conclusion}",
        "## 营养学角度的结论\n" f"{nutrition_conclusion}",
        "## 命中的冲突规则\n" f"{_format_matched_rules(matched_rules)}",
    ]
    if avoid_note:
        # 重试场景专用小节(见模块文档)——只在调用方检测到上一版有问题、
        # 带着具体反馈重新调用时才会出现，正常首次调用不会有这一段。
        parts.append("## 重新生成要求\n" f"{avoid_note}")
    return "\n\n".join(parts)


async def reconcile(
    tcm_conclusion: str,
    nutrition_conclusion: str,
    *,
    user_profile: dict[str, Any] | None = None,
    matched_rules: list[dict[str, Any]] | None = None,
    avoid_note: str | None = None,
    complete: CompleteFn | None = None,
    locale: str = "zh",
) -> ReconciliationResult:
    """一次调和层调用。`tcm_conclusion`/`nutrition_conclusion` 必须是两侧
    SubAgent 的最终结论文本(含它们自己产出的 `[source: chunk_id]` 引用标记)，
    不是原始检索 chunk 原文——调用方负责保证这一点，本函数不做内容嗅探。

    `avoid_note`：重试专用(见模块文档)——调用方检测到上一次调用的输出有问题
    (目前是过敏原命中)时，把"要避开什么"的具体反馈传进来，拼进"重新生成要求"
    小节。为空(默认)时和之前完全一样，不影响首次调用的行为。
    """
    complete = complete or llm_adapter.complete
    rule_ids = [r.get("rule_id") for r in (matched_rules or []) if r.get("rule_id")]

    logger.info(
        "Reconciliation input · tcm_conclusion_chars=%d · nutrition_conclusion_chars=%d · "
        "matched_rules=%d · has_user_profile=%s · is_retry=%s "
        "(D14: 仅两侧结论文本，不含原始检索 chunk)",
        len(tcm_conclusion), len(nutrition_conclusion),
        len(matched_rules or []), user_profile is not None, bool(avoid_note),
    )

    t0 = time.perf_counter()
    with observation(
        "reconcile",
        as_type="span",
        input={
            "tcm_conclusion": redact_text(tcm_conclusion),
            "nutrition_conclusion": redact_text(nutrition_conclusion),
            "matched_rule_ids": rule_ids,
            "has_user_profile": user_profile is not None,
            "is_retry": bool(avoid_note),
        },
    ):
        system_prompt = apply_language_instruction(
            compose_prompt_with_skills(_BASE_SYSTEM_PROMPT, [_SKILL_RECONCILIATION_RUBRIC]),
            locale,
        )
        user_message = _build_user_message(
            tcm_conclusion, nutrition_conclusion, user_profile, matched_rules, avoid_note
        )

        # D19 修订(2026-08-27)：调和层调用不再强制 force_prod_tier，跟随
        # MODEL_TIER 和其余调用一致——原例外要求本地开发环境额外配置一套 prod
        # 档凭据，否则调和层直接因鉴权失败报错；改成跟随 MODEL_TIER 后，正式
        # 跑分/交付时把 MODEL_TIER 设成 prod 档即可达到同样效果，不需要单独例外。
        result = await complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        update_current(
            output={"text": redact_text(result.text), "model": result.model},
            metadata={
                "latency_ms": round(latency_ms, 1),
                "matched_rule_ids": rule_ids,
                "tier": result.tier.value,
                "provider": result.provider,
                "cost_est": result.cost_est,
                "tokens": result.usage.total_tokens if result.usage else None,
            },
        )
        logger.info(
            "Reconciliation done · model=%s · tier=%s · provider=%s",
            result.model, result.tier.value, result.provider,
        )
        stage_log(
            logger,
            "reconcile",
            latency_ms=round(latency_ms, 1),
            tokens=result.usage.total_tokens if result.usage else None,
            cost_est=result.cost_est,
            matched_rule_ids=rule_ids,
            model=result.model,
        )
        return ReconciliationResult(
            text=result.text, model=result.model, tier=result.tier, provider=result.provider
        )


async def reconcile_subagent_results(
    tcm_result: SubAgentResult,
    nutrition_result: SubAgentResult,
    *,
    user_profile: dict[str, Any] | None = None,
    matched_rules: list[dict[str, Any]] | None = None,
    avoid_note: str | None = None,
    complete: CompleteFn | None = None,
    locale: str = "zh",
) -> ReconciliationResult:
    """薄封装：只从两侧 `SubAgentResult` 取 `.final_text`，`.messages`(原始
    tool_result，含检索到的 chunk 原文)明确不转发——打日志记录被丢弃的条数，
    作为"调和层确实没收到原始 chunk"这条完成判据的可核查证据。

    `avoid_note` 透传给 `reconcile()`，见该函数文档。
    """
    logger.info(
        "Reconciliation: extracting from SubAgentResult · "
        "tcm domain=%s tools_called=%s messages=%d(discarded, not forwarded) · "
        "nutrition domain=%s tools_called=%s messages=%d(discarded, not forwarded)",
        tcm_result.domain, tcm_result.tools_called, len(tcm_result.messages),
        nutrition_result.domain, nutrition_result.tools_called, len(nutrition_result.messages),
    )
    return await reconcile(
        tcm_result.final_text,
        nutrition_result.final_text,
        user_profile=user_profile,
        matched_rules=matched_rules,
        avoid_note=avoid_note,
        complete=complete,
        locale=locale,
    )
