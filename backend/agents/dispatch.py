#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
七选一分支分发的核心业务逻辑：`fact_query`/`single_domain`(单 SubAgent)、
`candidate_eval`/`full_recommend`(双 SubAgent + 调和 + 过敏原重试 + 核查)、
`other`(闲聊快速通道)，以及多任务场景下的逐个分发。`log_write`/`log_review`
两条分支自成一体，分别在 `backend/agents/{log_write,log_review}.py`。

设计依据：docs/ARCHITECTURE.md §5.2/§5.3；决策依据：docs/DECISIONS.md D14/D15/D20
硬约束（BUILD_PLAN 阶段4 #11 完成判据）：**核查 pass 必须在第一条 `token` 事件
之前完成**——这里靠代码结构强制，不是靠约定：下面每一条分支都是先把派发/
调和/核查全部跑完、拿到最终确定的文本，才第一次 `yield` 任何 SSE 事件；`verify()`
之后如果全部条目被拒绝，吐 `guardrail` + 兜底文本(`_stream_verification_result`
负责组装)。仅过敏原命中时保留原始建议并追加安全提示；ED/诊断性表述等其他
硬阻断仍不展示原始内容，避免真正的高风险文本被放出。

2026-08-28：从 api/main.py 拆出——那个文件原本同时装着 FastAPI 路由/DI wiring
和这一整套分发/调和/核查业务逻辑，后者和 FastAPI 完全无关，纯粹是"给定一个
`RouteDecision` 怎么产出 SSE 事件序列"的领域逻辑。纯粹搬文件，不改变任何
函数签名/行为。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Awaitable, Callable

from api.schemas import ChatRequest
from backend.agents._subagent_common import SubAgentResult
from backend.agents.citation import extract_clarification_question, strip_citation_markers
from backend.agents.clarification import ClarificationStore, PendingClarification
from backend.agents.conflict_gaps import record_conflict_gap
from backend.agents.log_review import stream_log_review
from backend.agents.log_write import stream_log_write
from backend.agents.profile_write import stream_profile_write
from backend.agents.agent_loop import AgentLoopResourceLimitError
from backend.agents.naive_rag_fallback import run_naive_rag_fallback
from backend.agents.nutrition_subagent import is_recipe_assembly_request, run_nutrition_subagent
from backend.agents.reconciliation import ReconciliationResult, reconcile_subagent_results
from backend.agents.routing import CompleteFn, MultiTaskCandidate, RouteBranch, RouteDecision
from backend.agents.sse import chunk_text, sse_event
from backend.agents.tcm_subagent import run_tcm_subagent
from backend.agents.timeouts import reraise_if_cancelled
from backend.agents.user_context import UserProfileContext
from backend.agents.verification import (
    BranchMode,
    SuggestionItem,
    VerificationResult,
    repair_insufficient_evidence,
    verify,
)
from backend.exceptions import (
    LLMCallError,
    ResourceLimitError,
    SubAgentTimeoutError,
)
from backend.guardrails.output_filters import check_allergens
from backend.i18n import apply_language_instruction, t
from backend.llm.adapter import ModelTier
from backend.mcp_server.server import DietExpertMcpServer
from backend.memory.critical_fact_scanner import CriticalFactScanResult
from backend.memory.pending_critical_facts import PendingCriticalFactStore
from backend.observability.cost import current_request_cost
from backend.observability.redact import redact_text
from backend.observability.tracing import observation, stage_log, update_current

logger = logging.getLogger("diet_expert.agents.dispatch")

ConflictRulesFetcher = Callable[[list, list], list]

# ---------------------------------------------------------------------------
# 2026-09-01：过程可见性(process visibility)——路由/派发/调和/核查全部跑完
# 才第一次 yield token/source/done，用户在这之前(有时是两个并行 LLM 调用 +
# 一次调和 + 一次核查)完全看不到任何反馈。`stage` 是新增的第五类 SSE 事件，
# 只携带"现在在跑哪个阶段"，不携带任何生成文本——`_stage_event()`在
# `status="start"`时核查/调和结果显然还没算出来，但反正这里从不传生成内容，
# 不会碰到模块文档开头那条"核查必须在第一条 token 事件前完成"的约束。
# 复用同一条 detail 文案给 start/done 两种状态，前端靠 `status` 字段区分，
# 不需要为每个阶段各配一句"开始"和一句"完成"。
# ---------------------------------------------------------------------------


def _stage_event(stage: str, status: str, locale: str) -> str:
    return sse_event(
        "stage", {"stage": stage, "status": status, "detail": t(f"dispatch.stage_{stage}", locale)}
    )

# ---------------------------------------------------------------------------
# 人在环追问(D20 五处 agent 行为点第3条"记录解析追问"，2026-08-27 实现并扩展
# 覆盖 candidate_eval/single_domain/fact_query/full_recommend 四个走 SubAgent
# 的分支——完整设计见 docs/ARCHITECTURE.md 新增小节、docs/DECISIONS.md D20
# 更新说明。SubAgent 侧的标记约定见 backend/agents/citation.py
# `build_clarification_instruction()`/`extract_clarification_question()`。
# ---------------------------------------------------------------------------


def _clarification_events_for(
    final_text: str,
    *,
    decision_branch: RouteBranch,
    domain_hint: str | None,
    request: ChatRequest,
    trace_id: str,
    clarification_store: ClarificationStore,
    allow_clarification: bool,
) -> list[str] | None:
    """`final_text` 命中 `[NEED_CLARIFICATION]` 标记时，返回调用方应该原样
    yield 完就 `return` 的 SSE 事件列表(不再继续走核查/调和)；没命中返回
    `None`，调用方按原逻辑继续处理，行为和这条机制加入之前完全一样。

    `allow_clarification=False`(已经问过一次的重试轮，见 PRD §11"追问一次，
    仍模糊则记为 unspecified")时，即使又命中标记也不再问第二次，直接判定
    为信息仍然不足。"""
    question = extract_clarification_question(final_text)
    if question is None:
        return None

    if not allow_clarification:
        return [
            sse_event(
                "guardrail",
                {
                    "type": "clarification_unresolved",
                    "detail": t("dispatch.clarification_unresolved", request.locale),
                },
            ),
            sse_event("done", {"trace_id": trace_id}),
        ]

    clarification_store.put(
        request.session_id,
        PendingClarification(
            original_text=request.message, branch=decision_branch, domain_hint=domain_hint
        ),
    )
    events = [sse_event("clarification", {"question": question})]
    events.extend(sse_event("token", {"text": chunk}) for chunk in chunk_text(question))
    events.append(sse_event("done", {"trace_id": trace_id}))
    return events


# 2026-08-31：用户反馈追问第二次仍不够时，前端拿到的是一句只有 guardrail 徽标、
# 没有任何 token 的空气泡——`clarification_unresolved` 那条 detail 文案只在
# guardrail 里，不会当正文吐出来(同 `_VERIFICATION_REJECTED_FALLBACK_MESSAGE`
# 加之前"全被拦截时前端是一个空气泡"的同一类问题)。PRD §11"追问一次，仍模糊
# 则记为 unspecified"这条上限本身不该动(不能真的问第三次)，但"问不出更多信息"
# 不等于"不能给出任何回答"——换一条指令强制 SubAgent 基于已知信息给一个
# 尽力而为的回答(可以假设，但必须标注)，比让用户对着一句"信息不足"卡住更有用。
_FORCE_ANSWER_NOTE = (
    "【用户已经补充过一次信息，仍不够具体，不能再追问第三次。请不要再输出 "
    "[NEED_CLARIFICATION]，改为基于目前已知的全部信息给出一个尽力而为的回答——"
    "可以做合理假设，但必须在回答里明确指出哪些地方是假设、建议用户自行确认，"
    "不要装作信息已经完备。】"
)


async def _force_clarifying_side_to_answer(
    result: SubAgentResult,
    task_input: str,
    rerun: Callable[[str], Awaitable[SubAgentResult]],
) -> SubAgentResult:
    """SubAgent 只吐 `[NEED_CLARIFICATION]` 时，换 `_FORCE_ANSWER_NOTE` 再跑一次，
    要求基于已有信息尽力回答。仍只吐问题时原样返回 `result`。"""
    if extract_clarification_question(result.final_text) is None:
        return result
    try:
        forced = await rerun(f"{task_input}\n\n{_FORCE_ANSWER_NOTE}")
    except SubAgentTimeoutError:
        return result
    if extract_clarification_question(forced.final_text) is not None:
        return result
    return forced


async def _resolve_unresolved_clarification(
    result: SubAgentResult,
    task_input: str,
    allow_clarification: bool,
    rerun: Callable[[str], Awaitable[SubAgentResult]],
) -> SubAgentResult:
    """`allow_clarification=False`(这是对上一轮追问的重试回答)时，如果这个
    SubAgent 还是只吐 `[NEED_CLARIFICATION]`，不能再问第三次，但也不该让用户
    卡在一句空气泡前——换一条指令(`_FORCE_ANSWER_NOTE`)再跑一次同一个
    SubAgent，强制它基于已有信息给出尽力而为的回答。如果它依然不听指令、第二次
    还是只吐问题，原样返回原始 `result`——调用方走既有的 `_clarification_events_for`
    死路兜底，这是真正给不出任何回答时唯一诚实的选择，不强行把一句"问题"包装成
    "答案"展示给用户。`allow_clarification=True`(第一次追问)或压根没命中标记时
    原样返回，不产生任何额外调用。"""
    if allow_clarification or extract_clarification_question(result.final_text) is None:
        return result
    return await _force_clarifying_side_to_answer(result, task_input, rerun)


# The allergen reconciliation retry remains bounded at one call. Evidence
# failures now use one no-tool repair call, avoiding repeated retrieval and
# SubAgent cost.
_ALLERGEN_RECONCILIATION_RETRY_LIMIT = 1

# SubAgent failures that should degrade instead of bubbling out of dispatch.
_SUBAGENT_DEGRADE_ERRORS = (
    LLMCallError,
    ResourceLimitError,
    AgentLoopResourceLimitError,
)


def _profile_notes(profile: UserProfileContext | None) -> str:
    return profile.profile_prompt_notes() if profile else ""


# Evidence failures (missing/invalid source ids or unsupported citation
# relevance) use the no-tool repair call. Allergen, ED, and diagnostic
# violations remain hard blocks and cannot be bypassed by repair.
_RETRYABLE_CHECK_NUMBERS = frozenset({1, 2})


def _should_retry_insufficient_evidence(verification: VerificationResult) -> bool:
    if verification.needs_reconciliation_retry:
        return True
    if verification.accepted:
        return False
    return any(
        rejected.check_number in _RETRYABLE_CHECK_NUMBERS for rejected in verification.rejected
    )


async def _repair_after_verification(
    verification: VerificationResult,
    draft: str,
    subagent_results: list[SubAgentResult],
    complete: CompleteFn,
    *,
    locale: str,
) -> VerificationResult:
    """Recover evidence failures without rerunning agents or retrieval."""
    if not _should_retry_insufficient_evidence(verification):
        return verification
    reasons = [rejected.reason for rejected in verification.rejected]
    repaired = await repair_insufficient_evidence(
        draft,
        reasons,
        available_source_ids=_available_source_ids(*subagent_results),
        complete=complete,
        locale=locale,
    )
    if repaired is None:
        return verification
    return VerificationResult(
        accepted=[repaired],
        rejected=verification.rejected,
        branch=verification.branch,
        skill_in_prompt=verification.skill_in_prompt,
        system_prompt=verification.system_prompt,
        llm_raw=verification.llm_raw,
    )


def _build_allergen_avoid_note(avoid_terms: list[str]) -> str:
    """构造过敏原重试的具体反馈文本——不是含糊地说"重试"，而是明确指出上一版
    命中了什么、该怎么处理(调整食材/调味，而不是放弃整道菜)，呼应
    `backend/agents/_subagent_common.py` 的 `build_allergen_avoidance_instruction()`
    在生成阶段用的同一条原则。"""
    terms = "、".join(avoid_terms)
    return (
        f"上一版建议里出现了用户过敏的成分或其常见隐藏来源：{terms}。"
        "请重新生成一版完全不提及这些成分的建议——优先调整食材或调味来避开"
        "（比如省略或替换某个调料），而不是因为用到这些成分就放弃整道菜的推荐；"
        "只有确实找不到安全替代时，才不推荐那道菜。"
    )


_RETRIEVAL_TOOL_NAMES = {"retrieve_tcm", "retrieve_nutrition"}


def _available_source_ids(*results: SubAgentResult) -> set[str]:
    """从 SubAgent 真实执行过的检索工具结果里，取出可信的 source_id 集合，
    供核查 pass 判断"这条引用是不是幻觉"。依赖 backend/agents/agent_loop.py
    `_json_default` 那处修复——工具结果里的 RetrievedChunk 现在是结构化 JSON
    对象（有 source_id 字段），不是 repr 字符串。"""
    ids: set[str] = set()
    for result in results:
        for m in result.messages:
            if m.get("role") != "tool" or not m.get("ok"):
                continue
            if m.get("name") not in _RETRIEVAL_TOOL_NAMES:
                continue
            try:
                payload = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, list):
                continue
            for chunk in payload:
                if isinstance(chunk, dict) and chunk.get("source_id"):
                    ids.add(chunk["source_id"])
    return ids


async def _run_verification(
    subagent_results: list[SubAgentResult],
    final_text: str,
    branch_mode: BranchMode,
    complete: CompleteFn,
    user_profile_summary: str = "",
    user_allergens: tuple[str, ...] = (),
    skip_allergen_check: bool = False,
    locale: str = "zh",
):
    """`_verify_and_stream()` 的非流式核心——单独抽出来是因为
    `_stream_dual_dispatch()` 的过敏原重试需要在"决定要不要吐 SSE 事件"之前先
    拿到 `VerificationResult` 本身做判断(见该函数的重试逻辑)，不能在一个
    async generator 里"先偷看结果再决定要不要真的 yield"。

    `user_profile_summary`(§5.2 步骤6:核查 pass 输入包含 user_profile)由调用方
    传入——`verify()` 本身早就支持这个参数，此前一直没人传过(2026-08-26 接线)。

    `user_allergens`：THREAT_MODEL.md E2("过敏原经别名/隐藏成分漏出"，此前是
    "真空"——`verify()` 支持这个参数但从没人传过真实数据)。现在 `profile`
    已经在 `_stream_chat` 里查出来了，`UserProfileContext.allergens` 直接就是
    `verify()` 需要的形状，这里补上这一条转发，E2 的"确定性集合比对"这条控制
    才算真的在跑，不是只有代码存在但没人调用。"""
    available_ids = _available_source_ids(*subagent_results)
    item = SuggestionItem(text=final_text)
    return await verify(
        [item],
        available_source_ids=available_ids,
        branch=branch_mode,
        complete=complete,
        user_profile_summary=user_profile_summary,
        user_allergens=user_allergens,
        skip_allergen_check=skip_allergen_check,
        locale=locale,
    )


async def _stream_verification_result(
    verification, trace_id: str, locale: str = "zh"
) -> AsyncIterator[str]:
    """把一个已经算好的 `VerificationResult` 变成 SSE 事件序列：(通过则)吐
    source + token，(不通过则)吐 guardrail + 兜底文本。仅过敏原命中、且无其他
    硬阻断时：先展示原文，再附安全提示——命中常常是「提到/举例」而非直接
    推荐食用，由用户结合配料表自行判断；其他硬阻断仍不展示原始内容。
    **这是"核查必须在第一条 token 事件前完成"这条约束真正被强制的地方**——
    调用方必须先 `await _run_verification()` 拿到结果才能调这个函数，不可能
    绕过这个顺序；兜底提示本身是在核查结果已经确定之后才组装的静态文本，
    不依赖、也不暴露任何未经核查的内容，不违反这条约束。"""
    if not verification.accepted:
        allergen_rejected = [
            rejected for rejected in verification.rejected if rejected.check_number == 4
        ]
        other_rejected = [
            rejected for rejected in verification.rejected if rejected.check_number != 4
        ]
        yield sse_event(
            "guardrail",
            {"type": "verification_rejected", "detail": t("dispatch.verification_rejected", locale)},
        )
        for rejected in verification.rejected:
            reason = (
                rejected.reason
                if locale != "en"
                else t("dispatch.rejected_item", locale, check_number=rejected.check_number)
            )
            yield sse_event(
                "guardrail",
                {"type": "rejected_item", "check_number": rejected.check_number, "reason": reason},
            )

        if allergen_rejected and not other_rejected:
            for rejected in allergen_rejected:
                for chunk in chunk_text(strip_citation_markers(rejected.item.text)):
                    yield sse_event("token", {"text": chunk})
            allergen_names = list(
                dict.fromkeys(
                    name
                    for rejected in allergen_rejected
                    for name in rejected.allergen_names
                )
            )
            separator = "、" if locale != "en" else ", "
            fallback_text = t(
                "dispatch.allergen_fallback",
                locale,
                allergens=separator.join(allergen_names)
                or t("dispatch.unknown_allergen", locale),
            )
        else:
            fallback_text = t("dispatch.verification_fallback", locale)
        for chunk in chunk_text(fallback_text):
            yield sse_event("token", {"text": chunk})
        yield sse_event("done", {"trace_id": trace_id})
        return

    accepted_item = verification.accepted[0]
    for source_id in accepted_item.resolved_source_ids():
        yield sse_event("source", {"source_id": source_id})
    # 引用标记本身(`[source: chunk_id]`)只在核查 pass 验证阶段有用——核查
    # 已经在上面 `verify()` 里跑完了，`source_id` 列表也已经通过独立的
    # `source` 事件吐给前端(溯源可展开)，这里再把机器可读标记原样混进用户
    # 看的正文里没有意义，剥离掉只留自然语言。
    for chunk in chunk_text(strip_citation_markers(accepted_item.text)):
        yield sse_event("token", {"text": chunk})
    yield sse_event("done", {"trace_id": trace_id})


async def _verify_and_stream(
    subagent_results: list[SubAgentResult],
    final_text: str,
    branch_mode: BranchMode,
    trace_id: str,
    complete: CompleteFn,
    user_profile_summary: str = "",
    user_allergens: tuple[str, ...] = (),
    locale: str = "zh",
) -> AsyncIterator[str]:
    """所有分支共用的收尾：核查 → 流式吐出结果。薄封装
    `_run_verification()` + `_stream_verification_result()`——`_stream_dual_dispatch()`
    的过敏原重试路径需要拆开这两步单独调用，其余三个调用方(单领域/事实查询、
    两侧 SubAgent 部分失败的降级路径)不需要重试，直接用这个组合版本。

    2026-09-01：`verify` stage 事件包在这里而不是每个调用方各吐一遍——四个
    调用方(部分失败降级 ×2、追问强制重试后仍单边成功的降级 ×2)复用的都是
    这一个函数，包一次就覆盖全部四处。"""
    yield _stage_event("verify", "start", locale)
    verification = await _run_verification(
        subagent_results, final_text, branch_mode, complete,
        user_profile_summary=user_profile_summary, user_allergens=user_allergens,
        locale=locale,
    )
    verification = await _repair_after_verification(
        verification, final_text, subagent_results, complete, locale=locale
    )
    yield _stage_event("verify", "done", locale)
    async for chunk in _stream_verification_result(verification, trace_id, locale=locale):
        yield chunk


# D27 补充(2026-08-28，backend/memory/session_store.py 接线)：把
# `load_session_history()` 组装出的会话历史接进 SubAgent 的 task_input——
# 接给两个真的会派发 SubAgent 的分支(fact_query/single_domain/
# candidate_eval/full_recommend)，log_write/log_review 两条分支不需要跨轮次
# 上下文也能正确处理这一轮，不额外接线。历史是"供参考的背景"，用单独小节和
# 分隔语隔开，不直接拼接——避免 SubAgent 把历史里的旧结论误当成这次问题本身
# 的一部分去回答。
# 2026-08-31 补充：`other` 原本也在"不需要"这一类里，实测发现路由器偶尔会把
# 依赖上文的含糊续问误判成 other，这条分支因为完全没有历史而答得像失忆——
# 见 `dispatch_branch()` OTHER 分支的注释，现在也接上了同一份 session_history。
async def _gather_dual_subagents(
    task_input: str,
    server: DietExpertMcpServer,
    complete: CompleteFn,
    *,
    constitution: str | None,
    allergens: tuple[str, ...],
    extra_notes: str,
    include_recipe: bool,
    user_id: str = "default_user",
    locale: str = "zh",
):
    tcm_result, nutrition_result = await asyncio.gather(
        run_tcm_subagent(
            task_input,
            server,
            constitution=constitution,
            allergens=allergens,
            extra_profile_notes=extra_notes,
            complete=complete,
            user_id=user_id,
            locale=locale,
        ),
        run_nutrition_subagent(
            task_input,
            server,
            allergens=allergens,
            extra_profile_notes=extra_notes,
            include_recipe_skill=include_recipe,
            complete=complete,
            user_id=user_id,
            locale=locale,
        ),
        return_exceptions=True,
    )
    reraise_if_cancelled(tcm_result, nutrition_result)
    return tcm_result, nutrition_result


def _compose_task_input(message: str, session_history: str) -> str:
    if not session_history:
        return message
    return (
        f"【近期对话历史，仅供参考背景，不是本次问题本身】\n{session_history}\n\n"
        f"【本次用户消息，只回答这一句】\n{message}"
    )


async def _run_single_subagent(
    domain: str,
    task_input: str,
    server: DietExpertMcpServer,
    complete: CompleteFn,
    *,
    constitution: str | None,
    allergens: tuple[str, ...],
    extra_notes: str,
    user_id: str,
    locale: str,
):
    if domain == "tcm":
        return await run_tcm_subagent(
            task_input,
            server,
            constitution=constitution,
            allergens=allergens,
            extra_profile_notes=extra_notes,
            complete=complete,
            user_id=user_id,
            locale=locale,
        )
    return await run_nutrition_subagent(
        task_input,
        server,
        allergens=allergens,
        extra_profile_notes=extra_notes,
        complete=complete,
        user_id=user_id,
        locale=locale,
    )


async def _stream_naive_rag_single_domain(
    *,
    request: ChatRequest,
    decision: RouteDecision,
    task_input: str,
    complete: CompleteFn,
    trace_id: str,
    profile: UserProfileContext | None,
    locale: str,
) -> AsyncIterator[str]:
    """PRD §11 naive RAG fallback for single-domain SubAgent failure."""
    constitution = profile.constitution if profile else None
    allergens = profile.allergens if profile else ()
    extra_notes = _profile_notes(profile)
    summary = profile.to_verification_summary() if profile else ""
    branch_mode: BranchMode = (
        "fact_query" if decision.branch is RouteBranch.FACT_QUERY else "single_domain"
    )
    yield sse_event(
        "guardrail",
        {"type": "naive_rag_fallback", "detail": t("dispatch.naive_rag_fallback", locale)},
    )
    yield _stage_event("naive_rag_fallback", "start", locale)
    try:
        naive_result = await run_naive_rag_fallback(
            task_input,
            complete,
            retrieval_query=request.message,
            constitution=constitution,
            allergens=allergens,
            extra_profile_notes=extra_notes,
            locale=locale,
        )
    except Exception as exc:
        logger.error(
            "single-domain naive RAG fallback failed · trace_id=%s · error=%s",
            trace_id,
            exc,
        )
        yield sse_event(
            "guardrail",
            {"type": "subagent_failed", "detail": t("dispatch.subagent_failed", locale)},
        )
        yield sse_event("done", {"trace_id": trace_id})
        return
    yield _stage_event("naive_rag_fallback", "done", locale)
    async for chunk in _verify_and_stream(
        [naive_result],
        naive_result.final_text,
        branch_mode,
        trace_id,
        complete,
        user_profile_summary=summary,
        user_allergens=allergens,
        locale=locale,
    ):
        yield chunk


async def _stream_single_domain(
    request: ChatRequest,
    decision: RouteDecision,
    server: DietExpertMcpServer,
    complete: CompleteFn,
    trace_id: str,
    profile: UserProfileContext | None,
    clarification_store: ClarificationStore,
    allow_clarification: bool = True,
    session_history: str = "",
) -> AsyncIterator[str]:
    domain = decision.domain_hint
    if domain is None:
        logger.warning(
            "no domain_hint for branch=%s, defaulting to nutrition · trace_id=%s",
            decision.branch.value, trace_id,
        )
        domain = "nutrition"

    constitution = profile.constitution if profile else None
    allergens = profile.allergens if profile else ()
    extra_notes = _profile_notes(profile)
    locale = request.locale
    task_input = _compose_task_input(request.message, session_history)
    subagent_stage = f"subagent_{domain}"
    yield _stage_event(subagent_stage, "start", locale)
    try:
        result = await _run_single_subagent(
            domain, task_input, server, complete,
            constitution=constitution, allergens=allergens, extra_notes=extra_notes,
            user_id=request.user_id, locale=locale,
        )
    except SubAgentTimeoutError:
        logger.warning(
            "single-domain subagent timed out · domain=%s · trace_id=%s", domain, trace_id
        )
        yield sse_event(
            "stage",
            {
                "stage": subagent_stage,
                "status": "error",
                "detail": t(f"dispatch.stage_{subagent_stage}", locale),
            },
        )
        yield sse_event(
            "guardrail",
            {"type": "subagent_timeout", "detail": t("dispatch.subagent_timeout", locale)},
        )
        yield sse_event("done", {"trace_id": trace_id})
        return
    except _SUBAGENT_DEGRADE_ERRORS as exc:
        logger.warning(
            "single-domain subagent failed · domain=%s · trace_id=%s · error=%s",
            domain,
            trace_id,
            exc,
        )
        yield sse_event(
            "stage",
            {
                "stage": subagent_stage,
                "status": "error",
                "detail": t(f"dispatch.stage_{subagent_stage}", locale),
            },
        )
        async for chunk in _stream_naive_rag_single_domain(
            request=request,
            decision=decision,
            task_input=task_input,
            complete=complete,
            trace_id=trace_id,
            profile=profile,
            locale=locale,
        ):
            yield chunk
        return
    except Exception as exc:
        logger.warning(
            "single-domain subagent unexpected failure · domain=%s · trace_id=%s · error=%s",
            domain,
            trace_id,
            exc,
            exc_info=True,
        )
        yield sse_event(
            "stage",
            {
                "stage": subagent_stage,
                "status": "error",
                "detail": t(f"dispatch.stage_{subagent_stage}", locale),
            },
        )
        async for chunk in _stream_naive_rag_single_domain(
            request=request,
            decision=decision,
            task_input=task_input,
            complete=complete,
            trace_id=trace_id,
            profile=profile,
            locale=locale,
        ):
            yield chunk
        return

    # 强制重试(追问一次仍不够信息时的 `_FORCE_ANSWER_NOTE` 那一轮)算同一次
    # SubAgent 调度的延续，不单独再吐一组 start/done——done 放在这两步都完成
    # 之后，对前端来说"这一侧分析"就是一个不可再分的阶段。
    result = await _resolve_unresolved_clarification(
        result,
        task_input,
        allow_clarification,
        lambda forced_input: _run_single_subagent(
            domain, forced_input, server, complete,
            constitution=constitution, allergens=allergens, extra_notes=extra_notes,
            user_id=request.user_id, locale=locale,
        ),
    )
    yield _stage_event(subagent_stage, "done", locale)
    clarification_events = _clarification_events_for(
        result.final_text,
        decision_branch=decision.branch,
        domain_hint=decision.domain_hint,
        request=request,
        trace_id=trace_id,
        clarification_store=clarification_store,
        allow_clarification=allow_clarification,
    )
    if clarification_events is not None:
        for event in clarification_events:
            yield event
        return

    branch_mode: BranchMode = (
        "fact_query" if decision.branch is RouteBranch.FACT_QUERY else "single_domain"
    )
    summary = profile.to_verification_summary() if profile else ""
    yield _stage_event("verify", "start", locale)
    verification = await _run_verification(
        [result], result.final_text, branch_mode, complete,
        user_profile_summary=summary, user_allergens=allergens, locale=locale,
    )
    verification = await _repair_after_verification(
        verification, result.final_text, [result], complete, locale=locale
    )
    yield _stage_event("verify", "done", locale)
    async for chunk in _stream_verification_result(verification, trace_id, locale=locale):
        yield chunk


async def _stream_dual_dispatch(
    request: ChatRequest,
    decision: RouteDecision,
    server: DietExpertMcpServer,
    complete: CompleteFn,
    trace_id: str,
    profile: UserProfileContext | None,
    conflict_rules_fetcher: ConflictRulesFetcher,
    clarification_store: ClarificationStore,
    allow_clarification: bool = True,
    session_history: str = "",
) -> AsyncIterator[str]:
    constitution = profile.constitution if profile else None
    summary = profile.to_verification_summary() if profile else ""
    allergens = profile.allergens if profile else ()
    extra_notes = _profile_notes(profile)
    locale = request.locale
    include_recipe = (
        decision.branch is RouteBranch.FULL_RECOMMEND
        and is_recipe_assembly_request(request.message)
    )
    task_input = _compose_task_input(request.message, session_history)
    # ENGINEERING §2：return_exceptions=True，一侧失败不能拖垮另一侧。
    # CancelledError 必须再抛出去，否则 gather 会把它收成结果，整链超时/
    # 客户端断开就取消不到还在跑的那一侧。
    cost_before = current_request_cost()
    tokens_before = cost_before.total_tokens if cost_before else 0
    cost_est_before = cost_before.cost_est if cost_before else None
    t0 = time.perf_counter()
    # 2026-09-01：两侧 start 事件在 `asyncio.gather` 派发前一次性吐出——两侧
    # 确实是同时开始跑的，这条事件如实反映。done 事件在 gather 返回之后一次性
    # 吐出，不是各自真正完成的那一刻(`gather` 本身就是"等两个都结束"，想要
    # 两侧各自独立的完成时间需要换成 `asyncio.as_completed`，是一次不必要的
    # 改动——两侧 SubAgent 循环本来就跑得差不多快，拆分带来的体验提升有限，
    # 不值得为了这条进度指示改变派发方式本身)。
    yield _stage_event("subagent_tcm", "start", locale)
    yield _stage_event("subagent_nutrition", "start", locale)
    tcm_result, nutrition_result = await _gather_dual_subagents(
        task_input,
        server,
        complete,
        constitution=constitution,
        allergens=allergens,
        extra_notes=extra_notes,
        include_recipe=include_recipe,
        user_id=request.user_id,
        locale=locale,
    )
    reraise_if_cancelled(tcm_result, nutrition_result)
    tcm_failed = isinstance(tcm_result, Exception)
    nutrition_failed = isinstance(nutrition_result, Exception)
    if tcm_failed:
        yield sse_event(
            "stage",
            {
                "stage": "subagent_tcm",
                "status": "error",
                "detail": t("dispatch.stage_subagent_tcm", locale),
            },
        )
    else:
        yield _stage_event("subagent_tcm", "done", locale)
    if nutrition_failed:
        yield sse_event(
            "stage",
            {
                "stage": "subagent_nutrition",
                "status": "error",
                "detail": t("dispatch.stage_subagent_nutrition", locale),
            },
        )
    else:
        yield _stage_event("subagent_nutrition", "done", locale)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    cost_after = current_request_cost()
    tokens_delta = None
    cost_delta = None
    if cost_after is not None:
        tokens_delta = cost_after.total_tokens - tokens_before
        if cost_after.cost_est is not None:
            cost_delta = cost_after.cost_est - (cost_est_before or 0.0)
    # Pit 3: wall-clock is max(tcm, nutrition); tokens/cost are the sum.
    stage_log(
        logger,
        "dual_dispatch",
        latency_ms=round(wall_ms, 1),
        tokens=tokens_delta,
        cost_est=cost_delta,
        parallel=True,
        cost_is_sum_not_wall=True,
        tcm_failed=tcm_failed,
        nutrition_failed=nutrition_failed,
    )

    if tcm_failed and nutrition_failed:
        logger.error(
            "both subagents failed · trace_id=%s · tcm_error=%s · nutrition_error=%s",
            trace_id, tcm_result, nutrition_result,
        )
        yield sse_event("guardrail", {"type": "both_subagents_failed", "detail": t("dispatch.both_subagents_failed", locale)})
        yield sse_event("done", {"trace_id": trace_id})
        return

    # PRD §11 fallback：单个 SubAgent 失败 → 单边输出并标注。
    single_domain_mode: BranchMode = (
        "candidate_eval" if decision.branch is RouteBranch.CANDIDATE_EVAL else "single_domain"
    )
    if tcm_failed:
        logger.warning("tcm subagent failed, single-sided nutrition output · trace_id=%s · error=%s", trace_id, tcm_result)
        nutrition_result = await _resolve_unresolved_clarification(
            nutrition_result, task_input, allow_clarification,
            lambda forced_input: run_nutrition_subagent(
                forced_input, server, allergens=allergens, extra_profile_notes=extra_notes,
                include_recipe_skill=include_recipe, complete=complete,
                user_id=request.user_id, locale=locale,
            ),
        )
        clarification_events = _clarification_events_for(
            nutrition_result.final_text, decision_branch=decision.branch, domain_hint=None,
            request=request, trace_id=trace_id, clarification_store=clarification_store,
            allow_clarification=allow_clarification,
        )
        if clarification_events is not None:
            for event in clarification_events:
                yield event
            return
        yield sse_event("guardrail", {"type": "partial_failure", "detail": t("dispatch.partial_failure_tcm", locale)})
        async for chunk in _verify_and_stream(
            [nutrition_result], nutrition_result.final_text, single_domain_mode, trace_id, complete,
            user_profile_summary=summary, user_allergens=allergens, locale=locale,
        ):
            yield chunk
        return
    if nutrition_failed:
        logger.warning("nutrition subagent failed, single-sided tcm output · trace_id=%s · error=%s", trace_id, nutrition_result)
        tcm_result = await _resolve_unresolved_clarification(
            tcm_result, task_input, allow_clarification,
            lambda forced_input: run_tcm_subagent(
                forced_input, server, constitution=constitution, allergens=allergens,
                extra_profile_notes=extra_notes, complete=complete,
                user_id=request.user_id, locale=locale,
            ),
        )
        clarification_events = _clarification_events_for(
            tcm_result.final_text, decision_branch=decision.branch, domain_hint=None,
            request=request, trace_id=trace_id, clarification_store=clarification_store,
            allow_clarification=allow_clarification,
        )
        if clarification_events is not None:
            for event in clarification_events:
                yield event
            return
        yield sse_event("guardrail", {"type": "partial_failure", "detail": t("dispatch.partial_failure_nutrition", locale)})
        async for chunk in _verify_and_stream(
            [tcm_result], tcm_result.final_text, single_domain_mode, trace_id, complete,
            user_profile_summary=summary, user_allergens=allergens, locale=locale,
        ):
            yield chunk
        return

    # D20: insufficient context sends the dual path into clarification.
    # - First round + only one side clarifies: force that side to answer instead of
    #   discarding it and degrading to single-sided output.
    # - Retry round (`allow_clarification=False`): force every side that still
    #   clarifies; if one side still refuses after force, degrade to the other.
    run_tcm = lambda forced_input: run_tcm_subagent(
        forced_input, server, constitution=constitution, allergens=allergens,
        extra_profile_notes=extra_notes, complete=complete,
        user_id=request.user_id, locale=locale,
    )
    run_nutrition = lambda forced_input: run_nutrition_subagent(
        forced_input, server, allergens=allergens, extra_profile_notes=extra_notes,
        include_recipe_skill=include_recipe, complete=complete,
        user_id=request.user_id, locale=locale,
    )
    if allow_clarification:
        tcm_wants_clarify = extract_clarification_question(tcm_result.final_text) is not None
        nutrition_wants_clarify = (
            extract_clarification_question(nutrition_result.final_text) is not None
        )
        if tcm_wants_clarify and not nutrition_wants_clarify:
            tcm_result = await _force_clarifying_side_to_answer(
                tcm_result, task_input, run_tcm,
            )
        elif nutrition_wants_clarify and not tcm_wants_clarify:
            nutrition_result = await _force_clarifying_side_to_answer(
                nutrition_result, task_input, run_nutrition,
            )
    else:
        tcm_result = await _resolve_unresolved_clarification(
            tcm_result, task_input, allow_clarification, run_tcm,
        )
        nutrition_result = await _resolve_unresolved_clarification(
            nutrition_result, task_input, allow_clarification, run_nutrition,
        )
    tcm_still_clarifying = extract_clarification_question(tcm_result.final_text) is not None
    nutrition_still_clarifying = (
        extract_clarification_question(nutrition_result.final_text) is not None
    )
    if not allow_clarification and tcm_still_clarifying and not nutrition_still_clarifying:
        logger.warning(
            "tcm still requested clarification after forced retry; "
            "degrading to nutrition-only output · trace_id=%s",
            trace_id,
        )
        yield sse_event(
            "guardrail",
            {"type": "partial_failure", "detail": t("dispatch.partial_failure_tcm", locale)},
        )
        async for chunk in _verify_and_stream(
            [nutrition_result], nutrition_result.final_text, single_domain_mode, trace_id, complete,
            user_profile_summary=summary, user_allergens=allergens, locale=locale,
        ):
            yield chunk
        return
    if not allow_clarification and nutrition_still_clarifying and not tcm_still_clarifying:
        logger.warning(
            "nutrition still requested clarification after forced retry; "
            "degrading to tcm-only output · trace_id=%s",
            trace_id,
        )
        yield sse_event(
            "guardrail",
            {"type": "partial_failure", "detail": t("dispatch.partial_failure_nutrition", locale)},
        )
        async for chunk in _verify_and_stream(
            [tcm_result], tcm_result.final_text, single_domain_mode, trace_id, complete,
            user_profile_summary=summary, user_allergens=allergens, locale=locale,
        ):
            yield chunk
        return
    clarification_events = (
        _clarification_events_for(
            tcm_result.final_text, decision_branch=decision.branch, domain_hint=None,
            request=request, trace_id=trace_id, clarification_store=clarification_store,
            allow_clarification=allow_clarification,
        )
        or _clarification_events_for(
            nutrition_result.final_text, decision_branch=decision.branch, domain_hint=None,
            request=request, trace_id=trace_id, clarification_store=clarification_store,
            allow_clarification=allow_clarification,
        )
    )
    if clarification_events is not None:
        for event in clarification_events:
            yield event
        return

    # §5.2 步骤5:调和层输入包含 user_profile(0.5k)+ 命中的 conflict_rules(≤2k)——
    # 此前这两处一直是 None/未传，2026-08-26 接线，见模块文档开头的说明。
    matched_rules = conflict_rules_fetcher(
        list(profile.constitutions()) if profile else [],
        list(profile.goal_tags) if profile else [],
    )
    if not matched_rules:
        # Both sides produced conclusions and we are about to reconcile
        # without a table hit — PRD §11 conflict_gaps fodder.
        record_conflict_gap(
            trace_id=trace_id,
            constitutions=list(profile.constitutions()) if profile else [],
            goal_tags=list(profile.goal_tags) if profile else [],
        )
    reconciliation_kwargs = dict(
        tcm_result=tcm_result,
        nutrition_result=nutrition_result,
        user_profile=profile.to_reconciliation_dict() if profile else None,
        matched_rules=matched_rules,
    )
    yield _stage_event("reconcile", "start", locale)
    try:
        reconciled = await reconcile_subagent_results(
            **reconciliation_kwargs, complete=complete, locale=locale
        )
    except Exception as exc:
        logger.warning(
            "reconciliation failed, concatenating subagent texts · trace_id=%s · error=%s",
            trace_id,
            exc,
            exc_info=True,
        )
        reconciled = ReconciliationResult(
            text=f"{tcm_result.final_text}\n\n{nutrition_result.final_text}",
            model="",
            tier=ModelTier.DEV,
            provider="fallback",
        )
        yield sse_event(
            "guardrail",
            {
                "type": "reconciliation_failed",
                "detail": t("dispatch.reconciliation_failed", locale),
            },
        )
    branch_mode: BranchMode = (
        "candidate_eval" if decision.branch is RouteBranch.CANDIDATE_EVAL else "full_recommend"
    )

    # DECISIONS.md 待决问题表"核查 pass 退回调和层的最大重试次数(当前设1次)"——
    # 命中过敏原时，不直接把整段回复扔掉，而是带着具体反馈重新调一次调和层
    # （生成前的避让指令见 backend/agents/_subagent_common.py，这里是它没生效
    # 时的第二道保险）。先用便宜的确定性检查(不打 LLM)判断要不要重试；只要
    # 这一轮真的重试过，verify() 就 skip 第 4 条过敏原硬拦（调和层已按避开
    # 说明重写过，扫描器仍会把「今天避开：花生」误判成推荐）。ED / 诊断性
    # 表述 / source_id 其余检查照跑。
    retries_left = _ALLERGEN_RECONCILIATION_RETRY_LIMIT
    allergen_retried = False
    while retries_left > 0:
        allergen_hits = check_allergens(reconciled.text, allergens)
        if not allergen_hits:
            break
        avoid_terms = sorted({f.matched_term for f in allergen_hits})
        logger.warning(
            "reconciliation retry: allergen hit, regenerating with avoidance note · "
            "trace_id=%s · terms=%s · retries_left=%d",
            trace_id, avoid_terms, retries_left,
        )
        try:
            reconciled = await reconcile_subagent_results(
                **reconciliation_kwargs,
                avoid_note=_build_allergen_avoid_note(avoid_terms),
                complete=complete,
                locale=locale,
            )
        except Exception as exc:
            logger.warning(
                "allergen reconciliation retry failed, keeping previous text · trace_id=%s · error=%s",
                trace_id,
                exc,
                exc_info=True,
            )
            break
        allergen_retried = True
        retries_left -= 1
    # 对用户来说这仍然是"两侧结论调和"这一件事，不是调和完了又单独一个
    # "重试"阶段。
    yield _stage_event("reconcile", "done", locale)

    yield _stage_event("verify", "start", locale)
    verification = await _run_verification(
        [tcm_result, nutrition_result], reconciled.text, branch_mode, complete,
        user_profile_summary=summary, user_allergens=allergens,
        skip_allergen_check=allergen_retried,
        locale=locale,
    )
    verification = await _repair_after_verification(
        verification,
        reconciled.text,
        [tcm_result, nutrition_result],
        complete,
        locale=locale,
    )
    yield _stage_event("verify", "done", locale)
    async for chunk in _stream_verification_result(verification, trace_id, locale=locale):
        yield chunk


# D33/PRD §17：不属于其余六个分支的消息(闲聊/问候/完全无关的问题/食物相关但
# 检索不到的常识性问题)。一次直接 complete() 调用，不经过 SubAgent、不经过
# verify()——和 log_review/log_write 一样是简化路径(D20 workflow 精神：不是
# 每个分支都要走七条核查规则，这里本来就没有可核查的检索依据)。
_OTHER_SYSTEM_PROMPT = (
    "你是 diet_expert，一个专注中医食养+营养学建议的饮食助手。用户这句话不属于"
    "你的六个正式功能(记录饮食/查记录/查知识库事实/评估候选食物/单侧专业问题/"
    "综合推荐)之一，按下面的情况分别处理：\n"
    "1. 闲聊、问候、致谢、告别：简短友好地回应，可以顺带一句你能帮忙做什么。\n"
    "2. 食物相关，但不是你知识库检索范围内的问题(比如具体的烹饪做法、一般性"
    "食品安全常识)：可以用你自己的通用知识回答，但**必须**在回答开头或结尾"
    "明确说明「这是通用知识，未经知识库验证」这类话，**不能**使用 "
    "[source: chunk_id] 这种引用标记(那是给经过检索验证的内容用的)。\n"
    "3. 和饮食完全无关的问题(纯问天气怎么样、数学、新闻等，句里没有在问吃什么)："
    "礼貌说明这超出你的范围，引导用户提饮食相关的问题。按今天天气/时令决定吃"
    "什么不属于这一类——那是综合推荐，不应落到本分支；如果落到了，按第 2 条"
    "当食物问题处理，不要用超出范围来回绝。\n"
    "不要在这里派发工具调用、不要生成结构化建议，就是一次简短的自然语言回复。"
)


async def _stream_other(
    request: ChatRequest, complete: CompleteFn, trace_id: str, session_history: str = ""
) -> AsyncIterator[str]:
    locale = request.locale
    unavailable = t("dispatch.other_unavailable", locale)
    try:
        with observation("other", as_type="span", input={"message": redact_text(request.message)}):
            result = await complete(
                [
                    {
                        "role": "system",
                        "content": apply_language_instruction(_OTHER_SYSTEM_PROMPT, locale),
                    },
                    {"role": "user", "content": _compose_task_input(request.message, session_history)},
                ],
                force_prod_tier=False,
            )
            update_current(output={"text": redact_text(result.text or "")})
    except Exception as exc:
        logger.warning(
            "other branch LLM failed · trace_id=%s · error=%s",
            trace_id,
            exc,
            exc_info=True,
        )
        yield sse_event(
            "guardrail",
            {"type": "other_unavailable", "detail": unavailable},
        )
        for chunk in chunk_text(unavailable):
            yield sse_event("token", {"text": chunk})
        yield sse_event("done", {"trace_id": trace_id})
        return
    for chunk in chunk_text(result.text or ""):
        yield sse_event("token", {"text": chunk})
    yield sse_event("done", {"trace_id": trace_id})


async def dispatch_branch(
    request: ChatRequest,
    decision: RouteDecision,
    server: DietExpertMcpServer,
    complete: CompleteFn,
    trace_id: str,
    profile: UserProfileContext | None,
    conflict_rules_fetcher: ConflictRulesFetcher,
    clarification_store: ClarificationStore,
    allow_clarification: bool = True,
    session_history: str = "",
    pending_critical_store: PendingCriticalFactStore | None = None,
    prefetched_fact_scan: CriticalFactScanResult | None = None,
) -> AsyncIterator[str]:
    """七选一分支的完整处理——各自吐一条完整的 token/source/guardrail/done
    序列。单任务路径(api/main.py 的 `_stream_chat_inner`)和多任务路径
    (`stream_multi_task`，D32/§5.1.1)共用这一个函数，不重新实现任何一条分支
    自己的逻辑；区别只在多任务路径会把这里吐出的 `done` 换成 `task_done`
    (见 `stream_multi_task`)。

    `allow_clarification=False` 只在"这是对上一轮追问的重试回答"这一种情况下
    由调用方显式传 False(见 api/main.py 消费 `ClarificationStore` 的那一段)——
    正常调用一律用默认值 True，包括 `stream_multi_task` 的每个子任务。

    `session_history`：`backend/memory/session_store.py` `load_session_history()`
    组装出的会话历史文本，默认空字符串(不接线的既有调用点/单测都不用改)。
    会派发 SubAgent 的四条分支，以及 `other`(2026-08-31 补，见下)都会用到，
    见 `_compose_task_input()`。

    2026-09-01：`stage`("routing", status="done")事件在这里*不*发——这个
    函数拿到的 `decision` 早就是路由已经决定之后的结果，"路由已确定"这条
    进度对调用方(api/main.py `_stream_chat_inner`)来说才是"刚刚发生"的事，
    放在那边（拿到 `decision`/`tasks` 之后、调用这个函数之前）语义上更准确，
    也只需要吐一次，不用在 `dispatch_branch`/`stream_multi_task` 每个分支
    里各吐一遍。"""
    if decision.branch is RouteBranch.OTHER:
        # 2026-08-31：`other` 此前完全不接 session_history，是"纯寒暄/无关问题
        # 不需要上下文"这条设计假设的直接推论——但路由器(尤其是 LLM 兜底那一步)
        # 偶尔会把"依赖上文才有意义的续问"(比如追问未解决之后用户一句含糊的
        # 追问)也分类成 other，这种情况下这条分支就成了名副其实的"失忆"：
        # 只看这一句话，回一个和上文毫无关系的通用问候。接上 session_history
        # 不改变这条分支"不派发工具/不走核查"的简化路径本质，只是让它在被
        # 误分类命中时也不至于把用户晾在一个完全陌生的对话里。
        async for chunk in _stream_other(request, complete, trace_id, session_history=session_history):
            yield chunk
        return
    if decision.branch is RouteBranch.LOG_REVIEW:
        # profile 用来把 logged_at 转换成这个用户自己的时区显示(2026-08-31)——
        # 数据库驱动带回来的原始 tzinfo 只是连接会话的显示时区，不是用户所在地。
        async for chunk in stream_log_review(request, server, trace_id, profile):
            yield chunk
        return
    if decision.branch is RouteBranch.PROFILE_WRITE:
        if pending_critical_store is None:
            raise RuntimeError("profile_write requires pending_critical_store")
        async for chunk in stream_profile_write(
            request,
            trace_id,
            profile,
            complete,
            pending_critical_store,
            prefetched_scan=prefetched_fact_scan,
        ):
            yield chunk
        return
    if decision.branch is RouteBranch.LOG_WRITE:
        async for chunk in stream_log_write(
            request, server, trace_id, profile, complete, clarification_store,
            allow_clarification=allow_clarification,
        ):
            yield chunk
        return
    if decision.branch in (RouteBranch.FACT_QUERY, RouteBranch.SINGLE_DOMAIN):
        async for chunk in _stream_single_domain(
            request, decision, server, complete, trace_id, profile, clarification_store,
            allow_clarification=allow_clarification, session_history=session_history,
        ):
            yield chunk
        return
    # candidate_eval / full_recommend
    async for chunk in _stream_dual_dispatch(
        request, decision, server, complete, trace_id, profile, conflict_rules_fetcher,
        clarification_store, allow_clarification=allow_clarification, session_history=session_history,
    ):
        yield chunk


# `sse_event("done", {...})` 总是产出 "event: done\ndata: ...\n\n" —— 我们是这个
# 格式唯一的生产者，用这个前缀识别"子任务自己吐的 done"是稳定、可控的，不是
# 拿正则去猜第三方格式。
_EVENT_DONE_PREFIX = "event: done\n"


async def stream_multi_task(
    request: ChatRequest,
    tasks: list[MultiTaskCandidate],
    server: DietExpertMcpServer,
    complete: CompleteFn,
    profile: UserProfileContext | None,
    conflict_rules_fetcher: ConflictRulesFetcher,
    trace_id: str,
    clarification_store: ClarificationStore,
    session_history: str = "",
    pending_critical_store: PendingCriticalFactStore | None = None,
    prefetched_fact_scan: CriticalFactScanResult | None = None,
) -> AsyncIterator[str]:
    """D32/§5.1.1：一句话包含多个独立意图时逐个分发。**顺序**执行，不并发——
    写入类(log_write)子任务必须先完整落库，不应该被其他子任务的并发调度
    打断或产生时序上的不确定性。每个子任务复用 `dispatch_branch()`，
    `allow_clarification` 用默认值 True——某个子任务触发追问不影响其他子任务
    照常执行完成；`PendingClarification.original_text` 存的是 `candidate.text`
    (那个子任务切分出来的片段)而不是整条原始消息，见 `segment_request` 的
    构造——恢复时天然只重跑那一个子任务，不重复已经完成的部分。`session_history`
    原样转发给每个子任务，同一条会话历史对所有子任务都是同样的"这轮之前发生
    过什么"背景，不需要按子任务拆分。"""
    for index, candidate in enumerate(tasks):
        yield sse_event(
            "task",
            {
                "index": index,
                "total": len(tasks),
                "branch": candidate.decision.branch.value,
                "text": candidate.text,
            },
        )
        segment_request = request.model_copy(update={"message": candidate.text})
        async for chunk in dispatch_branch(
            segment_request, candidate.decision, server, complete, trace_id, profile,
            conflict_rules_fetcher, clarification_store, session_history=session_history,
            pending_critical_store=pending_critical_store,
            prefetched_fact_scan=prefetched_fact_scan
            if candidate.decision.branch is RouteBranch.PROFILE_WRITE
            else None,
        ):
            if chunk.startswith(_EVENT_DONE_PREFIX):
                yield sse_event("task_done", {"index": index})
            else:
                yield chunk
    yield sse_event("done", {"trace_id": trace_id})
