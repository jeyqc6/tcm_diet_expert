#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCM/Nutrition SubAgent 共用的循环编排——两者除了工具子集(MCP 角色白名单，
backend/mcp_server/roles.py)和领域措辞(各自模块里的 system prompt)之外，
"怎么跑一次 SubAgent 循环"这件事完全一样，抽出来避免两份文件互相漂移。
先例：backend/mcp_server/tools/_retrieval_common.py 对
retrieve_tcm/retrieve_nutrition 做的是同一件事。

设计依据：docs/ARCHITECTURE.md §5.2 步骤 3、§5.4(资源限额/循环防护)
roadmap：阶段 4.2 任务 7

只复用 backend/agents/router.py 的 `run_agent_loop()`(阶段 4.2 任务 5)，
不重新实现一套循环——中枢 agent 和 SubAgent 的循环终止靠同一套 tool_use
有无判断，区别只在这里传的两个参数更严格(§5.4)。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Iterable

from backend.agents.agent_loop import run_agent_loop
from backend.agents.timeouts import subagent_timeout_s
from backend.exceptions import SubAgentTimeoutError
from backend.guardrails.output_filters import hidden_sources_for_allergens
from backend.i18n import normalize_locale, set_current_locale
from backend.llm.adapter import CompleteFn
from backend.mcp_server.roles import CallerRole
from backend.mcp_server.server import DietExpertMcpServer
from backend.memory.status_prompt import build_status_message
from backend.observability.redact import redact_text
from backend.observability.tracing import observation, stage_log, update_current

logger = logging.getLogger("diet_expert.agents.subagent")


def build_allergen_avoidance_instruction(allergens: Iterable[str] | None) -> str:
    """两个 SubAgent 共用——生成阶段就让模型知道要避开什么，而不是等生成完了
    靠 `backend/guardrails/output_filters.py` 的 `check_allergens()` 事后拦截。

    这条和核查 pass 的硬阻断是互补关系，不是二选一：这里降低触发频率(多数
    情况下模型一开始就会避开)，核查 pass 仍然保留作为兜底(万一这里没生效)。
    明确要求"调整食材/调味来避开，而不是因为用到这个成分就放弃推荐整道菜"——
    直接回应"能不能就是不放蚝油"这个问题：能，而且应该优先这样做，不是把
    整道菜的推荐一起扔掉。
    """
    allergens = [a.strip() for a in (allergens or []) if a and a.strip()]
    if not allergens:
        return ""
    lines = [f"用户对以下过敏原/成分过敏，生成建议时必须避开：{'、'.join(allergens)}。"]
    hidden = hidden_sources_for_allergens(allergens)
    for category, terms in hidden.items():
        lines.append(
            f"注意「{category}」常见的隐藏来源包括：{'、'.join(terms)}，"
            "这些调料/加工品即使名字里不带过敏原类别本身的字样，也同样要避开。"
        )
    lines.append(
        "优先做法是调整食材或调味来避开这些成分（比如省略某个调料或换成不含"
        "该成分的替代品），而不是因为一道菜通常会用到这些成分就整道放弃推荐——"
        "只有确实找不到安全替代时，才不推荐这道菜。"
    )
    return "\n".join(lines)

# ARCHITECTURE §5.4/§10：单会话 ≤15 次工具调用——这是 SubAgent 专属的严格限额，
# 不是 router.py DEFAULT_MAX_TOOL_CALLS(=50，中枢自己的安全兜底)的复用。
SUBAGENT_MAX_TOOL_CALLS = 15

# ARCHITECTURE §5.4："循环防护(连续 3 轮无新增信息)"，SubAgent 循环终止条件的
# 一部分，中枢 agent 没有这条约束。
SUBAGENT_STALL_ROUND_LIMIT = 3


@dataclass
class SubAgentResult:
    domain: str
    final_text: str
    tool_call_count: int
    iterations: int
    terminated_reason: str
    messages: list[dict]
    tools_called: list[str]


async def run_subagent(
    *,
    domain: str,
    role: CallerRole,
    system_prompt: str,
    task_input: str,
    server: DietExpertMcpServer,
    complete: CompleteFn | None = None,
    user_id: str = "default_user",
    locale: str = "zh",
) -> SubAgentResult:
    """开一个按角色隔离的 MCP session，跑一次 SubAgent 循环。

    `task_input` 是用户原始提问文本，不是结构化字段——PRD §12.3"任务上下文
    携带用户原始提问文本"，"加班到很晚了"这类一次性情境信息靠 SubAgent 直接
    读原文理解(D25)，不在这里拆成结构化参数。

    `user_id` 绑定这个 session 里 `query_diet_log`(只读)会查到哪个用户的
    `diet_log`——SubAgent 自己不知道也不该知道"当前是哪个用户"，这个值由
    中枢从 `profile`/`request` 里取出后传进来(见 `backend/mcp_server/server.py`
    `McpSession`/`_USER_SCOPED_TOOLS` 的注入机制)。
    """
    locale = normalize_locale(locale)
    set_current_locale(locale)
    session = server.open_session(role, user_id=user_id)
    available_tools = sorted(t.name for t in session.list_tools())
    logger.info(
        "SubAgent[%s] loop start · role=%s · visible_tools=%s",
        domain, role.value, available_tools,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_input},
    ]
    t0 = time.perf_counter()
    timeout_s = subagent_timeout_s()
    with observation(
        f"subagent.{domain}",
        as_type="agent",
        input={"task": redact_text(task_input), "visible_tools": available_tools},
        metadata={"role": role.value, "domain": domain, "timeout_s": timeout_s},
    ):
        try:
            # ENGINEERING §1.1 / §2 pit 2: wait_for cancels the loop (and the
            # in-flight complete()) when this side hits 45s, so a slow sibling
            # cannot keep spending tokens after we have already given up.
            result = await asyncio.wait_for(
                run_agent_loop(
                    messages,
                    session,
                    complete=complete,
                    max_tool_calls=SUBAGENT_MAX_TOOL_CALLS,
                    stall_round_limit=SUBAGENT_STALL_ROUND_LIMIT,
                    # ARCHITECTURE §4.5：状态提示只加在 SubAgent 这一处开放式循环，中枢自己
                    # 调用 run_agent_loop 时不传这个参数——两者共用同一份循环实现，区别就在
                    # 这一行。build_status_message 本身是纯代码，不经过任何 LLM 调用
                    # （backend/memory/status_prompt.py，D27"状态栏投毒"防护）。
                    before_next_call=lambda msgs, count, cap: build_status_message(msgs, count, cap),
                ),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning(
                "SubAgent[%s] timed out after %.1fs · role=%s",
                domain, timeout_s, role.value,
            )
            update_current(
                output={"terminated_reason": "timeout"},
                metadata={"latency_ms": round(latency_ms, 1), "timeout_s": timeout_s},
                level="WARNING",
            )
            stage_log(
                logger, domain, latency_ms=round(latency_ms, 1), terminated_reason="timeout",
            )
            raise SubAgentTimeoutError(
                f"SubAgent {domain} exceeded {timeout_s:.1f}s"
            ) from exc
        except asyncio.CancelledError:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            update_current(
                output={"terminated_reason": "cancelled"},
                metadata={"latency_ms": round(latency_ms, 1)},
            )
            raise

        # 只统计"真的执行成功"的工具(`ok=True`)——被协议层拒绝的越权尝试也会在
        # messages 里留一条 role="tool" 消息(见 router.py `_execute_tool_call`)，
        # 但那不代表这一侧真的拿到了对方领域的内容，不该算进"调用过的工具"。
        tools_called = sorted(
            {m["name"] for m in result.messages if m.get("role") == "tool" and m.get("ok")}
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        update_current(
            output={
                "terminated_reason": result.terminated_reason,
                "tool_call_count": result.tool_call_count,
                "iterations": result.iterations,
                "tools_called": tools_called,
                "final_text": redact_text(result.final_text),
            },
            metadata={"latency_ms": round(latency_ms, 1)},
        )
        # 打日志确认这一侧的检索/工具调用没有越出这一侧的领域(BUILD_PLAN 阶段4任务7
        # 完成判据)——真正的隔离由协议层白名单强制(§2.3,越权调用在 McpSession 里
        # 直接被拒绝,见 backend/mcp_server/server.py),这条日志是给这条判据一个可
        # 核查的旁证,不是再实现一遍隔离逻辑。
        logger.info(
            "SubAgent[%s] loop done · tool_calls_used=%d/%d · iterations=%d · "
            "terminated_reason=%s · tools_called=%s",
            domain, result.tool_call_count, SUBAGENT_MAX_TOOL_CALLS, result.iterations,
            result.terminated_reason, tools_called,
        )
        stage_log(
            logger,
            domain,
            latency_ms=round(latency_ms, 1),
            tool_call_count=result.tool_call_count,
            iterations=result.iterations,
            terminated_reason=result.terminated_reason,
            tools_called=tools_called,
        )

        return SubAgentResult(
            domain=domain,
            final_text=result.final_text,
            tool_call_count=result.tool_call_count,
            iterations=result.iterations,
            terminated_reason=result.terminated_reason,
            messages=result.messages,
            tools_called=tools_called,
        )
