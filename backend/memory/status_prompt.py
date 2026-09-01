#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SubAgent 循环状态提示：代码维护，不经过 LLM，防"状态栏投毒"。

设计依据：docs/ARCHITECTURE.md §4.5
决策依据：docs/DECISIONS.md D27
⚠️ 必须单测覆盖（确定性优先）——D27 原话:"如果这条状态消息本身算错了...模型
会无条件信任它,错误会直接传导成错误决策"，所以这个模块的正确性比大多数业务
逻辑更需要穷举边界，不是"看起来对就行"。

D20 的五处 agent 行为里，只有 TCM/Nutrition SubAgent 自主决定"要不要再调用一次
工具"是开放式循环——中枢 agent 自己复用同一个 backend/agents/router.py 的
`run_agent_loop()`，但它的循环不属于这五处之一（路由/调和/核查都是固定 workflow
步骤）。状态提示因此**不能塞进 `run_agent_loop` 内部**（那会连中枢的循环也捎带
上），只能是 `run_agent_loop` 一个可选的 hook 参数，由
backend/agents/_subagent_common.py 的 `run_subagent()` 传入，中枢自己调用
`run_agent_loop` 时不传，行为不变。

两条硬约束（对应 ARCHITECTURE §4.5 原文）：
  1. 必须由代码计算，不能让模型自己总结"到目前为止发生了什么"。
  2. "候选信息要点"是从已发生的 tool_calls 事件里用确定性规则抽取的短列表
     （查过哪些食材、查过哪几天天气），不携带任何检索原文——只看 assistant
     消息里 `tool_calls` 的调用参数（"问了什么"），不看 role="tool" 的执行
     结果内容（"查到了什么"，那里面可能是完整的检索原文）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 每个工具挑哪些参数当"要点"——不同工具的入参形状不一样，不能一刀切全塞进去；
# 没在这里列出的工具用全部参数兜底（宁可啰嗦，也不要因为漏配置一个新工具
# 就完全没有信息）。
_KEY_ARG_FIELDS: dict[str, tuple[str, ...]] = {
    "retrieve_tcm": ("query",),
    "retrieve_nutrition": ("query",),
    "query_weather": ("city", "date"),
    "query_diet_log": ("time_range", "aggregation"),
    "query_recipes_by_ingredients": ("ingredients",),
}


@dataclass(frozen=True)
class ToolCallSummary:
    """一次工具调用的要点——只有工具名 + 挑出来的关键参数，不携带任何执行结果。"""

    tool_name: str
    key_args: dict[str, Any] = field(default_factory=dict)


def summarize_tool_call(name: str, arguments: dict[str, Any] | None) -> ToolCallSummary:
    arguments = arguments or {}
    fields = _KEY_ARG_FIELDS.get(name)
    if fields is None:
        key_args = dict(arguments)
    else:
        key_args = {k: arguments[k] for k in fields if k in arguments}
    return ToolCallSummary(tool_name=name, key_args=key_args)


def extract_tool_call_summaries(messages: list[dict]) -> list[ToolCallSummary]:
    """从 run_agent_loop 产出的 messages 历史里，按发生顺序抽出每一次"模型发起的
    工具调用"的要点。只扫 assistant 消息的 `tool_calls` 字段（调用请求本身），
    刻意不碰 role="tool" 的消息（那里面是执行结果，可能是完整检索原文）。
    """
    summaries: list[ToolCallSummary] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            summaries.append(summarize_tool_call(call.get("name", ""), call.get("arguments")))
    return summaries


def _format_summary(s: ToolCallSummary) -> str:
    if not s.key_args:
        return s.tool_name
    args_str = ", ".join(f"{k}={v}" for k, v in s.key_args.items())
    return f"{s.tool_name}({args_str})"


def format_status_prompt(
    tool_call_count: int,
    max_tool_calls: int,
    summaries: list[ToolCallSummary],
) -> str:
    """拼出 ARCHITECTURE §4.5 定义的那一行状态提示文本：

        [状态] 已用工具调用:{n}/15(资源限额,§5.4)· 已检索到的候选信息要点:{短列表}
    """
    points = "、".join(_format_summary(s) for s in summaries) if summaries else "（还没有）"
    return (
        f"[状态] 已用工具调用:{tool_call_count}/{max_tool_calls}(资源限额,§5.4)"
        f"· 已检索到的候选信息要点:{points}"
    )


def build_status_message(
    messages: list[dict],
    tool_call_count: int,
    max_tool_calls: int,
) -> dict:
    """构造一条可以直接 append 进 SubAgent messages 列表、追加到上下文末尾的
    状态提示消息。

    用 role="user" 而不是 "system"：backend/llm/providers/anthropic_provider.py
    的 `_translate_messages` 会把**所有** role="system" 的消息抽出来合并进顶层
    `system` 参数，不管它在对话历史里处于什么位置——如果这里用 "system"，这条
    本该"出现在最新位置"的状态提示会被搬到最前面跟最初的 system prompt 混在
    一起，D27 强调的"必须在最新位置"这条硬约束就失效了。role="user" 才会被两家
    provider 都按原位置保留。
    """
    summaries = extract_tool_call_summaries(messages)
    text = format_status_prompt(tool_call_count, max_tool_calls, summaries)
    return {"role": "user", "content": text}
