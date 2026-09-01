#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Loop（`run_agent_loop`）——工具调用机制的骨架，中枢 agent 和两个
SubAgent（backend/agents/_subagent_common.py）共用同一份实现。

设计依据：docs/ARCHITECTURE.md §3(工具调用机制，两层：tool_use vs MCP 协议)
roadmap：阶段 4.2 任务 5(先只做 loop 骨架)

实现 ARCHITECTURE §3.2"一次调用的完整链路"步骤 1-7：LLM 返回 tool_use →
业务代码拦截 → 经 MCP session 执行 → 结果喂回下一轮 → 模型决定是否继续。
**循环的终止条件是"这一轮有没有 tool_use"，不是硬编码的固定轮数**——
`max_tool_calls` 只是防止失控循环的安全兜底，不是判断"该不该继续"的依据。

2026-08-28：从 `backend/agents/router.py` 拆出——那个文件原本同时装着"六条
分支路由判断"和"Agent Loop"两件互不相关的事(见该文件当时的模块文档"本文件
目前有两个独立部分,不要混为一谈")，六条分支判断部分搬去了
`backend/agents/routing.py`。纯粹搬文件，不改变任何函数签名/行为。
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from backend.exceptions import ResourceLimitError
from backend.memory.compression import (
    CompressibleChunk,
    fifo_drop_oldest_chunks,
    should_compress_retrieval,
)
from backend.llm import adapter as llm_adapter
from backend.llm.adapter import CompleteFn, LLMResult
from backend.mcp_server.exceptions import ToolNotDeclaredError
from backend.mcp_server.server import McpSession

logger = logging.getLogger("diet_expert.agents.agent_loop")

# 安全兜底，不是终止条件本身——真正的终止条件是"这一轮 LLM 返回里有没有
# tool_use"(ARCHITECTURE §3.2 步骤 7)。这个值只用来防止模型陷入死循环反复
# 调用工具、一直不收敛。SubAgent 有自己的专属限额(ARCHITECTURE §5.4：≤15 次)，
# 那是 backend/agents/_subagent_common.py 要传的更严格的值，不是本文件替
# SubAgent 预先决定的。
DEFAULT_MAX_TOOL_CALLS = 50
_RETRIEVAL_TOOLS = frozenset({"retrieve_tcm", "retrieve_nutrition"})


def _maybe_fifo_compress_retrieval(messages: list[dict]) -> None:
    """After each tool_result: if retrieval text exceeds the 12k budget, drop
    the oldest retrieve_* chunks in place. Mid-loop cannot drop 'uncited'
    chunks — final_text does not exist yet (D27 / ARCHITECTURE §4.4.1)."""
    indexed: list[tuple[int, int, CompressibleChunk]] = []
    parsed_lists: dict[int, list[Any]] = {}
    for msg_i, msg in enumerate(messages):
        if msg.get("role") != "tool" or msg.get("name") not in _RETRIEVAL_TOOLS:
            continue
        if not msg.get("ok", True):
            continue
        try:
            payload = json.loads(msg.get("content") or "[]")
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        parsed_lists[msg_i] = payload
        for chunk_i, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "")
            text = str(item.get("text") or "")
            if source_id or text:
                indexed.append((msg_i, chunk_i, CompressibleChunk(source_id=source_id, text=text)))
    chunks = [c for _, _, c in indexed]
    if not should_compress_retrieval(chunks):
        return
    kept = fifo_drop_oldest_chunks(chunks)
    kept_ids = {(c.source_id, c.text) for c in kept}
    drop_keys = {
        (msg_i, chunk_i)
        for msg_i, chunk_i, c in indexed
        if (c.source_id, c.text) not in kept_ids
    }
    for msg_i, payload in parsed_lists.items():
        filtered = [
            item
            for chunk_i, item in enumerate(payload)
            if (msg_i, chunk_i) not in drop_keys
        ]
        messages[msg_i]["content"] = json.dumps(filtered, ensure_ascii=False)


class AgentLoopResourceLimitError(ResourceLimitError):
    """触达 `max_tool_calls` 安全兜底——不是正常终止路径，正常终止靠 tool_use 缺席。"""


@dataclass
class AgentLoopResult:
    final_text: str
    messages: list[dict]
    tool_call_count: int
    iterations: int
    # "no_tool_use"(正常终止,§3.2 步骤 7)/ "stall_guard"(§5.4 循环防护，
    # 连续 `stall_round_limit` 轮工具调用没有产生任何新信息，判定为已收敛)。
    terminated_reason: str = "no_tool_use"


def _tool_schemas(session: McpSession) -> list[dict]:
    """MCP ToolDefinition -> provider 层认的归一化 tool schema
    (backend/llm/providers/base.py 模块文档)。"""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in session.list_tools()
    ]


def _assistant_tool_call_message(result: LLMResult) -> dict:
    return {
        "role": "assistant",
        "content": result.text or None,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in (result.tool_calls or [])
        ],
    }


def _execute_tool_call(session: McpSession, call: Any) -> tuple[str, bool]:
    """执行一次工具调用，返回 (喂回下一轮的 tool_result 文本, 是否真的执行成功)。

    越权调用(ToolNotDeclaredError)和工具业务逻辑本身抛出的异常都不让 loop
    崩溃——按 ARCHITECTURE §3.2 步骤 4"协议层拒绝,记录越权尝试"，把错误信息
    喂回给模型，由它决定下一步(比如换一个它有权限的工具，或者据此终止)。

    `ok` 这个布尔值单独返回、不靠调用方解析 JSON 内容猜——下游(比如
    backend/agents/_subagent_common.py 统计"这一侧真正调用过哪些工具"用于领域
    隔离的日志)需要区分"模型尝试调用但被协议层拒绝"和"工具真的执行了"，
    这两者都会在 messages 里留下一条 role="tool" 消息，内容上不该靠猜。
    """
    try:
        raw_result = session.call_tool(call.name, call.arguments or {})
    except ToolNotDeclaredError as exc:
        logger.warning("Agent Loop: unauthorized tool call blocked: %s", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False), False
    except Exception as exc:  # noqa: BLE001 — 工具实现自己的失败，不属于 loop 的职责范围
        logger.warning("Agent Loop: tool %r raised %s", call.name, exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False), False
    logger.debug("Agent Loop: tool %r called, args=%r", call.name, call.arguments)
    return json.dumps(raw_result, ensure_ascii=False, default=_json_default), True


def _json_default(obj: Any) -> Any:
    """工具处理函数经常直接返回 dataclass 实例(比如 retrieve_tcm 返回
    `list[RetrievedChunk]`)。`json.dumps(default=str)` 会把它们变成 Python
    repr 字符串（比如 "RetrievedChunk(source_id='tcm_000123', ...)"），塞进
    JSON 数组里是几个字符串，不是结构化对象——下游(核查pass/API层)想从
    tool_result 里精确取出 source_id 字段时，只能拿到一坨要用正则去抠的文本，
    不是能直接 `.get("source_id")` 的 JSON 对象。这里改成 dataclass 感知：
    是 dataclass 就转成 dict（能被 json 正常递归序列化），只有其他真正不知道
    怎么处理的类型才退回 str()。
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def _round_signature(call: Any, content: str) -> tuple[str, str, str]:
    """一次工具调用的"内容指纹"——用来判断循环防护(§5.4)里"这一轮有没有产生
    新信息"。同一个工具、同样的参数、同样的结果，就算模型又调用了一次，也不算
    新信息。

    这是精确字符串匹配，不是语义判重——模型换一种问法查询本质相同的信息
    （比如"气虚质吃什么"和"气虚质饮食建议"）会被当成两次不同的调用，不会被
    这条防护拦下。这是可接受的近似：语义判重需要 embedding 或额外一次 LLM
    调用，成本和不确定性都不值得为这条防护引入。

    ⚠️ 隐性假设：工具对相同输入返回确定性内容。如果某个工具的返回值里混入了
    时间戳、请求 id 这类每次调用都会变的字段（比如 `query_weather` 真正实现后
    很可能在结果里带 `queried_at`），"同一个查询"连续两次的指纹也会不一样，
    这条防护就会失效——而这恰恰是它最该拦住的场景（模型卡在重复调用同一个
    工具）。当前各工具的错误消息都是静态字符串（已核实），不是活 bug，但新增
    工具实现时要留意这一点，不要在返回值里悄悄带上易变字段。
    """
    return (call.name, json.dumps(call.arguments or {}, sort_keys=True, ensure_ascii=False), content)


async def run_agent_loop(
    messages: list[dict],
    session: McpSession,
    *,
    complete: CompleteFn | None = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    stall_round_limit: int | None = None,
    before_next_call: Callable[[list[dict], int, int], dict | None] | None = None,
    **complete_kwargs: Any,
) -> AgentLoopResult:
    """ARCHITECTURE §3.2 步骤 1-7 的完整实现。

    `complete` 默认用 backend/llm/adapter.py 的 `complete()`；单测可以注入假实现，
    不用打真实网络(同 tests/unit/llm/test_adapter.py 的模式)。`session` 是已经按
    调用方角色开好的 MCP session(§2.3 权限分层)，决定这一轮模型能看到哪些工具。

    `stall_round_limit`：ARCHITECTURE §5.4"循环防护(连续 3 轮无新增信息)"——
    这条只描述为"SubAgent 循环终止条件的一部分"，中枢 agent 自己的 loop 没有
    这条约束，所以默认 `None`(关闭)；SubAgent(backend/agents/_subagent_common.py)
    显式传 3。触发时不是报错，是当作已收敛的正常终止(§3.2 步骤 7 的延伸：
    "不再产出新信息"和"不再请求工具"在效果上是同一件事)，但不会再问一次模型，
    而是直接结束——工具结果都已经在 messages 里了，没有必要再多打一次 LLM 调用
    去"总结"，那笔钱不该花在一个已经确定收敛的地方。

    `before_next_call`：ARCHITECTURE §4.5 SubAgent 循环状态提示的挂载点——**默认
    None，中枢 agent 自己调用这个函数时不传，行为完全不变**。D20 的五处 agent
    行为里只有 TCM/Nutrition SubAgent 的这个循环是开放式的，状态提示不该塞进
    `run_agent_loop` 内部(那会连中枢的循环也捎带上)，只能是一个可选 hook，由
    backend/agents/_subagent_common.py 传入(backend/memory/status_prompt.py 的
    `build_status_message`)。签名是 `(messages, tool_call_count, max_tool_calls)
    -> dict | None`，返回非 None 时被 append 到 messages 末尾、再喂给下一次
    `complete()` 调用——即"追加到上下文末尾"，不是替换或插到中间。
    """
    complete = complete or llm_adapter.complete
    messages = list(messages)
    tools = _tool_schemas(session)
    tool_call_count = 0
    iterations = 0
    seen_signatures: set[tuple[str, str, str]] = set()
    stall_rounds = 0

    while True:
        iterations += 1
        if before_next_call is not None:
            status_message = before_next_call(messages, tool_call_count, max_tool_calls)
            if status_message is not None:
                messages.append(status_message)
        result = await complete(messages, tools=tools, **complete_kwargs)

        if not result.tool_calls:
            # 终止条件：这一轮没有 tool_use，不是数到第几轮。
            return AgentLoopResult(
                final_text=result.text,
                messages=messages,
                tool_call_count=tool_call_count,
                iterations=iterations,
                terminated_reason="no_tool_use",
            )

        messages.append(_assistant_tool_call_message(result))
        round_has_new_info = False

        # 一轮里模型请求的多个工具调用互相没有依赖(都是只读检索/查询)，并发
        # 执行——真实 trace 量过：3 次 retrieve_nutrition 顺序执行耗时 10.4s，
        # 三次里最慢的一次只要 6.0s，多出来的 4.4s 纯粹是排队。
        # `_execute_tool_call` 内部是同步阻塞调用(psycopg2/HTTP)，直接
        # `asyncio.gather` 包同步函数不会并发，用 `asyncio.to_thread` 丢进
        # 线程池才是真的并发；`to_thread` 会给每个线程拷贝一份当前
        # contextvars(trace_id/span stack)，互不干扰。
        #
        # ⚠️ 2026-08-31 教训(第一次实现时真实撞过、已修好)：这里第一版上线后
        # 真实请求会挂死撞 45s SubAgent 超时——根因是
        # `backend/mcp_server/tools/_retrieval_common.py` 的 MQE 通过
        # `_run_coroutine_sync()` 每次调用都现造一个跑完即销毁的 event loop
        # 去调用异步的 `complete()`，而 `backend/llm/adapter.py` 的
        # `_PROVIDERS` 当时把 provider 的异步 SDK 客户端(如 `AsyncAnthropic`，
        # 内部持有 httpx 连接池)按进程生命周期缓存——多个并发 `to_thread`
        # worker 各自起一个新 loop 时，会把同一个缓存客户端在互不相同的
        # event loop 之间反复复用，这是已知会静默挂死的 asyncio 反模式。已经
        # 把 `_PROVIDERS` 改成按 `(provider 名字, 当前 loop)` 缓存(见
        # adapter.py 该函数的注释)，`_get_embedder()` 也补了锁(同一类"惰性
        # 单例没加锁"问题)——这两处修好之后才重新启用这里的并发执行，并且
        # 用真实请求(尤其是当初撞坑的 Anthropic provider 路径)反复验证过
        # 不再卡死，不是只看单测绿了就当作修好。
        #
        # `max_tool_calls` 的检查从"逐个调用前检查"改成"整轮批量检查"：原来的
        # 语义是"哪次调用把计数顶破了就在那次调用之前报错，之前已经真的执行过
        # 的调用不会被撤销"；并发之后一批调用同时发出去，没有"这次调用之前"这个
        # 时间点了。这里选择在整批任何一个调用真正执行之前，先算这一整批加起来
        # 会不会超限——超限就整批都不执行直接报错。这是同一个安全兜底(防止失控
        # 循环)的等价实现，只是检查粒度从"每次调用"变成"每轮"，不影响正常路径
        # (未超限时行为完全一致)。
        prospective_count = tool_call_count + len(result.tool_calls)
        if prospective_count > max_tool_calls:
            raise AgentLoopResourceLimitError(
                f"Agent Loop exceeded safety cap of {max_tool_calls} tool calls "
                f"in one run (this is a runaway-loop backstop, not a normal "
                f"termination path — see module docstring)"
            )
        tool_call_count = prospective_count

        executed = await asyncio.gather(
            *(asyncio.to_thread(_execute_tool_call, session, call) for call in result.tool_calls)
        )
        for call, (content, ok) in zip(result.tool_calls, executed):
            signature = _round_signature(call, content)
            if signature not in seen_signatures:
                round_has_new_info = True
            seen_signatures.add(signature)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": content,
                    "ok": ok,
                }
            )
        _maybe_fifo_compress_retrieval(messages)

        if stall_round_limit is None:
            continue
        if round_has_new_info:
            stall_rounds = 0
            continue
        stall_rounds += 1
        if stall_rounds >= stall_round_limit:
            logger.warning(
                "Agent Loop: stall guard triggered — %d consecutive rounds with no "
                "new tool information, stopping without another LLM call",
                stall_rounds,
            )
            return AgentLoopResult(
                final_text=result.text,
                messages=messages,
                tool_call_count=tool_call_count,
                iterations=iterations,
                terminated_reason="stall_guard",
            )
