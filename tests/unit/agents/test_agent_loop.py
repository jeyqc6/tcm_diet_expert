"""
测试目标：backend/agents/agent_loop.py 的 `run_agent_loop`(ARCHITECTURE §3.2)
——不打真实模型/网络：注入假 `complete`(同 tests/unit/llm/test_adapter.py 的模式)，
用真实 DietExpertMcpServer + 一个 stub 工具验证"单工具能调通"，以及循环的终止
条件确实是 tool_use 的有无，不是硬编码轮数。
对应实现：backend/agents/agent_loop.py
"""
from __future__ import annotations

import asyncio

import pytest

from backend.agents.agent_loop import (
    DEFAULT_MAX_TOOL_CALLS,
    AgentLoopResourceLimitError,
    run_agent_loop,
)
from backend.llm.adapter import LLMResult, ModelTier
from backend.llm.providers.base import ToolCall
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.roles import CallerRole
from backend.mcp_server.server import DietExpertMcpServer


def _run(coro):
    return asyncio.run(coro)


def _stub_registry(handler=None) -> dict[str, ToolDefinition]:
    base = default_tool_definitions()
    handler = handler or (lambda **kwargs: {"ok": True, "kwargs": kwargs})
    stubs: dict[str, ToolDefinition] = {}
    for name, tool in base.items():
        stubs[name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=handler,
        )
    return stubs


def _result(text="", tool_calls=None) -> LLMResult:
    return LLMResult(text=text, model="m", tier=ModelTier.DEV, provider="fake", tool_calls=tool_calls)


class _ScriptedComplete:
    """按顺序回放 LLMResult，并记录每次调用时收到的 messages/tools，方便断言。"""

    def __init__(self, script: list[LLMResult]):
        self._script = list(script)
        self.calls: list[dict] = []

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        return self._script.pop(0)


def test_no_tool_use_terminates_on_first_turn():
    """完整推荐/事实查询等分支里模型不需要工具时，loop 应该一轮就结束。"""
    server = DietExpertMcpServer(tools=_stub_registry())
    session = server.open_session(CallerRole.ROUTER)
    complete = _ScriptedComplete([_result(text="不需要查任何工具，直接回答。")])

    result = _run(
        run_agent_loop(
            [{"role": "user", "content": "阳虚质是什么意思"}], session, complete=complete
        )
    )

    assert result.final_text == "不需要查任何工具，直接回答。"
    assert result.tool_call_count == 0
    assert result.iterations == 1
    assert len(complete.calls) == 1


def test_single_tool_call_round_trips_and_then_terminates():
    """单工具能调通：第一轮返回 tool_use，第二轮不再返回 → loop 结束。"""
    server = DietExpertMcpServer(
        tools=_stub_registry(handler=lambda **kwargs: {"temp_c": 21, "echo": kwargs})
    )
    session = server.open_session(CallerRole.ROUTER)
    complete = _ScriptedComplete(
        [
            _result(
                tool_calls=[
                    ToolCall(id="call_1", name="query_weather", arguments={"city": "北京"})
                ]
            ),
            _result(text="北京今天 21 度。"),
        ]
    )

    result = _run(
        run_agent_loop([{"role": "user", "content": "北京天气怎么样"}], session, complete=complete)
    )

    assert result.final_text == "北京今天 21 度。"
    assert result.tool_call_count == 1
    assert result.iterations == 2
    assert len(complete.calls) == 2

    # 第二轮喂给模型的 messages 里必须包含 assistant 的 tool_calls 回合和工具结果
    second_call_messages = complete.calls[1]["messages"]
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-2]["tool_calls"][0]["name"] == "query_weather"
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "call_1"
    assert '"temp_c": 21' in second_call_messages[-1]["content"]

    # 第一轮把 session.list_tools() 的 schema 喂给了模型(router 能看到全部 6 个工具)
    first_call_tools = {t["name"] for t in complete.calls[0]["tools"]}
    assert "query_weather" in first_call_tools
    assert len(first_call_tools) == 6


def test_multiple_tool_calls_in_one_turn_all_execute_before_next_call():
    server = DietExpertMcpServer(tools=_stub_registry(handler=lambda **kwargs: {"echo": kwargs}))
    session = server.open_session(CallerRole.ROUTER)
    complete = _ScriptedComplete(
        [
            _result(
                tool_calls=[
                    ToolCall(id="call_1", name="query_weather", arguments={"city": "北京"}),
                    ToolCall(id="call_2", name="query_weather", arguments={"city": "上海"}),
                ]
            ),
            _result(text="两个城市都查到了。"),
        ]
    )

    result = _run(
        run_agent_loop([{"role": "user", "content": "查两个城市"}], session, complete=complete)
    )

    assert result.tool_call_count == 2
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"call_1", "call_2"}


def test_multiple_tool_calls_in_one_turn_execute_concurrently_not_serially():
    """回归测试：一轮里的多个工具调用之前是同步 for 循环逐个执行——真实
    trace 量过，3 次 retrieve_nutrition 顺序执行耗时 10.4s，最慢那次单独
    只要 6.0s，多出来的纯粹是排队。这里用会阻塞的 stub handler(`time.sleep`,
    `_execute_tool_call` 经 `asyncio.to_thread` 丢进线程池执行)验证三次调用
    的总墙钟时间接近最慢的那一次，而不是三次耗时相加——如果哪天这段又被
    改回顺序执行，这个测试会因为耗时断言超时而失败，不会安静地退化。

    这条测试本身测不出 2026-08-31 那次真实撞过的"跨 event loop 复用 provider
    客户端导致挂死"的坑(那个 bug 只有真实调 LLM SDK 才会触发，stub handler
    不涉及)——那个坑靠真实请求验证过，见
    `backend/llm/adapter.py` `_get_provider()` 和 `_retrieval_common.py`
    `_get_embedder()` 的注释。这条测试只保证"确实是并发执行，没有偷偷退化
    回顺序执行"。"""
    import time

    def slow_handler(**kwargs):
        time.sleep(0.2)
        return {"ok": True}

    server = DietExpertMcpServer(tools=_stub_registry(handler=slow_handler))
    session = server.open_session(CallerRole.ROUTER)
    complete = _ScriptedComplete(
        [
            _result(
                tool_calls=[
                    ToolCall(id=f"call_{i}", name="query_weather", arguments={"city": "北京"})
                    for i in range(3)
                ]
            ),
            _result(text="三个都查完了。"),
        ]
    )

    t0 = time.perf_counter()
    result = _run(
        run_agent_loop([{"role": "user", "content": "查三次"}], session, complete=complete)
    )
    elapsed = time.perf_counter() - t0

    assert result.tool_call_count == 3
    # 串行的话至少 0.6s(3 × 0.2s)；并发的话接近 0.2s。0.45s 留了充足余量，
    # 既能容忍线程调度抖动，又足以在"又变回串行"时可靠地失败。
    assert elapsed < 0.45, f"tool calls took {elapsed:.2f}s — looks serial, not concurrent"


def test_unauthorized_tool_call_is_caught_and_fed_back_not_raised():
    """SubAgent 越权调用 write_memory：协议层拒绝，错误喂回模型而不是让 loop 崩溃。"""
    server = DietExpertMcpServer(tools=_stub_registry())
    session = server.open_session(CallerRole.TCM_SUBAGENT)
    complete = _ScriptedComplete(
        [
            _result(
                tool_calls=[
                    ToolCall(id="call_1", name="write_memory", arguments={"category": "daily_log", "payload": {}})
                ]
            ),
            _result(text="好的，我没有权限写记忆。"),
        ]
    )

    result = _run(
        run_agent_loop([{"role": "user", "content": "帮我记一下"}], session, complete=complete)
    )

    assert result.final_text == "好的，我没有权限写记忆。"
    tool_message = result.messages[-1]
    assert tool_message["role"] == "tool"
    assert "error" in tool_message["content"]
    assert len(server.unauthorized_attempts) == 1


def test_before_next_call_hook_appends_message_each_round():
    """ARCHITECTURE §4.5 的挂载点：hook 在每次 complete() 之前被调用一次，
    拿到 (messages, tool_call_count, max_tool_calls)，返回的消息被追加到
    喂给下一次 complete() 的 messages 末尾——不传这个参数时（其余测试都没传）
    行为完全不变，这里单独验证"传了会发生什么"。"""
    server = DietExpertMcpServer(tools=_stub_registry())
    session = server.open_session(CallerRole.ROUTER)
    complete = _ScriptedComplete(
        [
            _result(tool_calls=[ToolCall(id="c1", name="query_weather", arguments={"city": "北京"})]),
            _result(text="完成"),
        ]
    )
    seen_args = []

    def hook(messages, tool_call_count, max_tool_calls):
        seen_args.append((tool_call_count, max_tool_calls))
        return {"role": "user", "content": f"[hook] count={tool_call_count}"}

    _run(
        run_agent_loop(
            [{"role": "user", "content": "北京天气"}],
            session,
            complete=complete,
            before_next_call=hook,
        )
    )

    # 两轮 complete() 各触发一次 hook：第一次 0 个工具调用，第二次已经用了 1 个
    assert seen_args == [(0, DEFAULT_MAX_TOOL_CALLS), (1, DEFAULT_MAX_TOOL_CALLS)]

    first_call_messages = complete.calls[0]["messages"]
    assert first_call_messages[-1] == {"role": "user", "content": "[hook] count=0"}

    second_call_messages = complete.calls[1]["messages"]
    assert second_call_messages[-1] == {"role": "user", "content": "[hook] count=1"}


def test_before_next_call_none_return_appends_nothing():
    server = DietExpertMcpServer(tools=_stub_registry())
    session = server.open_session(CallerRole.ROUTER)
    complete = _ScriptedComplete([_result(text="不需要工具")])
    original_len = 1  # 只有初始那条 user 消息

    _run(
        run_agent_loop(
            [{"role": "user", "content": "阳虚质是什么"}],
            session,
            complete=complete,
            before_next_call=lambda *a: None,
        )
    )

    assert len(complete.calls[0]["messages"]) == original_len


def test_dataclass_tool_results_serialize_as_structured_json_not_repr_string():
    """真实工具（retrieve_tcm/retrieve_nutrition）返回 list[RetrievedChunk]，
    这些是 dataclass 实例，不是天然可 json.dumps 的类型。修复前用的是
    default=str，会把每个 chunk 变成一个 Python repr 字符串塞进 JSON 数组里
    （比如 "RetrievedChunk(source_id='...', ...)"），下游没法结构化取出
    source_id。这里验证喂给下一轮模型的 tool 消息里，chunk 的字段是真正的
    JSON 对象字段，不是字符串。"""
    from backend.mcp_server.tools._retrieval_common import RetrievedChunk

    chunk = RetrievedChunk(
        source_id="tcm_000123", domain="tcm", source_file="x.md",
        source_type="md_table_row", text="阳虚质忌生冷", metadata={}, score=0.5,
    )
    server = DietExpertMcpServer(tools=_stub_registry(handler=lambda **kwargs: [chunk]))
    session = server.open_session(CallerRole.ROUTER)
    complete = _ScriptedComplete(
        [
            _result(tool_calls=[ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "阳虚质"})]),
            _result(text="done"),
        ]
    )

    result = _run(
        run_agent_loop([{"role": "user", "content": "阳虚质吃什么"}], session, complete=complete)
    )

    tool_message = next(m for m in result.messages if m.get("role") == "tool")
    import json

    parsed = json.loads(tool_message["content"])
    assert parsed == [
        {
            "source_id": "tcm_000123", "domain": "tcm", "source_file": "x.md",
            "source_type": "md_table_row", "text": "阳虚质忌生冷", "metadata": {}, "score": 0.5,
        }
    ]
    # 结构化字段可以直接取，不用从字符串里正则抠
    assert parsed[0]["source_id"] == "tcm_000123"


def test_resource_cap_raises_not_silently_terminates():
    """安全兜底和"正常终止"是两条不同的路径——模型一直不停调用工具时必须报错，
    不能悄悄在触顶时假装成功结束(那会掩盖失控循环)。"""
    server = DietExpertMcpServer(tools=_stub_registry())
    session = server.open_session(CallerRole.ROUTER)
    # 脚本永远返回一个新的 tool_use，模拟模型陷入死循环
    infinite_tool_use = [
        _result(tool_calls=[ToolCall(id=f"call_{i}", name="query_weather", arguments={"city": "北京"})])
        for i in range(10)
    ]
    complete = _ScriptedComplete(infinite_tool_use)

    with pytest.raises(AgentLoopResourceLimitError):
        _run(
            run_agent_loop(
                [{"role": "user", "content": "天气"}],
                session,
                complete=complete,
                max_tool_calls=3,
            )
        )


def test_retrieve_tcm_over_budget_drops_oldest_chunks_before_next_llm_call():
    huge = "字" * 8000  # ~4444 tokens each; six chunks exceed the 12k budget
    payload = [{"source_id": f"t{i}", "text": huge} for i in range(6)]

    def handler(**_kwargs):
        return payload

    server = DietExpertMcpServer(tools=_stub_registry(handler=handler))
    session = server.open_session(CallerRole.TCM_SUBAGENT)
    complete = _ScriptedComplete(
        [
            _result(
                tool_calls=[
                    ToolCall(id="call_1", name="retrieve_tcm", arguments={"query": "阳虚"})
                ]
            ),
            _result(text="阳虚质忌生冷。"),
        ]
    )
    _run(run_agent_loop([{"role": "user", "content": "阳虚质吃什么"}], session, complete=complete))
    import json

    tool_msg = next(m for m in complete.calls[1]["messages"] if m.get("role") == "tool")
    kept = json.loads(tool_msg["content"])
    assert len(kept) < 6
    assert kept[0]["source_id"] != "t0" or len(kept) == 1
    assert kept[-1]["source_id"] == "t5"
