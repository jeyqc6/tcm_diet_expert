"""
测试目标：一句话包含多个独立意图时的分发(D32,ARCHITECTURE §5.1.1)——
新增的 `task`/`task_done` SSE 事件、子任务顺序执行、每个子任务复用既有分支
处理逻辑(不重新实现)。
对应实现：api/main.py(`_dispatch_branch`/`_stream_multi_task`)、
backend/agents/router.py(`classify_multi_task`/`segment_intents`)
覆盖要求：集成测试，注入假 complete()/server，不打真实网络/LLM/DB。
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from api.main import (
    app,
    get_complete_fn,
    get_idle_session_folder,
    get_mcp_server,
    get_pending_critical_store,
    get_session_history_loader,
    get_turn_recorder,
    get_user_profile_fetcher,
)
from backend.agents.user_context import UserProfileContext
from backend.llm.adapter import LLMResult, ModelTier
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.server import DietExpertMcpServer
from backend.mcp_server.tools.write_memory import WriteResult
from backend.memory.pending_critical_facts import InMemoryPendingCriticalFactStore


def _result(text="") -> LLMResult:
    return LLMResult(text=text, model="m", tier=ModelTier.DEV, provider="fake", tool_calls=None)


class _ScriptedComplete:
    def __init__(self, script: list[LLMResult]):
        self._script = list(script)
        self.call_count = 0

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.call_count += 1
        if not self._script:
            raise AssertionError(f"complete() 被调用第 {self.call_count} 次，但脚本已经用完")
        return self._script.pop(0)


def _server_with_handlers(**handlers) -> DietExpertMcpServer:
    base = default_tool_definitions()
    tools: dict[str, ToolDefinition] = {}
    for name, tool in base.items():
        handler = handlers.get(name, lambda **kw: {"stub": True, "kwargs": kw})
        tools[name] = ToolDefinition(
            name=tool.name, description=tool.description, input_schema=tool.input_schema, handler=handler
        )
    return DietExpertMcpServer(tools=tools)


def _parse_sse(body: str) -> list[tuple[str, str]]:
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        m = re.search(r"event:\s*(\S+)", block)
        d = re.search(r"data:\s*(.*)", block)
        if m and d:
            events.append((m.group(1), d.group(1)))
    return events


@pytest.fixture(autouse=True)
def _clear_overrides():
    # D27 补充(2026-08-28)：这些测试反复复用硬编码的 session_id("s1")，不
    # 覆盖 backend/memory/session_store.py 的三个注入点会真的写真实
    # Postgres、且不同测试之间通过同一个 session_id 互相污染，见
    # test_api_chat_sse.py `_clear_overrides` 的同款注释。
    # Past-intro stub: create_user rows would otherwise start onboarding.
    stub = UserProfileContext(user_id="default_user", onboarding_done=True)
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: stub)
    app.dependency_overrides[get_session_history_loader] = lambda: (lambda session_id: "")
    app.dependency_overrides[get_turn_recorder] = lambda: (lambda session_id, turn, **kw: None)
    app.dependency_overrides[get_idle_session_folder] = lambda: (lambda session_id: None)
    app.dependency_overrides[get_pending_critical_store] = lambda: InMemoryPendingCriticalFactStore()
    yield
    app.dependency_overrides.clear()


def test_multi_task_dispatches_log_write_then_single_domain():
    """"帮我记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么"——log_write(全局表
    命中，不打LLM)+ single_domain(tcm，需要 SubAgent 循环 + 核查)。"""
    written: list[dict] = []

    def fake_write_memory(**kwargs):
        written.append(kwargs)
        return WriteResult(ok=True, table="diet_log", user_id="default_user", fields_written=(), row_id=1)

    server = _server_with_handlers(
        write_memory=fake_write_memory,
        retrieve_tcm=lambda **kw: [
            {"source_id": "t1", "domain": "tcm", "source_file": "a", "source_type": "t", "text": "阳虚忌生冷", "metadata": {}, "score": 0.8}
        ],
    )

    class _TcmComplete(_ScriptedComplete):
        async def __call__(self, messages, *, tools=None, **kwargs):
            self.call_count += 1
            system_text = messages[0].get("content") or "" if messages else ""
            has_tool_result = any(m.get("role") == "tool" for m in messages)
            if "中医饮食 SubAgent" in system_text:
                if has_tool_result:
                    return _result("阳虚质忌生冷 [source: t1]")
                from backend.llm.providers.base import ToolCall
                return LLMResult(
                    text="", model="m", tier=ModelTier.DEV, provider="fake",
                    tool_calls=[ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "阳虚"})],
                )
            return _result('{"reject": [], "retry_reconciliation": false}')

    complete = _TcmComplete([])
    app.dependency_overrides[get_mcp_server] = lambda: server
    app.dependency_overrides[get_complete_fn] = lambda: complete
    client = TestClient(app)

    resp = client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "帮我记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么"},
    )
    events = _parse_sse(resp.text)
    event_types = [e for e, _ in events]

    # 两个 task 事件按切分顺序出现，各自带正确的分支标签。
    task_events = [d for e, d in events if e == "task"]
    assert len(task_events) == 2
    assert '"branch": "log_write"' in task_events[0]
    assert '"branch": "single_domain"' in task_events[1]

    # 子任务自己的 done 被换成了 task_done，整个响应只有最后一条真正的 done。
    assert event_types.count("done") == 1
    assert event_types[-1] == "done"
    assert event_types.count("task_done") == 2

    token_texts = "".join(d for e, d in events if e == "token")
    assert "麻婆豆腐" in token_texts
    assert "阳虚" in token_texts

    # log_write 子任务真的写库了，且 payload 只包含切分出来的那半句话。
    assert len(written) == 1
    assert "麻婆豆腐" in written[0]["payload"]["raw_input"]
    assert "阳虚" not in written[0]["payload"]["raw_input"]


def test_single_intent_message_has_no_task_events():
    """没有连接词的普通单意图消息——完全不触发多任务路径，行为和这条设计
    生效前一致(D32 的核心约束：不引入回归)。"""

    def fake_write_memory(**kwargs):
        return WriteResult(ok=True, table="diet_log", user_id="default_user", fields_written=(), row_id=1)

    server = _server_with_handlers(write_memory=fake_write_memory)
    complete = _ScriptedComplete([])  # 麻婆豆腐命中全局表，不需要 LLM
    app.dependency_overrides[get_mcp_server] = lambda: server
    app.dependency_overrides[get_complete_fn] = lambda: complete
    client = TestClient(app)

    resp = client.post("/api/chat", json={"session_id": "s1", "message": "帮我记录一下，中午吃了麻婆豆腐"})
    events = _parse_sse(resp.text)
    event_types = [e for e, _ in events]
    assert "task" not in event_types
    assert "task_done" not in event_types
    assert event_types.count("done") == 1


def test_same_branch_segments_do_not_trigger_multi_task():
    """两句话都是"记录"——不触发多任务路径，整条消息按单分支处理，
    dish_decomposition 的全局表扫描本来就能在一次调用里识别出两个菜品。"""
    written: list[dict] = []

    def fake_write_memory(**kwargs):
        written.append(kwargs)
        return WriteResult(ok=True, table="diet_log", user_id="default_user", fields_written=(), row_id=1)

    server = _server_with_handlers(write_memory=fake_write_memory)
    # 两个菜名都命中全局表后，剩余文本("帮我记录一下昨天吃了，还有帮我记录
    # 一下今天吃了")里含"还有"——dish_decomposition.py 自己的连接词短路规则
    # 判定"可能还有别的东西"，会多打一次 LLM 兜底；这里脚本给它一个"没有更多
    # 食物了"的响应。
    complete = _ScriptedComplete([_result('{"dishes":[]}')])
    app.dependency_overrides[get_mcp_server] = lambda: server
    app.dependency_overrides[get_complete_fn] = lambda: complete
    client = TestClient(app)

    resp = client.post(
        "/api/chat",
        json={"session_id": "s1", "message": "帮我记录一下昨天吃了红烧肉，还有帮我记录一下今天吃了宫保鸡丁"},
    )
    events = _parse_sse(resp.text)
    event_types = [e for e, _ in events]
    assert "task" not in event_types
    assert len(written) == 1  # 一次 write_memory 调用，两个菜都在同一条记录里
    dish_names = {d["dish"] for d in written[0]["payload"]["dishes"]}
    assert dish_names == {"红烧肉", "宫保鸡丁"}
