"""
测试目标：`McpSession.call_tool()` 对 query_diet_log/write_memory 这两个
"user-scoped"工具注入 `user_id`(2026-08-30 真实多用户支持新增)——SubAgent
自己不知道也不该知道"当前是哪个用户"，这个值必须由 session 打开时绑定的
`user_id` 决定，不能是 LLM 自己填的参数(不在公开 JSON Schema 里)。
对应实现：backend/mcp_server/server.py `_USER_SCOPED_TOOLS`/`McpSession.call_tool`
覆盖要求：常规
"""
from __future__ import annotations

from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.roles import CallerRole
from backend.mcp_server.server import DietExpertMcpServer


def _capturing_registry() -> tuple[dict[str, ToolDefinition], dict[str, dict]]:
    """Handlers record the kwargs they were actually called with."""
    calls: dict[str, dict] = {}
    base = default_tool_definitions()
    stubs: dict[str, ToolDefinition] = {}
    for name, tool in base.items():
        def handler(_n=name, **kw):
            calls[_n] = kw
            return _n
        stubs[name] = ToolDefinition(
            name=tool.name, description=tool.description, input_schema=tool.input_schema,
            handler=handler,
        )
    return stubs, calls


def test_query_diet_log_receives_session_user_id():
    stubs, calls = _capturing_registry()
    server = DietExpertMcpServer(tools=stubs)
    session = server.open_session(CallerRole.TCM_SUBAGENT, user_id="alice")
    session.call_tool("query_diet_log", {"time_range": "今天"})
    assert calls["query_diet_log"]["user_id"] == "alice"


def test_write_memory_receives_session_user_id():
    stubs, calls = _capturing_registry()
    server = DietExpertMcpServer(tools=stubs)
    session = server.open_session(CallerRole.ROUTER, user_id="bob")
    session.call_tool("write_memory", {"category": "daily_log", "payload": {}})
    assert calls["write_memory"]["user_id"] == "bob"


def test_different_sessions_inject_different_user_ids():
    """两个用户各自的 session 不会互相污染——回归这次多用户支持要解决的
    核心问题：改之前所有 SubAgent 工具调用都悄悄落在 'default_user' 上。"""
    stubs, calls = _capturing_registry()
    server = DietExpertMcpServer(tools=stubs)
    server.open_session(CallerRole.ROUTER, user_id="alice").call_tool(
        "query_diet_log", {"time_range": "今天"}
    )
    alice_user_id = calls["query_diet_log"]["user_id"]
    server.open_session(CallerRole.ROUTER, user_id="bob").call_tool(
        "query_diet_log", {"time_range": "今天"}
    )
    bob_user_id = calls["query_diet_log"]["user_id"]
    assert alice_user_id == "alice"
    assert bob_user_id == "bob"


def test_non_user_scoped_tool_does_not_receive_user_id():
    """retrieve_tcm 不碰用户专属数据，不该被硬塞一个它的 handler 不认识
    的参数。"""
    stubs, calls = _capturing_registry()
    server = DietExpertMcpServer(tools=stubs)
    session = server.open_session(CallerRole.TCM_SUBAGENT, user_id="alice")
    session.call_tool("retrieve_tcm", {"query": "红枣"})
    assert "user_id" not in calls["retrieve_tcm"]


def test_open_session_defaults_to_default_user_when_not_specified():
    stubs, calls = _capturing_registry()
    server = DietExpertMcpServer(tools=stubs)
    session = server.open_session(CallerRole.ROUTER)
    session.call_tool("query_diet_log", {"time_range": "今天"})
    assert calls["query_diet_log"]["user_id"] == "default_user"
