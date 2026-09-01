"""
测试目标：越权调用在协议层被拒绝（D7修订核心论点）
对应实现：backend/mcp_server/server.py
覆盖要求：常规
"""
from __future__ import annotations

import pytest

from backend.mcp_server.exceptions import ToolNotDeclaredError
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.roles import (
    ALL_TOOLS,
    TOOL_QUERY_DIET_LOG,
    TOOL_QUERY_RECIPES,
    TOOL_QUERY_WEATHER,
    TOOL_RETRIEVE_NUTRITION,
    TOOL_RETRIEVE_TCM,
    TOOL_WRITE_MEMORY,
    CallerRole,
)
from backend.mcp_server.server import DietExpertMcpServer


def _stub_registry() -> dict[str, ToolDefinition]:
    """Handlers return tool name — whitelist tests never hit real DB/network."""
    base = default_tool_definitions()
    stubs: dict[str, ToolDefinition] = {}
    for name, tool in base.items():
        stubs[name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=lambda _n=name, **_: _n,
        )
    return stubs


@pytest.fixture
def server() -> DietExpertMcpServer:
    return DietExpertMcpServer(tools=_stub_registry())


class TestToolVisibility:
    def test_router_sees_all_six_tools(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.ROUTER)
        names = {t.name for t in session.list_tools()}
        assert names == set(ALL_TOOLS)

    def test_tcm_subagent_tool_subset(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.TCM_SUBAGENT)
        names = {t.name for t in session.list_tools()}
        assert names == {TOOL_RETRIEVE_TCM, TOOL_QUERY_WEATHER, TOOL_QUERY_DIET_LOG}
        assert TOOL_WRITE_MEMORY not in names
        assert TOOL_RETRIEVE_NUTRITION not in names
        assert TOOL_QUERY_RECIPES not in names

    def test_nutrition_subagent_tool_subset(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.NUTRITION_SUBAGENT)
        names = {t.name for t in session.list_tools()}
        assert names == {TOOL_RETRIEVE_NUTRITION, TOOL_QUERY_DIET_LOG, TOOL_QUERY_RECIPES}
        assert TOOL_RETRIEVE_TCM not in names
        assert TOOL_QUERY_WEATHER not in names
        assert TOOL_WRITE_MEMORY not in names

    def test_reconciliation_and_verification_have_no_tools(
        self, server: DietExpertMcpServer
    ) -> None:
        for role in (CallerRole.RECONCILIATION, CallerRole.VERIFICATION):
            session = server.open_session(role)
            assert session.list_tools() == []


class TestProtocolLayerRejection:
    def test_tcm_cannot_call_write_memory(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.TCM_SUBAGENT)
        with pytest.raises(ToolNotDeclaredError) as exc:
            session.call_tool(TOOL_WRITE_MEMORY, {"category": "daily_log", "payload": {}})
        assert exc.value.role is CallerRole.TCM_SUBAGENT
        assert exc.value.tool_name == TOOL_WRITE_MEMORY

    def test_nutrition_cannot_call_retrieve_tcm(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.NUTRITION_SUBAGENT)
        with pytest.raises(ToolNotDeclaredError):
            session.call_tool(TOOL_RETRIEVE_TCM, {"query": "test"})

    def test_reconciliation_cannot_call_any_tool(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.RECONCILIATION)
        with pytest.raises(ToolNotDeclaredError):
            session.call_tool(TOOL_QUERY_DIET_LOG, {"time_range": "昨天"})

    def test_unauthorized_attempt_is_recorded(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.TCM_SUBAGENT)
        with pytest.raises(ToolNotDeclaredError):
            session.call_tool(TOOL_RETRIEVE_NUTRITION, {"query": "x"})
        assert len(server.unauthorized_attempts) == 1
        assert server.unauthorized_attempts[0].role is CallerRole.TCM_SUBAGENT
        assert server.unauthorized_attempts[0].tool_name == TOOL_RETRIEVE_NUTRITION


class TestAllowedDispatch:
    def test_allowed_call_reaches_handler(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.TCM_SUBAGENT)
        assert session.call_tool(TOOL_RETRIEVE_TCM, {"query": "q"}) == TOOL_RETRIEVE_TCM

    def test_router_can_call_write_memory(self, server: DietExpertMcpServer) -> None:
        session = server.open_session(CallerRole.ROUTER)
        assert (
            session.call_tool(
                TOOL_WRITE_MEMORY,
                {"category": "daily_log", "payload": {"dish": "test"}},
            )
            == TOOL_WRITE_MEMORY
        )

    def test_different_sessions_are_isolated(self, server: DietExpertMcpServer) -> None:
        tcm = server.open_session(CallerRole.TCM_SUBAGENT)
        router = server.open_session(CallerRole.ROUTER)
        assert len(tcm.list_tools()) == 3
        assert len(router.list_tools()) == 6
