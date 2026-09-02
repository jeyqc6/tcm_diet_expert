from __future__ import annotations

from backend.mcp_server.safe_call import safe_call_tool


class _FakeSession:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict | None]] = []

    def call_tool(self, name: str, arguments: dict | None = None):
        self.calls.append((name, arguments))
        if self.fail:
            raise RuntimeError("tool down")
        return {"ok": True, "name": name}


def test_safe_call_tool_returns_result_on_success():
    session = _FakeSession()
    out = safe_call_tool(session, "query_diet_log", {"time_range": "today"})
    assert out.ok is True
    assert out.result == {"ok": True, "name": "query_diet_log"}
    assert session.calls == [("query_diet_log", {"time_range": "today"})]


def test_safe_call_tool_returns_structured_error_on_failure():
    session = _FakeSession(fail=True)
    out = safe_call_tool(session, "write_memory", {"category": "critical", "payload": {}})
    assert out.ok is False
    assert out.error_type == "RuntimeError"
    assert "tool down" in (out.detail or "")
