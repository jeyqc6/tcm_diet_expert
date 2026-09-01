"""
测试目标：`/api/onboarding/start`、`/api/onboarding/answer`、`GET/PATCH /api/profile`
(2026-08-26 补，见 api/main.py 模块文档)——不经过 LLM，`get_mcp_server` 换成
注入了假 `write_memory`/画像读取 handler 的 server，`get_user_profile_fetcher`
换成固定值，不连真实数据库。
对应实现：api/main.py、backend/onboarding/flow.py、backend/mcp_server/tools/write_memory.py
"""
from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from api.main import (
    app,
    get_complete_fn,
    get_idle_session_folder,
    get_mcp_server,
    get_onboarding_store,
    get_pending_critical_store,
    get_session_history_loader,
    get_turn_recorder,
    get_user_profile_ensurer,
    get_user_profile_fetcher,
)
from backend.agents.user_context import UserProfileContext
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.server import DietExpertMcpServer
from backend.mcp_server.tools.write_memory import WriteResult
from backend.memory.pending_critical_facts import InMemoryPendingCriticalFactStore
from backend.onboarding.session_store import InMemoryOnboardingSessionStore


@pytest.fixture(autouse=True)
def _clear_overrides():
    # `/api/onboarding/start` stamps a stub user_profile row; tests must not
    # hit the developer's real database. One store instance per test so
    # multi-turn `/api/chat` onboarding keeps state.
    store = InMemoryOnboardingSessionStore()
    app.dependency_overrides[get_user_profile_ensurer] = lambda: (lambda **kw: True)
    app.dependency_overrides[get_onboarding_store] = lambda: store
    app.dependency_overrides[get_pending_critical_store] = lambda: InMemoryPendingCriticalFactStore()
    # D27 补充(2026-08-28)：同 test_api_chat_sse.py 的同款注释——这些测试
    # 复用硬编码 session_id("s1")，不覆盖会真的写真实 Postgres。
    app.dependency_overrides[get_session_history_loader] = lambda: (lambda session_id: "")
    app.dependency_overrides[get_turn_recorder] = lambda: (lambda session_id, turn, **kw: None)
    app.dependency_overrides[get_idle_session_folder] = lambda: (lambda session_id: None)
    yield
    app.dependency_overrides.clear()


def _server_with_write_memory(handler) -> DietExpertMcpServer:
    base = default_tool_definitions()
    tools: dict[str, ToolDefinition] = dict(base)
    tools["write_memory"] = ToolDefinition(
        name="write_memory",
        description=base["write_memory"].description,
        input_schema=base["write_memory"].input_schema,
        handler=handler,
    )
    return DietExpertMcpServer(tools=tools)


def test_onboarding_start_returns_allergens_step():
    client = TestClient(app)
    resp = client.post("/api/onboarding/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["step_id"] == "allergens"
    assert "过敏" in body["prompt"]


def test_onboarding_answer_walks_self_reported_constitution_path_and_writes():
    captured: list[dict] = []

    def fake_write_memory(**kwargs):
        captured.append(kwargs)
        return WriteResult(ok=True, table="user_profile", user_id="default_user", fields_written=tuple(sorted(kwargs["payload"])))

    server = _server_with_write_memory(fake_write_memory)
    app.dependency_overrides[get_mcp_server] = lambda: server
    client = TestClient(app)

    step = client.post("/api/onboarding/start").json()
    assert step["step_id"] == "allergens"

    step = client.post(
        "/api/onboarding/answer",
        json={"step_id": step["step_id"], "answer": "花生", "state": step["state"]},
    ).json()
    assert step["step_id"] == "preferences"

    step = client.post(
        "/api/onboarding/answer",
        json={"step_id": step["step_id"], "answer": "没有", "state": step["state"]},
    ).json()
    assert step["step_id"] == "city"

    step = client.post(
        "/api/onboarding/answer",
        json={"step_id": step["step_id"], "answer": "上海", "state": step["state"]},
    ).json()
    assert step["step_id"] == "timezone_confirm"

    step = client.post(
        "/api/onboarding/answer",
        json={"step_id": step["step_id"], "answer": "对", "state": step["state"]},
    ).json()
    assert step["step_id"] == "constitution_known"

    final = client.post(
        "/api/onboarding/answer",
        json={"step_id": step["step_id"], "answer": "我是气虚质", "state": step["state"]},
    ).json()

    assert final["step_id"] == "done"
    assert final["written"] is True
    assert final["profile_updates"]["constitution"] == "qi_xu"
    assert final["profile_updates"]["allergens"] == ["花生"]
    assert final["profile_updates"]["city"] == "上海"
    assert final["profile_updates"]["timezone"] == "Asia/Shanghai"

    # write_memory 真的被调用了一次，payload 就是最终收集到的字段。
    assert len(captured) == 1
    assert captured[0]["category"] == "critical"
    assert captured[0]["payload"]["constitution"] == "qi_xu"


def test_onboarding_start_stamps_profile_so_later_visits_do_not_reprompt():
    stamped: list[int] = []

    def fake_ensure(**kw) -> bool:
        stamped.append(1)
        return True

    app.dependency_overrides[get_user_profile_ensurer] = lambda: fake_ensure
    client = TestClient(app)
    resp = client.post("/api/onboarding/start")
    assert resp.status_code == 200
    assert stamped == [1]


def test_get_profile_returns_exists_false_without_profile():
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: None)
    client = TestClient(app)
    resp = client.get("/api/profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is False
    assert body["onboarding_recommended"] is True


def test_get_profile_recommends_onboarding_for_create_user_stub():
    profile = UserProfileContext(user_id="u_new", constitution=None, allergens=())
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: profile)
    client = TestClient(app)
    resp = client.get("/api/profile")
    body = resp.json()
    assert body["exists"] is True
    assert body["onboarding_recommended"] is True


def test_get_profile_does_not_recommend_onboarding_after_skip():
    profile = UserProfileContext(
        user_id="default_user", constitution=None, allergens=(), onboarding_done=True
    )
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: profile)
    client = TestClient(app)
    resp = client.get("/api/profile")
    body = resp.json()
    assert body["exists"] is True
    assert body["onboarding_recommended"] is False


def test_get_profile_does_not_recommend_onboarding_when_complete():
    profile = UserProfileContext(
        user_id="default_user", constitution="qi_xu", allergens=("花生",), onboarding_done=True
    )
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: profile)
    client = TestClient(app)
    resp = client.get("/api/profile")
    body = resp.json()
    assert body["onboarding_recommended"] is False
    assert body["constitution"] == "qi_xu"


def test_patch_profile_rejects_unconfirmed():
    client = TestClient(app)
    resp = client.patch("/api/profile", json={"field": "city", "value": "北京", "confirmed": False})
    assert resp.status_code == 400


def test_patch_profile_rejects_unknown_field():
    client = TestClient(app)
    resp = client.patch("/api/profile", json={"field": "id", "value": 1, "confirmed": True})
    assert resp.status_code == 400


def test_patch_profile_writes_confirmed_field():
    captured: list[dict] = []

    def fake_write_memory(**kwargs):
        captured.append(kwargs)
        return WriteResult(ok=True, table="user_profile", user_id="default_user", fields_written=("city",))

    server = _server_with_write_memory(fake_write_memory)
    app.dependency_overrides[get_mcp_server] = lambda: server
    client = TestClient(app)

    resp = client.patch("/api/profile", json={"field": "city", "value": "北京", "confirmed": True})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert captured[0]["payload"] == {"city": "北京"}


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


def _token_text(events: list[tuple[str, str]]) -> str:
    parts = []
    for event, data in events:
        if event != "token":
            continue
        parts.append(json.loads(data).get("text", ""))
    return "".join(parts)


class _ForbidLLM:
    async def __call__(self, *args, **kwargs):
        raise AssertionError("onboarding over /api/chat must not call the LLM")


def test_chat_first_turn_without_profile_is_onboarding_not_agent_pipeline():
    stamped: list[int] = []
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: None)
    app.dependency_overrides[get_user_profile_ensurer] = lambda: (lambda **kw: stamped.append(1) or True)
    app.dependency_overrides[get_complete_fn] = lambda: _ForbidLLM()
    app.dependency_overrides[get_mcp_server] = lambda: _server_with_write_memory(
        lambda **kw: WriteResult(ok=True, table="user_profile", user_id="default_user", fields_written=())
    )
    client = TestClient(app)

    resp = client.post("/api/chat", json={"session_id": "s1", "message": "今天该吃什么"})
    assert resp.status_code == 200
    text = _token_text(_parse_sse(resp.text))
    assert "过敏" in text
    assert stamped == [1]


def test_chat_onboarding_continues_then_abort_writes_and_does_not_reprompt():
    captured: list[dict] = []
    profile_holder: dict = {"p": None}

    def fake_write_memory(**kwargs):
        captured.append(kwargs)
        return WriteResult(ok=True, table="user_profile", user_id="default_user", fields_written=("allergens",))

    def fake_ensure(**kw) -> bool:
        profile_holder["p"] = UserProfileContext(user_id="default_user")
        return True

    store = InMemoryOnboardingSessionStore()
    app.dependency_overrides[get_onboarding_store] = lambda: store
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: profile_holder["p"])
    app.dependency_overrides[get_user_profile_ensurer] = lambda: fake_ensure
    app.dependency_overrides[get_complete_fn] = lambda: _ForbidLLM()
    app.dependency_overrides[get_mcp_server] = lambda: _server_with_write_memory(fake_write_memory)
    client = TestClient(app)

    first = client.post("/api/chat", json={"session_id": "s1", "message": "今天该吃什么"})
    assert "过敏" in _token_text(_parse_sse(first.text))

    second = client.post("/api/chat", json={"session_id": "s1", "message": "全部跳过"})
    done_text = _token_text(_parse_sse(second.text))
    assert "引导完成" in done_text or "之后可以直接继续提问" in done_text
    assert captured and captured[0]["category"] == "critical"
    assert captured[0]["payload"]["constitution_source"] == "unconfirmed"
    assert captured[0]["payload"]["onboarding_done"] is True
    assert store.get("default_user") is None
    # Skip writes onboarding_done so a later fetch does not re-enter intro.
    assert profile_holder["p"] is not None
