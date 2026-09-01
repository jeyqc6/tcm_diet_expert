"""
测试目标：`GET /api/sessions/{session_id}/messages`(§10.1，之前一直没实现)——
前端刷新页面后拉取历史消息重建聊天气泡。不经过 LLM/SubAgent，
`get_session_messages_fetcher` 换成固定假函数，不连真实数据库(真实 DB 接线
的行为由 `backend/memory/session_store.py` `load_session_messages()` 自己的
单测覆盖，见 tests/integration/test_session_store.py)。
对应实现：api/main.py、backend/memory/session_store.py
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app, get_session_messages_fetcher


def test_get_session_messages_returns_fetched_rows():
    fake_rows = [
        {
            "turn_index": 0,
            "compression_tier": 0,
            "archived": False,
            "branch": "fact_query",
            "user_text": "红枣是什么性味",
            "assistant_text": "红枣性温",
            "cited_source_ids": ["tcm_001"],
            "rejected_suggestions": [],
            "triggered_guardrails": [],
            "created_at": 1735500000.0,
        }
    ]
    app.dependency_overrides[get_session_messages_fetcher] = lambda: (lambda session_id: fake_rows)
    try:
        client = TestClient(app)
        resp = client.get("/api/sessions/s1/messages")
        assert resp.status_code == 200
        assert resp.json()["messages"] == fake_rows
    finally:
        app.dependency_overrides.clear()


def test_get_session_messages_passes_session_id_through_to_fetcher():
    captured: list[str] = []
    app.dependency_overrides[get_session_messages_fetcher] = lambda: (
        lambda session_id: captured.append(session_id) or []
    )
    try:
        client = TestClient(app)
        client.get("/api/sessions/some-particular-session/messages")
        assert captured == ["some-particular-session"]
    finally:
        app.dependency_overrides.clear()


def test_get_session_messages_empty_list_for_unknown_session_is_not_an_error():
    """"没有历史"是正常状态,不是错误——不存在的 session_id 不应该 404。"""
    app.dependency_overrides[get_session_messages_fetcher] = lambda: (lambda session_id: [])
    try:
        client = TestClient(app)
        resp = client.get("/api/sessions/does-not-exist/messages")
        assert resp.status_code == 200
        assert resp.json() == {"messages": []}
    finally:
        app.dependency_overrides.clear()
