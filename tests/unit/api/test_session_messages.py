"""`GET /api/sessions/{session_id}/messages`——不打真实 Postgres，
`get_session_messages_fetcher` 用 `dependency_overrides` 换成假函数，同
tests/unit/api/test_exception_handlers.py 的既有模式。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app, get_session_messages_fetcher


def test_returns_fetched_messages_for_the_given_session_id():
    captured_session_id: list[str] = []

    def fake_fetcher(session_id: str):
        captured_session_id.append(session_id)
        return [
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
                "created_at": 1234567890.0,
            }
        ]

    app.dependency_overrides[get_session_messages_fetcher] = lambda: fake_fetcher
    try:
        client = TestClient(app)
        resp = client.get("/api/sessions/some-session/messages")
        assert resp.status_code == 200
        body = resp.json()
        assert captured_session_id == ["some-session"]
        assert len(body["messages"]) == 1
        assert body["messages"][0]["user_text"] == "红枣是什么性味"
        assert body["messages"][0]["assistant_text"] == "红枣性温"
    finally:
        app.dependency_overrides.pop(get_session_messages_fetcher, None)


def test_unknown_session_returns_empty_list_not_404():
    """"没有历史"是正常状态，不是错误——查不到这个 session 时
    `load_session_messages()` 静默降级为空列表，端点原样返回，不该拼出一个
    404。"""
    app.dependency_overrides[get_session_messages_fetcher] = lambda: (lambda session_id: [])
    try:
        client = TestClient(app)
        resp = client.get("/api/sessions/does-not-exist/messages")
        assert resp.status_code == 200
        assert resp.json() == {"messages": []}
    finally:
        app.dependency_overrides.pop(get_session_messages_fetcher, None)
