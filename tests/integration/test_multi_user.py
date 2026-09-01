"""
测试目标：2026-08-30 真实多用户支持——`create_user()`/`list_users()` 对真实
Postgres 的读写，以及最核心的正确性属性：不同用户各自的 `conversation_sessions`/
`messages` 互不可见。改之前所有写路径都硬编码 `user_id="default_user"`，这条
测试就是回归"改完之后真的隔离了，不是看起来隔离"。
对应实现：backend/agents/user_context.py `list_users()`/`create_user()`，
backend/memory/session_store.py `load_all_messages()`
覆盖要求：集成测试，需要真实本地 Postgres(DIET_EXPERT_PG_DSN)；连不上时整个
文件跳过，同 tests/integration/test_session_store.py 的既有约定。
"""
from __future__ import annotations

import uuid

import pytest

from backend.env import get_pg_dsn
from backend.agents.user_context import create_user, list_users
from backend.memory.compression import TurnRecord
from backend.memory.session_store import load_all_messages, record_turn

psycopg2 = pytest.importorskip("psycopg2")

_DSN = get_pg_dsn()
if not _DSN:
    pytest.skip("DIET_EXPERT_PG_DSN not configured, skipping real-Postgres multi-user tests", allow_module_level=True)
else:
    try:
        _conn = psycopg2.connect(_DSN, connect_timeout=3)
        _conn.close()
    except Exception:
        pytest.skip("cannot connect to Postgres, skipping real-Postgres multi-user tests", allow_module_level=True)


def _new_session_id() -> str:
    return f"test-session-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def _cleanup():
    created_sessions: list[str] = []
    created_users: list[str] = []
    yield {"sessions": created_sessions, "users": created_users}
    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        if created_sessions:
            cur.execute("DELETE FROM messages WHERE session_id = ANY(%s)", (created_sessions,))
            cur.execute("DELETE FROM conversation_sessions WHERE session_id = ANY(%s)", (created_sessions,))
        if created_users:
            cur.execute("DELETE FROM user_profile WHERE user_id = ANY(%s)", (created_users,))
        conn.commit()
    finally:
        conn.close()


def test_create_user_returns_a_distinct_user_id(_cleanup):
    created = create_user("测试用户A")
    assert created is not None
    _cleanup["users"].append(created["user_id"])
    assert created["user_id"] != "default_user"
    assert created["name"] == "测试用户A"


def test_create_user_appears_in_list_users(_cleanup):
    created = create_user("测试用户B")
    _cleanup["users"].append(created["user_id"])
    users = list_users()
    matching = [u for u in users if u["user_id"] == created["user_id"]]
    assert len(matching) == 1
    assert matching[0]["name"] == "测试用户B"


def test_messages_are_isolated_between_users(_cleanup):
    """核心正确性属性：alice 说的话，bob 的 load_all_messages() 看不到。"""
    alice = create_user("alice-test")
    bob = create_user("bob-test")
    _cleanup["users"].extend([alice["user_id"], bob["user_id"]])

    alice_session = _new_session_id()
    bob_session = _new_session_id()
    _cleanup["sessions"].extend([alice_session, bob_session])

    record_turn(
        alice_session,
        TurnRecord(turn_id="0", branch="fact_query", raw_text="用户: alice的话\n助手: alice的回答", conclusion="alice的回答"),
        user_id=alice["user_id"],
    )
    record_turn(
        bob_session,
        TurnRecord(turn_id="0", branch="fact_query", raw_text="用户: bob的话\n助手: bob的回答", conclusion="bob的回答"),
        user_id=bob["user_id"],
    )

    alice_messages = load_all_messages(user_id=alice["user_id"])
    bob_messages = load_all_messages(user_id=bob["user_id"])

    alice_texts = {m["assistant_text"] for m in alice_messages}
    bob_texts = {m["assistant_text"] for m in bob_messages}

    assert "alice的回答" in alice_texts
    assert "bob的回答" not in alice_texts
    assert "bob的回答" in bob_texts
    assert "alice的回答" not in bob_texts
