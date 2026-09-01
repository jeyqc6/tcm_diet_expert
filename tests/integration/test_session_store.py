"""
测试目标：backend/memory/session_store.py —— compression.py 压缩算法的真实
Postgres 接线：record_turn() 写入+触发归档、maybe_fold_idle_session() 折叠、
load_session_history() 组装。
对应实现：backend/memory/session_store.py、db/schema.sql messages/conversation_sessions
覆盖要求：集成测试，需要真实本地 Postgres(DIET_EXPERT_PG_DSN)；连不上时整个
文件跳过，不让"没配数据库"变成一堆假失败(同 db/load_conflict_rules.py 一类
脚本对真实环境的依赖方式，不是新发明的跳过约定)。
"""
from __future__ import annotations

import threading
import uuid

import pytest

from backend.env import get_pg_dsn
from backend.memory.compression import TurnRecord
from backend.memory.session_store import (
    TIER_ARCHIVED_ACTIVE,
    TIER_ARCHIVED_IDLE,
    TIER_RAW,
    load_session_history,
    load_session_messages,
    maybe_fold_idle_session,
    record_turn,
)

psycopg2 = pytest.importorskip("psycopg2")

_DSN = get_pg_dsn()
if not _DSN:
    pytest.skip("DIET_EXPERT_PG_DSN not configured, skipping real-Postgres session_store tests", allow_module_level=True)
else:
    try:
        _conn = psycopg2.connect(_DSN, connect_timeout=3)
        _conn.close()
    except Exception:
        pytest.skip("cannot connect to Postgres, skipping real-Postgres session_store tests", allow_module_level=True)


def _new_session_id() -> str:
    return f"test-session-{uuid.uuid4().hex[:12]}"


def _row_count(session_id: str) -> int:
    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM messages WHERE session_id = %s", (session_id,))
        return cur.fetchone()[0]
    finally:
        conn.close()


def _set_updated_at_in_past(session_id: str, seconds_ago: float) -> None:
    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversation_sessions SET updated_at = now() - (%s * interval '1 second') "
            "WHERE session_id = %s",
            (seconds_ago, session_id),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _cleanup():
    created: list[str] = []
    yield created
    if not created:
        return
    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ANY(%s)", (created,))
        cur.execute("DELETE FROM conversation_sessions WHERE session_id = ANY(%s)", (created,))
        conn.commit()
    finally:
        conn.close()


def test_record_turn_writes_a_row_with_all_structured_fields(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    turn = TurnRecord(
        turn_id="0",
        branch="fact_query",
        raw_text="用户: 红枣是什么性味\n助手: 红枣性温",
        conclusion="红枣性温",
        cited_source_ids=("tcm_001",),
        triggered_guardrails=(),
    )
    record_turn(session_id, turn)

    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT branch, conclusion, cited_source_ids, compression_tier FROM messages "
            "WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    assert row == ("fact_query", "红枣性温", ["tcm_001"], TIER_RAW)


def test_record_turn_increments_turn_index_per_session(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    for i in range(3):
        record_turn(
            session_id,
            TurnRecord(turn_id=str(i), branch="fact_query", raw_text=f"轮次{i}", conclusion=f"结论{i}"),
        )
    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT turn_index FROM messages WHERE session_id = %s ORDER BY turn_index", (session_id,))
        indices = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    assert indices == [0, 1, 2]


def test_record_turn_concurrent_writes_to_same_session_do_not_lose_a_turn(_cleanup):
    """回归测试：`SELECT MAX(turn_index)+1` 后面紧跟一条独立 INSERT，两条语句
    之间原本没有锁——同一个 session_id 并发调用可能读到同一个 MAX、撞
    UNIQUE(session_id, turn_index)，输的一方被 record_turn() 的静默降级吞掉，
    那一轮对话永久性地没有落库。加了 `pg_advisory_xact_lock` 之后，同一个
    session_id 的并发写入应该被序列化，全部成功、turn_index 各不相同。"""
    session_id = _new_session_id()
    _cleanup.append(session_id)
    n = 16
    barrier = threading.Barrier(n)  # 尽量让所有线程同时撞进临界区，扩大竞态窗口

    def _write(i: int) -> None:
        barrier.wait()
        record_turn(
            session_id,
            TurnRecord(turn_id=str(i), branch="fact_query", raw_text=f"轮次{i}", conclusion=f"结论{i}"),
        )

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _row_count(session_id) == n
    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT turn_index FROM messages WHERE session_id = %s ORDER BY turn_index", (session_id,))
        indices = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    assert indices == list(range(n))


def test_record_turn_upserts_conversation_sessions_row(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    record_turn(session_id, TurnRecord(turn_id="0", branch="fact_query", raw_text="x", conclusion="c"))
    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM conversation_sessions WHERE session_id = %s", (session_id,))
        row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None


def test_record_turn_triggers_archiving_when_over_threshold(_cleanup):
    """够大的原文应该在写入后自动触发归档——compression_tier 从 0 变成 1，
    content 变成 D27 结构化模板而不是原文。"""
    session_id = _new_session_id()
    _cleanup.append(session_id)
    big_text = "字" * 30_000  # 远超 SESSION_HISTORY_BUDGET_TOKENS(10k) 的触发阈值
    record_turn(
        session_id,
        TurnRecord(turn_id="0", branch="full_recommend", raw_text=big_text, conclusion="第一轮的结论"),
    )
    # keep_recent=1 的默认行为下，第二轮进来后第一轮应该已经不再是"最近一轮"，
    # 触发对第一轮的归档检查。
    record_turn(
        session_id,
        TurnRecord(turn_id="1", branch="full_recommend", raw_text="短", conclusion="第二轮的结论"),
    )

    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT turn_index, content, compression_tier FROM messages "
            "WHERE session_id = %s ORDER BY turn_index",
            (session_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    first_row = rows[0]
    assert first_row[2] == TIER_ARCHIVED_ACTIVE
    assert "第一轮的结论" in first_row[1]
    assert "full_recommend" in first_row[1]  # D27 模板里带 branch


def test_maybe_fold_idle_session_folds_tier2_into_tier3_after_threshold(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    big_text = "字" * 30_000
    record_turn(session_id, TurnRecord(turn_id="0", branch="full_recommend", raw_text=big_text, conclusion="c0"))
    record_turn(session_id, TurnRecord(turn_id="1", branch="full_recommend", raw_text="短", conclusion="c1"))

    # 确认归档确实发生了(前置条件)，再手动把 updated_at 拨到很久以前模拟空闲。
    _set_updated_at_in_past(session_id, seconds_ago=999999)
    maybe_fold_idle_session(session_id)

    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT compression_tier FROM messages WHERE session_id = %s AND turn_index = 0",
            (session_id,),
        )
        tier = cur.fetchone()[0]
    finally:
        conn.close()
    assert tier == TIER_ARCHIVED_IDLE


def test_maybe_fold_idle_session_noop_when_recently_active(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    big_text = "字" * 30_000
    record_turn(session_id, TurnRecord(turn_id="0", branch="full_recommend", raw_text=big_text, conclusion="c0"))
    record_turn(session_id, TurnRecord(turn_id="1", branch="full_recommend", raw_text="短", conclusion="c1"))

    maybe_fold_idle_session(session_id)  # 刚刚活跃过，不应该折叠

    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT compression_tier FROM messages WHERE session_id = %s AND turn_index = 0",
            (session_id,),
        )
        tier = cur.fetchone()[0]
    finally:
        conn.close()
    assert tier == TIER_ARCHIVED_ACTIVE


def test_maybe_fold_idle_session_unknown_session_is_a_noop():
    maybe_fold_idle_session("does-not-exist-" + uuid.uuid4().hex)  # 不应该抛异常


def test_load_session_history_empty_for_unknown_session():
    assert load_session_history("does-not-exist-" + uuid.uuid4().hex) == ""


def test_load_session_history_includes_raw_and_archived_content(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    record_turn(session_id, TurnRecord(turn_id="0", branch="fact_query", raw_text="用户:红枣性味\n助手:性温", conclusion="红枣性温"))

    history = load_session_history(session_id)
    assert "红枣性味" in history  # Tier1 原文仍在


def test_load_session_history_reflects_archived_summary_after_archiving(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    big_text = "字" * 30_000
    record_turn(session_id, TurnRecord(turn_id="0", branch="full_recommend", raw_text=big_text, conclusion="过敏原相关结论"))
    record_turn(session_id, TurnRecord(turn_id="1", branch="full_recommend", raw_text="短", conclusion="c1"))

    history = load_session_history(session_id)
    assert "过敏原相关结论" in history
    assert big_text not in history  # 归档之后原文本身已经被结构化摘要取代


def test_record_turn_silently_degrades_when_db_unreachable(monkeypatch):
    """连不上库时不抛异常——压缩/归档是锦上添花，不能拖垮已经成功的响应。"""
    monkeypatch.setenv("DIET_EXPERT_PG_DSN", "postgresql://nouser:nopass@localhost:1/nonexistent")
    record_turn("some-session", TurnRecord(turn_id="0", branch="fact_query", raw_text="x", conclusion="c"))


def test_load_session_history_silently_degrades_when_db_unreachable(monkeypatch):
    monkeypatch.setenv("DIET_EXPERT_PG_DSN", "postgresql://nouser:nopass@localhost:1/nonexistent")
    assert load_session_history("some-session") == ""


def test_load_session_messages_empty_for_unknown_session():
    assert load_session_messages("does-not-exist-" + uuid.uuid4().hex) == []


def test_load_session_messages_parses_user_text_out_of_tier1_content(_cleanup):
    """`record_turn()` 把这一轮写成 `"用户: {user}\\n助手: {conclusion}"` 一整
    条 content——`load_session_messages()` 应该把 `user_text` 从这个既有约定
    里解析出来，不是原样把 content 甩给前端自己解析。"""
    session_id = _new_session_id()
    _cleanup.append(session_id)
    record_turn(
        session_id,
        TurnRecord(
            turn_id="0",
            branch="fact_query",
            raw_text="用户: 红枣是什么性味\n助手: 红枣性温",
            conclusion="红枣性温",
            cited_source_ids=("tcm_001",),
        ),
    )

    messages = load_session_messages(session_id)
    assert len(messages) == 1
    row = messages[0]
    assert row["turn_index"] == 0
    assert row["archived"] is False
    assert row["user_text"] == "红枣是什么性味"
    assert row["assistant_text"] == "红枣性温"
    assert row["branch"] == "fact_query"
    assert row["cited_source_ids"] == ["tcm_001"]


def test_load_session_messages_archived_row_has_no_user_text(_cleanup):
    """归档之后原始用户提问已经不在库里了(D27 归档设计如此)——`user_text`
    应该是 `None`，不该伪造一条用户气泡；`assistant_text` 直接来自 `conclusion`
    列，两个 tier 下都完整保留，不需要重新解析已经被替换成结构化摘要的 content。"""
    session_id = _new_session_id()
    _cleanup.append(session_id)
    big_text = "字" * 30_000
    record_turn(session_id, TurnRecord(turn_id="0", branch="full_recommend", raw_text=big_text, conclusion="第一轮的结论"))
    record_turn(session_id, TurnRecord(turn_id="1", branch="full_recommend", raw_text="短", conclusion="第二轮的结论"))

    messages = load_session_messages(session_id)
    archived = next(m for m in messages if m["turn_index"] == 0)
    assert archived["archived"] is True
    assert archived["user_text"] is None
    assert archived["assistant_text"] == "第一轮的结论"


def test_load_session_messages_orders_by_turn_index(_cleanup):
    session_id = _new_session_id()
    _cleanup.append(session_id)
    for i in range(3):
        record_turn(
            session_id,
            TurnRecord(turn_id=str(i), branch="fact_query", raw_text=f"用户: q{i}\n助手: a{i}", conclusion=f"a{i}"),
        )
    messages = load_session_messages(session_id)
    assert [m["turn_index"] for m in messages] == [0, 1, 2]
    assert [m["user_text"] for m in messages] == ["q0", "q1", "q2"]


def test_load_session_messages_silently_degrades_when_db_unreachable(monkeypatch):
    monkeypatch.setenv("DIET_EXPERT_PG_DSN", "postgresql://nouser:nopass@localhost:1/nonexistent")
    assert load_session_messages("some-session") == []
