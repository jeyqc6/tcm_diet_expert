#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的"人在环追问"状态——D20 五处 agent 行为点第3条(记录解析追问,✅V1做)
的实现,并按用户要求扩展覆盖到 candidate_eval/single_domain/fact_query/
full_recommend 四个走 SubAgent 的分支(完整设计见 docs/ARCHITECTURE.md 新增
小节、docs/DECISIONS.md D20 更新说明)。

设计依据：docs/ARCHITECTURE.md §5.2(追问机制小节)
决策依据：docs/DECISIONS.md D20(五处agent行为点第3条)、PRD.md §11 Fallback
("输入模糊 → 追问一次，仍模糊则记为 unspecified" —— 单次重试上限)

形状完全仿照 backend/onboarding/session_store.py 的既有先例(抽象基类 +
内存实现 + 防御性拷贝)，不是发明新的存储模式。**唯一的区别是 key 用
`session_id` 不是 `user_id`**——追问是"这一次对话里正在问的一个具体问题"，
不像 onboarding 是跨会话的全局一次性流程；不同 session 可以各自有不同的
待补充问题，互不影响。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.agents.routing import RouteBranch
from backend.env import get_pg_dsn

logger = logging.getLogger("diet_expert.agents.clarification")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pending_clarifications (
    session_id      TEXT PRIMARY KEY,
    original_text   TEXT NOT NULL,
    branch          TEXT NOT NULL,
    domain_hint     TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class PendingClarification:
    original_text: str
    """触发追问的那段文本。多任务(D32)场景下是切分出的那一个子任务片段，
    不是整条原始消息——恢复时只需要重跑那一个子任务，不重复已经完成的部分。"""
    branch: RouteBranch
    domain_hint: str | None = None


class ClarificationStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> PendingClarification | None: ...

    @abstractmethod
    def put(self, session_id: str, pending: PendingClarification) -> None: ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...


class InMemoryClarificationStore(ClarificationStore):
    """V1 单进程内存实现——`api/main.py` 里的单例，测试换成每个测试各自
    一份新实例(同 InMemoryOnboardingSessionStore 的既有用法)。"""

    def __init__(self) -> None:
        self._pending: dict[str, PendingClarification] = {}

    def get(self, session_id: str) -> PendingClarification | None:
        return self._pending.get(session_id)

    def put(self, session_id: str, pending: PendingClarification) -> None:
        self._pending[session_id] = pending

    def clear(self, session_id: str) -> None:
        self._pending.pop(session_id, None)


class PostgresClarificationStore(ClarificationStore):
    """Persist pending clarification across process restarts (P1-6)."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn
        self._ensured = False

    def _connect(self):
        if psycopg2 is None:
            return None
        resolved = get_pg_dsn(self._dsn)
        if not resolved:
            return None
        try:
            return psycopg2.connect(resolved)
        except Exception:
            return None

    def _ensure(self, conn) -> None:
        if self._ensured:
            return
        cur = conn.cursor()
        cur.execute(_CREATE_TABLE_SQL)
        conn.commit()
        cur.close()
        self._ensured = True

    def get(self, session_id: str) -> PendingClarification | None:
        conn = self._connect()
        if conn is None:
            return None
        try:
            self._ensure(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT original_text, branch, domain_hint FROM pending_clarifications "
                "WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            cur.close()
        except Exception:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        try:
            branch = RouteBranch(row[1])
        except ValueError:
            return None
        return PendingClarification(
            original_text=row[0], branch=branch, domain_hint=row[2]
        )

    def put(self, session_id: str, pending: PendingClarification) -> None:
        conn = self._connect()
        if conn is None:
            raise RuntimeError("pending_clarifications: no postgres")
        try:
            self._ensure(conn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO pending_clarifications
                    (session_id, original_text, branch, domain_hint)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    original_text = EXCLUDED.original_text,
                    branch = EXCLUDED.branch,
                    domain_hint = EXCLUDED.domain_hint,
                    updated_at = now()
                """,
                (session_id, pending.original_text, pending.branch.value, pending.domain_hint),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()

    def clear(self, session_id: str) -> None:
        conn = self._connect()
        if conn is None:
            return
        try:
            self._ensure(conn)
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM pending_clarifications WHERE session_id = %s", (session_id,)
            )
            conn.commit()
            cur.close()
        except Exception:
            logger.warning("pending_clarifications clear failed", exc_info=True)
        finally:
            conn.close()


def default_clarification_store() -> ClarificationStore:
    store = PostgresClarificationStore()
    conn = store._connect()
    if conn is None:
        logger.warning("clarification: falling back to in-memory store")
        return InMemoryClarificationStore()
    try:
        store._ensure(conn)
        conn.close()
        return store
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        logger.warning("clarification: falling back to in-memory store")
        return InMemoryClarificationStore()
