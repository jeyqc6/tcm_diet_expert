#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-progress onboarding state for `/api/chat`.

ChatRequest is only `{session_id, message}` — the client does not round-trip
`step_id`/`state` the way `/api/onboarding/answer` does. The agent has to
remember which question it just asked.

V1 is single-user (§1.2). Keyed by `user_id` so a page refresh (new
`session_id`) still continues the same first-conversation flow. Completing or
aborting clears the entry. Completing or skipping writes
`onboarding_done=TRUE` so `should_trigger` stays false even if this store is
gone after a restart. A stub `user_profile` row from `create_user` is not enough.

P1-6: Postgres is the default when a DSN works; InMemory remains for tests
and for local runs without Postgres.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn

logger = logging.getLogger("diet_expert.onboarding.session_store")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS onboarding_sessions (
    user_id         TEXT PRIMARY KEY,
    step_id         TEXT NOT NULL,
    state           JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


@dataclass
class ActiveOnboarding:
    step_id: str
    state: dict[str, Any] = field(default_factory=dict)


class OnboardingSessionStore(ABC):
    @abstractmethod
    def get(self, user_id: str) -> ActiveOnboarding | None: ...

    @abstractmethod
    def put(self, user_id: str, session: ActiveOnboarding) -> None: ...

    @abstractmethod
    def clear(self, user_id: str) -> None: ...


class InMemoryOnboardingSessionStore(OnboardingSessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, ActiveOnboarding] = {}

    def get(self, user_id: str) -> ActiveOnboarding | None:
        session = self._sessions.get(user_id)
        if session is None:
            return None
        return ActiveOnboarding(step_id=session.step_id, state=dict(session.state))

    def put(self, user_id: str, session: ActiveOnboarding) -> None:
        self._sessions[user_id] = ActiveOnboarding(
            step_id=session.step_id, state=dict(session.state)
        )

    def clear(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)


class PostgresOnboardingSessionStore(OnboardingSessionStore):
    """Persist in-progress onboarding across process restarts (P1-6)."""

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

    def get(self, user_id: str) -> ActiveOnboarding | None:
        conn = self._connect()
        if conn is None:
            return None
        try:
            self._ensure(conn)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT step_id, state FROM onboarding_sessions WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            cur.close()
        except Exception:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        state = row.get("state") or {}
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except json.JSONDecodeError:
                state = {}
        return ActiveOnboarding(step_id=row["step_id"], state=dict(state))

    def put(self, user_id: str, session: ActiveOnboarding) -> None:
        conn = self._connect()
        if conn is None:
            raise RuntimeError("onboarding_sessions: no postgres")
        try:
            self._ensure(conn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO onboarding_sessions (user_id, step_id, state)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    step_id = EXCLUDED.step_id,
                    state = EXCLUDED.state,
                    updated_at = now()
                """,
                (user_id, session.step_id, json.dumps(session.state, ensure_ascii=False)),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()

    def clear(self, user_id: str) -> None:
        conn = self._connect()
        if conn is None:
            return
        try:
            self._ensure(conn)
            cur = conn.cursor()
            cur.execute("DELETE FROM onboarding_sessions WHERE user_id = %s", (user_id,))
            conn.commit()
            cur.close()
        except Exception:
            logger.warning("onboarding_sessions clear failed", exc_info=True)
        finally:
            conn.close()


def default_onboarding_store() -> OnboardingSessionStore:
    store = PostgresOnboardingSessionStore()
    conn = store._connect()
    if conn is None:
        logger.warning("onboarding: falling back to in-memory store")
        return InMemoryOnboardingSessionStore()
    try:
        store._ensure(conn)
        conn.close()
        return store
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        logger.warning("onboarding: falling back to in-memory store")
        return InMemoryOnboardingSessionStore()
