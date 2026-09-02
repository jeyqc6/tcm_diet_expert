#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pending critical-fact store (PRD §10.2 human-in-the-loop).

Scan may run on every turn. Hits are stored here until the user confirms.
They are NOT merged into UserProfileContext for the current turn and are
NOT written via write_memory(critical) until POST .../confirm.

D34: this replaces the 2026-08-28 "scan → immediate UPSERT + same-turn merge"
shortcut. Confirm is the only write path for scanner-discovered facts.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn
from backend.i18n import current_locale, normalize_locale, t
from backend.memory.critical_fact_scanner import display_allergen_for_locale

logger = logging.getLogger("diet_expert.memory.pending_critical_facts")

DEFAULT_USER_ID = "default_user"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pending_critical_facts (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default_user',
    session_id      TEXT NOT NULL,
    allergens       TEXT[] NOT NULL DEFAULT '{}',
    supplements     JSONB NOT NULL DEFAULT '[]'::jsonb,
    preferences     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class PendingCriticalFact:
    pending_id: str
    user_id: str
    session_id: str
    allergens: tuple[str, ...] = ()
    supplements: tuple[str, ...] = ()
    preferences: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_event_dict(self, locale: str | None = None) -> dict[str, Any]:
        loc = locale if locale is not None else current_locale()
        return {
            "pending_id": self.pending_id,
            "allergens": list(self.allergens),
            "supplements": list(self.supplements),
            "preferences": dict(self.preferences),
            "detail": _detail_text(
                self.allergens, self.supplements, self.preferences, loc
            ),
        }


def _detail_text(
    allergens: tuple[str, ...],
    supplements: tuple[str, ...],
    preferences: dict[str, Any],
    locale: str = "zh",
) -> str:
    locale = normalize_locale(locale)
    item_separator = ", " if locale == "en" else "、"
    part_separator = "; " if locale == "en" else "，"
    parts = []
    if allergens:
        display_names = [display_allergen_for_locale(name, locale) for name in allergens]
        parts.append(t("pending.allergen", locale, names=item_separator.join(display_names)))
    if supplements:
        parts.append(t("pending.supplement", locale, names=item_separator.join(supplements)))
    if preferences:
        pref_text = json.dumps(preferences, ensure_ascii=False)
        parts.append(t("pending.preferences", locale, names=pref_text))
    joined = part_separator.join(parts) if parts else t("pending.generic", locale)
    return t("pending.detail", locale, joined=joined)


class PendingCriticalFactStore(Protocol):
    def put(self, fact: PendingCriticalFact) -> PendingCriticalFact: ...
    def get(self, pending_id: str) -> PendingCriticalFact | None: ...
    def list_for_session(self, session_id: str) -> list[PendingCriticalFact]: ...
    def delete(self, pending_id: str) -> PendingCriticalFact | None: ...


class InMemoryPendingCriticalFactStore:
    def __init__(self) -> None:
        self._items: dict[str, PendingCriticalFact] = {}

    def put(self, fact: PendingCriticalFact) -> PendingCriticalFact:
        self._items[fact.pending_id] = fact
        return fact

    def get(self, pending_id: str) -> PendingCriticalFact | None:
        return self._items.get(pending_id)

    def list_for_session(self, session_id: str) -> list[PendingCriticalFact]:
        return [f for f in self._items.values() if f.session_id == session_id]

    def delete(self, pending_id: str) -> PendingCriticalFact | None:
        return self._items.pop(pending_id, None)


class PostgresPendingCriticalFactStore:
    """Best-effort PG persist. Creates the table if missing (no Alembic yet)."""

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
        cur.execute(
            "ALTER TABLE pending_critical_facts "
            "ADD COLUMN IF NOT EXISTS preferences JSONB NOT NULL DEFAULT '{}'::jsonb"
        )
        conn.commit()
        cur.close()
        self._ensured = True

    def put(self, fact: PendingCriticalFact) -> PendingCriticalFact:
        conn = self._connect()
        if conn is None:
            raise RuntimeError("pending_critical_facts: no postgres")
        try:
            self._ensure(conn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO pending_critical_facts
                    (id, user_id, session_id, allergens, supplements, preferences)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    fact.pending_id,
                    fact.user_id,
                    fact.session_id,
                    list(fact.allergens),
                    json.dumps([{"name": n, "dose": None} for n in fact.supplements]),
                    json.dumps(fact.preferences, ensure_ascii=False),
                ),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()
        return fact

    def get(self, pending_id: str) -> PendingCriticalFact | None:
        return self._fetch_one("id = %s", (pending_id,))

    def list_for_session(self, session_id: str) -> list[PendingCriticalFact]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            self._ensure(conn)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT * FROM pending_critical_facts WHERE session_id = %s ORDER BY created_at",
                (session_id,),
            )
            rows = cur.fetchall()
            cur.close()
        except Exception:
            return []
        finally:
            conn.close()
        return [_row_to_fact(dict(r)) for r in rows]

    def delete(self, pending_id: str) -> PendingCriticalFact | None:
        existing = self.get(pending_id)
        if existing is None:
            return None
        conn = self._connect()
        if conn is None:
            return None
        try:
            self._ensure(conn)
            cur = conn.cursor()
            cur.execute("DELETE FROM pending_critical_facts WHERE id = %s", (pending_id,))
            conn.commit()
            cur.close()
        finally:
            conn.close()
        return existing

    def _fetch_one(self, where: str, params: tuple) -> PendingCriticalFact | None:
        conn = self._connect()
        if conn is None:
            return None
        try:
            self._ensure(conn)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(f"SELECT * FROM pending_critical_facts WHERE {where}", params)
            row = cur.fetchone()
            cur.close()
        except Exception:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        return _row_to_fact(dict(row))


def _row_to_fact(row: dict[str, Any]) -> PendingCriticalFact:
    supplements = row.get("supplements") or []
    if isinstance(supplements, str):
        try:
            supplements = json.loads(supplements)
        except json.JSONDecodeError:
            supplements = []
    names: list[str] = []
    for item in supplements:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
        elif isinstance(item, str):
            names.append(item)
    created = row.get("created_at")
    if isinstance(created, datetime):
        created_s = created.astimezone(timezone.utc).isoformat()
    else:
        created_s = str(created or "")
    preferences = row.get("preferences") or {}
    if isinstance(preferences, str):
        try:
            preferences = json.loads(preferences)
        except json.JSONDecodeError:
            preferences = {}
    if not isinstance(preferences, dict):
        preferences = {}
    return PendingCriticalFact(
        pending_id=row["id"],
        user_id=row.get("user_id") or DEFAULT_USER_ID,
        session_id=row["session_id"],
        allergens=tuple(row.get("allergens") or ()),
        supplements=tuple(names),
        preferences=preferences,
        created_at=created_s,
    )


def new_pending_id() -> str:
    return str(uuid.uuid4())


def default_pending_store() -> PendingCriticalFactStore:
    """Postgres when DSN works, otherwise process-local memory."""
    store = PostgresPendingCriticalFactStore()
    conn = store._connect()
    if conn is None:
        logger.warning("pending_critical_facts: falling back to in-memory store")
        return InMemoryPendingCriticalFactStore()
    try:
        store._ensure(conn)
        conn.close()
        return store
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        logger.warning("pending_critical_facts: falling back to in-memory store")
        return InMemoryPendingCriticalFactStore()
