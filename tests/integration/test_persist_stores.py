"""
Round-trip persist for clarification + onboarding (P1-6).

Needs real Postgres (same skip convention as test_session_store.py).
"""
from __future__ import annotations

import uuid

import pytest

from backend.agents.clarification import (
    PendingClarification,
    PostgresClarificationStore,
)
from backend.agents.routing import RouteBranch
from backend.env import get_pg_dsn
from backend.onboarding.session_store import (
    ActiveOnboarding,
    PostgresOnboardingSessionStore,
)

psycopg2 = pytest.importorskip("psycopg2")

_DSN = get_pg_dsn()
if not _DSN:
    pytest.skip(
        "DIET_EXPERT_PG_DSN not configured, skipping persist-store tests",
        allow_module_level=True,
    )
else:
    try:
        _conn = psycopg2.connect(_DSN, connect_timeout=3)
        _conn.close()
    except Exception:
        pytest.skip(
            "cannot connect to Postgres, skipping persist-store tests",
            allow_module_level=True,
        )


def test_clarification_postgres_round_trip():
    store = PostgresClarificationStore(dsn=_DSN)
    session_id = f"clar-{uuid.uuid4()}"
    pending = PendingClarification(
        original_text="这个能不能吃",
        branch=RouteBranch.CANDIDATE_EVAL,
        domain_hint="tcm",
    )
    store.put(session_id, pending)
    loaded = store.get(session_id)
    assert loaded is not None
    assert loaded.original_text == "这个能不能吃"
    assert loaded.branch is RouteBranch.CANDIDATE_EVAL
    assert loaded.domain_hint == "tcm"
    store.clear(session_id)
    assert store.get(session_id) is None


def test_onboarding_postgres_round_trip():
    store = PostgresOnboardingSessionStore(dsn=_DSN)
    user_id = f"ob-{uuid.uuid4()}"
    store.put(user_id, ActiveOnboarding(step_id="allergens", state={"asked": True}))
    loaded = store.get(user_id)
    assert loaded is not None
    assert loaded.step_id == "allergens"
    assert loaded.state == {"asked": True}
    store.clear(user_id)
    assert store.get(user_id) is None
