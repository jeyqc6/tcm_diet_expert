"""
Shared pytest fixtures.

Langfuse must never send spans during tests, even if the developer has
LANGFUSE_* keys in `.env`. Tests that want a span tree call
`use_memory_backend()` themselves.

Same principle extended to two other places a developer's real `.env` can
leak into test behavior (found 2026-08-31 chasing 5 tests that failed only
when run for real, never in a hand-picked subset — see the two blocks below
for exactly what each one was hiding):
  1. `LANGFUSE_CAPTURE_IO=1` (this repo's own `.env` has it set, for the
     developer's own trace debugging) silently flips `redact_messages()`/
     `redact_tool_args()`/`redact_log_payload()` out of their default
     "hide message bodies" behavior — tests for that default behavior broke
     without ever touching backend/observability/redact.py.
  2. `backend/env.py` `load_env()` runs its `.env`-overlay exactly once per
     process (`_ENV_LOADED`); that overlay deliberately makes `.env` win over
     a pre-set env var for `LLM_MODEL_DEV`/`LLM_PROVIDER_DEV`/etc (so a key
     rotation in `.env` is picked up without restarting the parent shell —
     see that module's docstring, this is intentional production behavior,
     not something to "fix" there). The trap: if a test is the *first* one in
     the process to trigger that overlay, and it does so via `monkeypatch.setenv()`
     on one of those same keys *before* calling into `complete()`, the overlay
     runs after the monkeypatch and silently clobbers it with the real `.env`
     value — order-dependent, so it only reproduces if that test happens to
     run first (e.g. in isolation). Forcing the one-time overlay to already be
     done *before* any test body runs means it never again fires mid-test,
     so a test's own `monkeypatch.setenv()` on those keys always wins for the
     rest of the session.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_langfuse_network(monkeypatch):
    # Order matters: `load_env()` must run *before* the delenv below, not
    # after. `load_dotenv(..., override=False)` only means "don't clobber an
    # already-set value" — a key that's *missing* (because we just deleted
    # it) still gets (re)populated from the file. Delenv first would get
    # silently undone by this call.
    from backend.env import load_env

    load_env()
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    monkeypatch.delenv("LANGFUSE_CAPTURE_IO", raising=False)
    from backend.observability.tracing import reset_tracing_backend

    reset_tracing_backend()
    yield
    reset_tracing_backend()
