#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-level env helpers: load repo-root `.env`, read shared config."""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_LOADED = False

# Provider / model / API-key vars on the LLM call path.  Values in ``.env``
# override a stale shell export so uvicorn picks up key rotations without
# restarting the parent terminal.  Non-LLM keys still keep shell-first.
_LLM_ENV_KEYS = frozenset({
    "MODEL_TIER",
    "CHAT_MODEL",
    "LLM_PROVIDER",
    "LLM_PROVIDER_DEV",
    "LLM_PROVIDER_PROD",
    "LLM_MODEL_DEV",
    "LLM_MODEL_PROD",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_CONTEXT_WINDOW_OVERRIDE",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
})


def overlay_llm_env_from_file(env_path: Path) -> None:
    """Copy LLM-related keys from ``env_path`` onto ``os.environ``.

    Empty / missing keys in the file are left unchanged (shell value kept).
    """
    if not env_path.is_file():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    values = dotenv_values(env_path)
    for key in _LLM_ENV_KEYS:
        raw = values.get(key)
        if raw is None or raw == "":
            continue
        os.environ[key] = raw


def load_env() -> None:
    """Load `<repo>/.env` once.

    Non-LLM keys: already-set shell vars win (``override=False``).
    LLM keys: ``.env`` wins so API keys / MODEL_TIER follow the file.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = _PROJECT_ROOT / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except ImportError:
            pass
        overlay_llm_env_from_file(env_path)
    _ENV_LOADED = True


def get_pg_dsn(explicit: str | None = None) -> str | None:
    """Postgres DSN: explicit arg > os.environ (after .env load) > None."""
    if explicit:
        return explicit
    load_env()
    return os.environ.get("DIET_EXPERT_PG_DSN")
