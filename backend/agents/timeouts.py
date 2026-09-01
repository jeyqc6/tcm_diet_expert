#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ENGINEERING §1.1 / §2 orchestration timeouts.

Adapter owns the per-call 20s HTTP timeout. This module owns the other two
layers: 45s per SubAgent and 90s for the whole request chain. Thresholds are
env-overridable (ENGINEERING §5); defaults match the table in §1.1.

`asyncio.wait_for` is the cancellation primitive: on timeout it cancels the
inner task, which is what stops a still-running SubAgent from burning tokens
(§2 pit 2). `return_exceptions=True` on `asyncio.gather` would otherwise
swallow `CancelledError` as a "result" — `reraise_if_cancelled` puts that
back on the raise path so it reaches the HTTP generator.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, TypeVar

from backend.env import load_env
from backend.exceptions import ChainTimeoutError

logger = logging.getLogger("diet_expert.agents.timeouts")

DEFAULT_SUBAGENT_TIMEOUT_S = 45.0
DEFAULT_CHAIN_TIMEOUT_S = 90.0

T = TypeVar("T")


def _positive_float(env_name: str, default: float) -> float:
    load_env()
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid %s=%r, using default %.1f", env_name, raw, default)
        return default
    if value <= 0:
        logger.warning("non-positive %s=%r, using default %.1f", env_name, raw, default)
        return default
    return value


def subagent_timeout_s() -> float:
    """ENGINEERING §1.1: abandon this side, fall back to unilateral output."""
    return _positive_float("SUBAGENT_TIMEOUT_S", DEFAULT_SUBAGENT_TIMEOUT_S)


def chain_timeout_s() -> float:
    """ENGINEERING §1.1: force-close the request, cancel leftover SubAgents."""
    return _positive_float("CHAIN_TIMEOUT_S", DEFAULT_CHAIN_TIMEOUT_S)


def reraise_if_cancelled(*results: object) -> None:
    """`asyncio.gather(..., return_exceptions=True)` stores CancelledError
    as a result instead of propagating it. Re-raise so chain timeout / client
    disconnect actually stops the sibling SubAgent (ENGINEERING §2 pit 2)."""
    for item in results:
        if isinstance(item, asyncio.CancelledError):
            raise item


async def _aclose_quietly(agen: AsyncIterator[object]) -> None:
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except (TimeoutError, StopAsyncIteration, GeneratorExit, asyncio.CancelledError):
        return
    except Exception:  # noqa: BLE001 — closing a cancelled generator is best-effort
        logger.debug("async generator aclose failed", exc_info=True)


async def aiter_with_timeout(
    agen: AsyncIterator[T],
    timeout: float,
) -> AsyncIterator[T]:
    """Iterate `agen` until it ends or `timeout` seconds elapse.

    Deadline is shared across chunks so a slow first `__anext__` (all LLM
    work happens before the first SSE token) cannot reset the budget.
    Timeout cancels the generator, which cancels whatever it is awaiting
    (typically `asyncio.gather` of the two SubAgents).
    """
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                chunk = await asyncio.wait_for(anext(agen), timeout=remaining)
            except StopAsyncIteration:
                return
            yield chunk
    except TimeoutError as exc:
        await _aclose_quietly(agen)
        raise ChainTimeoutError(
            f"request chain exceeded {timeout:.1f}s"
        ) from exc
    except asyncio.CancelledError:
        await _aclose_quietly(agen)
        raise
    except BaseException:
        await _aclose_quietly(agen)
        raise
