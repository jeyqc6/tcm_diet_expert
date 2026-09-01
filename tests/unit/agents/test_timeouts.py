"""ENGINEERING §1.1 / §2: SubAgent 45s timeout, chain wait_for, CancelledError."""
from __future__ import annotations

import asyncio

import pytest

from backend.agents.timeouts import (
    DEFAULT_CHAIN_TIMEOUT_S,
    DEFAULT_SUBAGENT_TIMEOUT_S,
    aiter_with_timeout,
    chain_timeout_s,
    reraise_if_cancelled,
    subagent_timeout_s,
)
from backend.exceptions import ChainTimeoutError, SubAgentTimeoutError
from backend.llm.adapter import LLMResult, ModelTier
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.server import DietExpertMcpServer


def _run(coro):
    return asyncio.run(coro)


def test_reraise_if_cancelled_raises_stored_cancelled_error():
    with pytest.raises(asyncio.CancelledError):
        reraise_if_cancelled("ok", asyncio.CancelledError())


def test_reraise_if_cancelled_ignores_ordinary_exceptions():
    reraise_if_cancelled(RuntimeError("side failed"), "ok")


def test_timeout_env_defaults(monkeypatch):
    monkeypatch.delenv("SUBAGENT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("CHAIN_TIMEOUT_S", raising=False)
    assert subagent_timeout_s() == DEFAULT_SUBAGENT_TIMEOUT_S
    assert chain_timeout_s() == DEFAULT_CHAIN_TIMEOUT_S


def test_timeout_env_override_and_invalid(monkeypatch):
    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "0.2")
    monkeypatch.setenv("CHAIN_TIMEOUT_S", "not-a-number")
    assert subagent_timeout_s() == 0.2
    assert chain_timeout_s() == DEFAULT_CHAIN_TIMEOUT_S
    monkeypatch.setenv("CHAIN_TIMEOUT_S", "0")
    assert chain_timeout_s() == DEFAULT_CHAIN_TIMEOUT_S


def test_wait_for_on_gather_cancels_both_sides():
    """ENGINEERING §2 pit 2: wrapping gather with wait_for must cancel leftovers."""
    cancelled: list[str] = []

    async def side(name: str) -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        return name

    async def run() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(side("tcm"), side("nutrition"), return_exceptions=True),
                timeout=0.05,
            )

    _run(run())
    assert set(cancelled) == {"tcm", "nutrition"}


def test_aiter_with_timeout_cancels_pending_work():
    cancelled = False

    async def slow_gen():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
            yield "should-not-yield"
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def run() -> None:
        with pytest.raises(ChainTimeoutError):
            async for _ in aiter_with_timeout(slow_gen(), timeout=0.05):
                pass

    _run(run())
    assert cancelled is True


def test_aiter_with_timeout_passes_through_fast_chunks():
    async def gen():
        yield "a"
        yield "b"

    async def run() -> list[str]:
        out = []
        async for chunk in aiter_with_timeout(gen(), timeout=1.0):
            out.append(chunk)
        return out

    assert _run(run()) == ["a", "b"]


def test_subagent_timeout_cancels_in_flight_complete(monkeypatch):
    from backend.agents.tcm_subagent import run_tcm_subagent

    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "0.05")
    cancelled = False

    async def complete(messages, *, tools=None, **kwargs):
        nonlocal cancelled
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return LLMResult(text="should not", model="m", tier=ModelTier.DEV, provider="fake")

    base = default_tool_definitions()
    tools = {
        name: ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=lambda **kw: {"stub": True},
        )
        for name, tool in base.items()
    }
    server = DietExpertMcpServer(tools=tools)

    with pytest.raises(SubAgentTimeoutError):
        _run(run_tcm_subagent("阳虚质吃什么", server, complete=complete))
    assert cancelled is True
