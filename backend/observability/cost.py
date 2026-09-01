#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local USD cost estimate for M12 (single-query cost).

Langfuse can infer cost from `model` + `usage_details` for well-known names.
We still compute a local estimate so structured logs carry `cost_est` even
when Langfuse is disabled, and so Ollama (free, local) is recorded as $0
instead of being looked up as an unknown cloud model.

ENGINEERING §2 pit 3: parallel SubAgents save wall-clock, not tokens. The
request-scoped accumulator below sums every `complete()` in the turn so the
root `chat` span records the total, not `max(tcm, nutrition)`.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from backend.llm.providers.base import TokenUsage

# USD per 1M tokens: (input, output). Approximate list prices; prefix-matched.
# Unknown models return None rather than guessing.
_PRICE_PER_MILLION: list[tuple[str, tuple[float, float]]] = [
    ("gpt-4o-mini", (0.15, 0.60)),
    ("gpt-4o", (2.50, 10.00)),
    ("gpt-4.1-mini", (0.40, 1.60)),
    ("gpt-4.1", (2.00, 8.00)),
    ("claude-haiku-4-5", (1.00, 5.00)),
    ("claude-haiku-4", (1.00, 5.00)),
    ("claude-3-5-haiku", (0.80, 4.00)),
    ("claude-sonnet-5", (3.00, 15.00)),
    ("claude-sonnet-4-5", (3.00, 15.00)),
    ("claude-3-5-sonnet", (3.00, 15.00)),
    ("claude-3-haiku", (0.25, 1.25)),
    ("claude-3-sonnet", (3.00, 15.00)),
    ("claude-3-opus", (15.00, 75.00)),
]


def _prices_for_model(model: str) -> tuple[float, float] | None:
    lowered = (model or "").strip().lower()
    if not lowered:
        return None
    for prefix, prices in _PRICE_PER_MILLION:
        if lowered.startswith(prefix):
            return prices
    return None


def estimate_cost_usd(
    model: str,
    usage: TokenUsage | None,
    *,
    provider: str = "",
) -> float | None:
    """Return USD cost or None if we cannot estimate.

    Ollama is local and unmetered → 0.0. Missing usage → None (do not invent tokens).
    """
    if (provider or "").strip().lower() == "ollama":
        return 0.0
    if usage is None:
        return None
    prices = _prices_for_model(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (usage.input_tokens * in_price + usage.output_tokens * out_price) / 1_000_000.0


@dataclass
class RequestCost:
    """Running total for one HTTP request. Parallel calls still add, not max."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_est: float | None = None
    cost_incomplete: bool = False

    def add(self, *, usage: TokenUsage | None, cost_est: float | None) -> None:
        self.calls += 1
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.total_tokens += usage.total_tokens or (
                usage.input_tokens + usage.output_tokens
            )
        if cost_est is None:
            # A real call with no estimate — do not report the request as $0.
            self.cost_incomplete = True
            return
        self.cost_est = (self.cost_est or 0.0) + cost_est


_request_cost: ContextVar[RequestCost | None] = ContextVar(
    "diet_expert_request_cost", default=None
)


def current_request_cost() -> RequestCost | None:
    return _request_cost.get()


def record_llm_call(*, usage: TokenUsage | None, cost_est: float | None) -> None:
    acc = _request_cost.get()
    if acc is None:
        return
    acc.add(usage=usage, cost_est=cost_est)


@contextmanager
def request_cost_scope() -> Iterator[RequestCost]:
    acc = RequestCost()
    token = _request_cost.set(acc)
    try:
        yield acc
    finally:
        _request_cost.reset(token)
