"""ENGINEERING §2 pit 3: request-scoped token/cost sum, not wall-clock max."""
from __future__ import annotations

import pytest

from backend.llm.adapter import LLMResult, ModelTier
from backend.llm.providers.base import TokenUsage
from backend.observability.cost import (
    RequestCost,
    current_request_cost,
    record_llm_call,
    request_cost_scope,
)


def test_request_cost_sums_parallel_sides_not_max():
    acc = RequestCost()
    acc.add(
        usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
        cost_est=0.01,
    )
    acc.add(
        usage=TokenUsage(input_tokens=80, output_tokens=40, total_tokens=120),
        cost_est=0.02,
    )
    assert acc.calls == 2
    assert acc.total_tokens == 240
    assert acc.input_tokens == 180
    assert acc.output_tokens == 60
    assert acc.cost_est == pytest.approx(0.03)
    assert acc.cost_incomplete is False


def test_missing_cost_est_marks_incomplete_without_inventing_zero():
    acc = RequestCost()
    acc.add(
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        cost_est=0.01,
    )
    acc.add(usage=None, cost_est=None)
    assert acc.total_tokens == 15
    assert acc.cost_est == 0.01
    assert acc.cost_incomplete is True


def test_all_missing_cost_stays_none():
    acc = RequestCost()
    acc.add(usage=None, cost_est=None)
    assert acc.cost_est is None
    assert acc.cost_incomplete is True
    assert acc.calls == 1


def test_request_cost_scope_records_and_resets():
    assert current_request_cost() is None
    with request_cost_scope() as acc:
        record_llm_call(
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost_est=0.0,
        )
        assert current_request_cost() is acc
        assert acc.total_tokens == 2
        assert acc.cost_est == 0.0
    assert current_request_cost() is None


def test_record_outside_scope_is_noop():
    record_llm_call(
        usage=TokenUsage(input_tokens=99, output_tokens=1, total_tokens=100),
        cost_est=1.0,
    )
    assert current_request_cost() is None


def test_metering_reads_usage_off_llm_result():
    result = LLMResult(
        text="ok",
        model="m",
        tier=ModelTier.DEV,
        provider="fake",
        usage=TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4),
        cost_est=0.005,
    )
    with request_cost_scope() as acc:
        record_llm_call(usage=result.usage, cost_est=result.cost_est)
        assert acc.total_tokens == 4
        assert acc.cost_est == pytest.approx(0.005)
