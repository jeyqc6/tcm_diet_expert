"""Deterministic helpers for MQE retrieval benchmark. No LLM / Postgres."""
import pytest

from evals.mqe_retrieval_bench import (
    QueryBenchResult,
    QueryLoopTiming,
    format_report,
    merge_queries,
    results_to_json,
    summarize_results,
)


def test_merge_queries_dedupes_and_keeps_original_first():
    assert merge_queries("原始", ["改写A", "原始", "  ", "改写B"]) == [
        "原始",
        "改写A",
        "改写B",
    ]


def test_summarize_results_averages():
    results = [
        QueryBenchResult(
            query="q1",
            domain="tcm",
            mqe_s=2.0,
            variants=["a"],
            queries=["q1", "a"],
            loop_timings=[
                QueryLoopTiming(0, "q1", 0.1, 0.05, 5, 5),
                QueryLoopTiming(1, "a", 0.08, 0.04, 5, 5),
            ],
            e2e_no_mqe_s=0.11,
            e2e_mqe_s=2.5,
        ),
        QueryBenchResult(
            query="q2",
            domain="tcm",
            mqe_s=4.0,
            variants=[],
            queries=["q2"],
            loop_timings=[QueryLoopTiming(0, "q2", 0.09, 0.06, 5, 5)],
            e2e_no_mqe_s=0.12,
            e2e_mqe_s=4.2,
        ),
    ]
    summary = summarize_results(results)
    assert summary["n"] == 2
    assert summary["avg_mqe_s"] == 3.0
    assert summary["avg_loop_s"] == pytest.approx(0.21)
    assert summary["avg_e2e_no_mqe_s"] == pytest.approx(0.115)
    assert summary["avg_e2e_mqe_s"] == pytest.approx(3.35)


def test_format_report_includes_query_and_summary():
    result = QueryBenchResult(
        query="测试问题",
        domain="tcm",
        mqe_s=1.5,
        variants=["改写"],
        queries=["测试问题", "改写"],
        loop_timings=[QueryLoopTiming(0, "测试问题", 0.07, 0.05, 10, 8)],
        e2e_no_mqe_s=0.1,
        e2e_mqe_s=1.8,
    )
    text = format_report([result], model="demo", provider="openrouter")
    assert "测试问题" in text
    assert "SUMMARY" in text
    assert "openrouter" in text


def test_results_to_json_roundtrip_shape():
    result = QueryBenchResult(
        query="q",
        domain="tcm",
        mqe_s=1.0,
        variants=[],
        queries=["q"],
        loop_timings=[],
    )
    payload = results_to_json([result])
    assert payload["summary"]["n"] == 1
    assert payload["items"][0]["query"] == "q"
    assert payload["items"][0]["mqe_plus_loop_s"] == 1.0
