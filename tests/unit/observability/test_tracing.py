"""
Tests for backend/observability: redaction, cost estimate, in-memory span tree.

Does not contact Langfuse Cloud. Memory backend records the same shape the
real SDK would receive.
"""
from __future__ import annotations

import json
import logging

from backend.llm.providers.base import TokenUsage
from backend.observability.cost import estimate_cost_usd
from backend.observability.redact import (
    capture_io_enabled,
    redact_messages,
    redact_text,
    redact_tool_args,
    summarize_tool_result,
)
from backend.observability.tracing import (
    current_trace_id,
    is_tracing_enabled,
    observation,
    stage_log,
    start_request_trace,
    update_current,
    use_memory_backend,
)


def test_tracing_disabled_without_keys():
    assert is_tracing_enabled() is False
    with observation("should_be_noop", as_type="span"):
        update_current(output="x")
    assert current_trace_id() is None


def test_memory_backend_nests_spans_and_records_generation():
    backend = use_memory_backend()
    with start_request_trace("a" * 32, name="chat", session_id="s1", user_id="u1") as root:
        assert current_trace_id() == "a" * 32
        with observation("router", as_type="chain"):
            update_current(output={"branch": "fact_query"})
        with observation("llm.complete", as_type="generation", model="gpt-4o-mini"):
            update_current(
                as_type="generation",
                usage_details={"input": 10, "output": 20, "total": 30},
                cost_details={"total": 0.0001},
                output={"stop_reason": "stop"},
            )
        root.update(output={"status": "ok"})
    assert current_trace_id() is None

    names = [s.name for s in backend.spans]
    assert names == ["chat", "router", "llm.complete"]
    chat, router, gen = backend.spans
    assert router.parent == "chat"
    assert gen.parent == "chat"
    assert gen.as_type == "generation"
    assert gen.model == "gpt-4o-mini"
    assert gen.usage_details == {"input": 10, "output": 20, "total": 30}
    assert router.output == {"branch": "fact_query"}
    assert chat.output == {"status": "ok"}
    assert chat.trace_id == "a" * 32
    assert all(s.closed for s in backend.spans)


def test_redact_text_default_hides_body(monkeypatch):
    monkeypatch.delenv("LANGFUSE_CAPTURE_IO", raising=False)
    assert capture_io_enabled() is False
    redacted = redact_text("气虚质午餐吃什么")
    assert redacted["chars"] == len("气虚质午餐吃什么")
    assert redacted["redacted"] is True
    assert "text" not in redacted


def test_redact_text_capture_io(monkeypatch):
    monkeypatch.setenv("LANGFUSE_CAPTURE_IO", "1")
    # capture_io_enabled calls load_env(); env var is already set.
    from backend.observability import redact as redact_mod

    assert redact_mod.capture_io_enabled() is True
    redacted = redact_text("hello")
    assert redacted["text"] == "hello"
    assert redacted["redacted"] is False


def test_diet_log_result_always_stripped():
    summary = summarize_tool_result(
        "query_diet_log",
        {"entries": [{"raw_input": "麻婆豆腐"}], "time_range": "今天", "ok": True},
    )
    assert summary["entry_count"] == 1
    assert summary["time_range"] == "今天"
    assert "麻婆豆腐" not in json.dumps(summary, ensure_ascii=False)


def test_retrieval_result_keeps_source_ids_not_text():
    summary = summarize_tool_result(
        "retrieve_tcm",
        [{"source_id": "tcm_1", "text": "红枣性温"}],
    )
    assert summary["source_ids"] == ["tcm_1"]
    assert "红枣性温" not in json.dumps(summary, ensure_ascii=False)


def test_redact_tool_args_keeps_safe_keys():
    args = redact_tool_args(
        "query_diet_log",
        {"time_range": "昨天", "aggregation": "raw", "secret_note": "low calorie"},
    )
    assert args["time_range"] == "昨天"
    assert args["aggregation"] == "raw"
    assert args["secret_note"]["redacted"] is True


def test_redact_messages_drops_bodies():
    out = redact_messages(
        [
            {"role": "user", "content": "我昨天吃了火锅"},
            {"role": "assistant", "content": "建议", "tool_calls": [{"name": "retrieve_tcm"}]},
        ]
    )
    assert out[0]["content"]["redacted"] is True
    assert "火锅" not in json.dumps(out, ensure_ascii=False)
    assert out[1]["tool_call_names"] == ["retrieve_tcm"]


def test_cost_estimate_known_model():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000)
    assert estimate_cost_usd("gpt-4o-mini-2024-07-18", usage) == 0.15


def test_cost_estimate_ollama_is_zero():
    usage = TokenUsage(input_tokens=100, output_tokens=100, total_tokens=200)
    assert estimate_cost_usd("qwen3:0.6b", usage, provider="ollama") == 0.0


def test_cost_estimate_unknown_model_is_none():
    usage = TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20)
    assert estimate_cost_usd("mystery-model", usage) is None


def test_cost_estimate_missing_usage_is_none():
    assert estimate_cost_usd("gpt-4o", None) is None


def test_stage_log_includes_trace_id(caplog):
    use_memory_backend()
    log = logging.getLogger("diet_expert.test_stage")
    with caplog.at_level(logging.INFO, logger=log.name):
        with start_request_trace("b" * 32, name="chat"):
            stage_log(log, "router", branch="fact_query", latency_ms=12.5)
    payload = json.loads(caplog.records[-1].getMessage())
    assert payload["trace_id"] == "b" * 32
    assert payload["stage"] == "router"
    assert payload["branch"] == "fact_query"
