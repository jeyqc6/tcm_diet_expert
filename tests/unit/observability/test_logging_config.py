"""configure_logging: JSON/text format, auto trace_id, health-field redaction."""
from __future__ import annotations

import json
import logging
from io import StringIO

from backend.logging_config import configure_logging
from backend.observability.redact import redact_log_payload
from backend.observability.tracing import stage_log, start_request_trace, use_memory_backend


def _reset_logger():
    logger = logging.getLogger("diet_expert")
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_json_format_includes_trace_id_from_contextvar(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    stream = StringIO()
    try:
        configure_logging(stream=stream, force=True)
        use_memory_backend()
        log = logging.getLogger("diet_expert.test_logging")
        with start_request_trace("c" * 32, name="chat"):
            log.info("hello from request")
        line = stream.getvalue().strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["trace_id"] == "c" * 32
        assert payload["message"] == "hello from request"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "diet_expert.test_logging"
    finally:
        _reset_logger()


def test_text_format_includes_trace_id(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "text")
    stream = StringIO()
    try:
        configure_logging(stream=stream, force=True)
        use_memory_backend()
        log = logging.getLogger("diet_expert.test_logging")
        with start_request_trace("d" * 32, name="chat"):
            log.info("plain")
        line = stream.getvalue()
        assert "trace_id=" + "d" * 32 in line
        assert "plain" in line
    finally:
        _reset_logger()


def test_json_formatter_redacts_sensitive_stage_fields(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LANGFUSE_CAPTURE_IO", "0")
    stream = StringIO()
    try:
        configure_logging(stream=stream, force=True)
        log = logging.getLogger("diet_expert.test_logging")
        stage_log(log, "router", query="气虚质午餐吃什么", branch="fact_query")
        payload = json.loads(stream.getvalue().strip().splitlines()[-1])
        assert payload["stage"] == "router"
        assert payload["branch"] == "fact_query"
        assert payload["query"]["redacted"] is True
        dumped = json.dumps(payload, ensure_ascii=False)
        assert "气虚质" not in dumped
    finally:
        _reset_logger()


def test_redact_log_payload_leaves_operational_fields():
    out = redact_log_payload(
        {"stage": "llm", "tokens": 12, "cost_est": 0.01, "query": "麻婆豆腐"}
    )
    assert out["stage"] == "llm"
    assert out["tokens"] == 12
    assert out["query"]["redacted"] is True
    assert "麻婆豆腐" not in json.dumps(out, ensure_ascii=False)
