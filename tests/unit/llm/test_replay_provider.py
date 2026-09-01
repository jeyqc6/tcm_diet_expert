"""
测试目标：docs/ENGINEERING.md §7.2 的 Record/Replay fixture 机制
对应实现：backend/llm/providers/replay.py
覆盖要求：单测，注入假 inner provider（同 test_adapter.py `_FakeProvider` 的模式），
不打真实网络。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from backend.llm.providers.base import ProviderResponse, ToolCall, TokenUsage
from backend.llm.providers.replay import (
    ReplayFixtureMissing,
    ReplayProvider,
    compute_fingerprint,
    replay_provider_for,
)


def _run(coro):
    return asyncio.run(coro)


class _FakeInner:
    def __init__(self, response: ProviderResponse):
        self._response = response
        self.calls = 0
        self.closed = False

    async def call(self, messages, *, model, tools=None, **kwargs):
        self.calls += 1
        return self._response

    def classify_error(self, exc):
        return "retryable"

    async def aclose(self):
        self.closed = True


# ---------------------------------------------------------------------------
# compute_fingerprint：确定性 + 内容敏感
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic():
    messages = [{"role": "user", "content": "红烧肉能不能吃"}]
    a = compute_fingerprint("claude-haiku-4-5", messages, tools=None)
    b = compute_fingerprint("claude-haiku-4-5", messages, tools=None)
    assert a == b


def test_fingerprint_changes_with_message_content():
    """这是"prompt 被意外改动"检测器的基础——内容变了指纹必须跟着变，
    replay 模式才能靠"查不到"发现 prompt 漂移。"""
    fp1 = compute_fingerprint("m", [{"role": "user", "content": "红烧肉能不能吃"}])
    fp2 = compute_fingerprint("m", [{"role": "user", "content": "红烧肉能吃吗"}])
    assert fp1 != fp2


def test_fingerprint_changes_with_model():
    messages = [{"role": "user", "content": "hi"}]
    assert compute_fingerprint("model-a", messages) != compute_fingerprint("model-b", messages)


def test_fingerprint_ignores_key_order():
    """dict 字段顺序不影响指纹——sort_keys=True 保证内容相同的调用落到同一个
    fixture 上，不会因为 Python dict 的插入顺序偶然产生两份重复 fixture。"""
    m1 = [{"role": "user", "content": "x"}]
    fp1 = compute_fingerprint("m", m1, tools=[{"name": "a", "description": "d", "input_schema": {}}])
    fp2 = compute_fingerprint("m", m1, tools=[{"description": "d", "name": "a", "input_schema": {}}])
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# record 模式：真的调 inner，落盘
# ---------------------------------------------------------------------------


def test_record_mode_calls_inner_and_writes_fixture(tmp_path):
    inner = _FakeInner(ProviderResponse(text="红烧肉性温，可以适量食用 [source: t1]", stop_reason="stop"))
    provider = ReplayProvider(mode="record", caller="tcm_subagent", inner=inner, fixtures_dir=tmp_path)

    messages = [{"role": "user", "content": "红烧肉能不能吃"}]
    result = _run(provider.call(messages, model="claude-haiku-4-5"))

    assert result.text == "红烧肉性温，可以适量食用 [source: t1]"
    assert inner.calls == 1
    fixture_files = list(tmp_path.glob("tcm_subagent__*.json"))
    assert len(fixture_files) == 1
    data = json.loads(fixture_files[0].read_text(encoding="utf-8"))
    assert data["caller"] == "tcm_subagent"
    assert data["response"]["text"] == "红烧肉性温，可以适量食用 [source: t1]"
    assert data["request"]["messages"] == messages


def test_record_mode_without_inner_raises():
    with pytest.raises(ValueError, match="inner"):
        ReplayProvider(mode="record", caller="x", inner=None)


def test_record_mode_round_trips_tool_calls_and_usage(tmp_path):
    response = ProviderResponse(
        text="",
        stop_reason="tool_use",
        tool_calls=[ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "红烧肉"})],
        usage=TokenUsage(input_tokens=120, output_tokens=30, total_tokens=150),
    )
    inner = _FakeInner(response)
    provider = ReplayProvider(mode="record", caller="tcm", inner=inner, fixtures_dir=tmp_path)
    _run(provider.call([{"role": "user", "content": "x"}], model="m"))

    replay = ReplayProvider(mode="replay", caller="tcm", fixtures_dir=tmp_path)
    result = _run(replay.call([{"role": "user", "content": "x"}], model="m"))
    assert result.tool_calls == [ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "红烧肉"})]
    assert result.usage == TokenUsage(input_tokens=120, output_tokens=30, total_tokens=150)


# ---------------------------------------------------------------------------
# replay 模式：不打网络，按指纹查
# ---------------------------------------------------------------------------


def test_replay_mode_returns_recorded_response_without_calling_inner(tmp_path):
    inner = _FakeInner(ProviderResponse(text="录制的回答 [source: t1]", stop_reason="stop"))
    record_provider = ReplayProvider(mode="record", caller="router", inner=inner, fixtures_dir=tmp_path)
    messages = [{"role": "user", "content": "今天该吃什么"}]
    _run(record_provider.call(messages, model="m"))

    replay_provider = ReplayProvider(mode="replay", caller="router", fixtures_dir=tmp_path)
    result = _run(replay_provider.call(messages, model="m"))

    assert result.text == "录制的回答 [source: t1]"
    assert inner.calls == 1  # 只在 record 那次真的调用过，replay 完全没碰 inner


def test_replay_mode_missing_fixture_raises_with_clear_message(tmp_path):
    provider = ReplayProvider(mode="replay", caller="router", fixtures_dir=tmp_path)
    with pytest.raises(ReplayFixtureMissing, match="router"):
        _run(provider.call([{"role": "user", "content": "从没录过的问题"}], model="m"))


def test_replay_mode_prompt_drift_is_detected_as_missing_fixture(tmp_path):
    """指纹对不上就报错，这就是"prompt 被意外改动"检测器(ENGINEERING §7.2)：
    录制时的消息和回放时的消息不一样，落到不同指纹，replay 查不到旧 fixture，
    而不是静默返回一个已经不对应真实 prompt 的陈旧响应。"""
    inner = _FakeInner(ProviderResponse(text="ok", stop_reason="stop"))
    record_provider = ReplayProvider(mode="record", caller="router", inner=inner, fixtures_dir=tmp_path)
    _run(record_provider.call([{"role": "user", "content": "原始 prompt"}], model="m"))

    replay_provider = ReplayProvider(mode="replay", caller="router", fixtures_dir=tmp_path)
    with pytest.raises(ReplayFixtureMissing):
        _run(replay_provider.call([{"role": "user", "content": "改过的 prompt"}], model="m"))


def test_replay_mode_does_not_require_inner():
    provider = ReplayProvider(mode="replay", caller="x")
    assert provider.inner is None


# ---------------------------------------------------------------------------
# classify_error / aclose：委托给 inner，replay 模式下没有 inner 也不报错
# ---------------------------------------------------------------------------


def test_classify_error_delegates_to_inner():
    inner = _FakeInner(ProviderResponse(text="", stop_reason="stop"))
    provider = ReplayProvider(mode="record", caller="x", inner=inner)
    assert provider.classify_error(RuntimeError("x")) == "retryable"


def test_classify_error_without_inner_defaults_non_retryable():
    provider = ReplayProvider(mode="replay", caller="x")
    assert provider.classify_error(RuntimeError("x")) == "non_retryable"


def test_aclose_delegates_to_inner_when_present():
    inner = _FakeInner(ProviderResponse(text="", stop_reason="stop"))
    provider = ReplayProvider(mode="record", caller="x", inner=inner)
    _run(provider.aclose())
    assert inner.closed is True


def test_aclose_without_inner_is_a_noop():
    provider = ReplayProvider(mode="replay", caller="x")
    _run(provider.aclose())  # 不应该抛异常


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        ReplayProvider(mode="bogus", caller="x")


# ---------------------------------------------------------------------------
# replay_provider_for：env var 切换，replay 模式不需要任何凭据/真实 provider
# ---------------------------------------------------------------------------


def test_replay_provider_for_defaults_to_replay_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_REPLAY_MODE", raising=False)
    provider = replay_provider_for("router", fixtures_dir=tmp_path)
    assert provider.mode == "replay"
    assert provider.inner is None  # 没有构造任何需要凭据的真实 provider


def test_replay_provider_for_record_mode_builds_real_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_REPLAY_MODE", "record")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-construction-only")
    provider = replay_provider_for("router", real_provider_name="anthropic", fixtures_dir=tmp_path)
    assert provider.mode == "record"
    assert provider.inner is not None
