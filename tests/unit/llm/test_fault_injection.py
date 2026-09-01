"""
测试目标：docs/ENGINEERING.md §7.2"故障注入"fixture——验证
tests/fixtures/fault_injection/ 提供的假故障，配合 backend/llm/adapter.py
的 `complete()`，确实能触发 §1.2(重试与退避)/§1.3(熔断与降级)对应的行为；
以及"格式错乱的 JSON"这几个样本能被真实的下游解析函数
(backend/agents/routing.py)正确处理或正确判定为解析失败。
对应实现：tests/fixtures/fault_injection/、backend/llm/adapter.py
覆盖要求：单测，不打真实网络。
"""
from __future__ import annotations

import asyncio

import pytest

from backend.agents.routing import _parse_route_llm_json
from backend.llm.adapter import (
    CircuitBreaker,
    LLMCallError,
    ModelTier,
    NonRetryableError,
    complete,
)
from backend.llm.providers.base import ProviderResponse
from tests.fixtures.fault_injection import (
    MALFORMED_JSON_RESPONSES,
    AuthFault,
    ContentFilterFault,
    FaultInjectingProvider,
    RateLimitFault,
    ServerErrorFault,
    TimeoutFault,
    malformed_json_response,
)


def _run(coro):
    return asyncio.run(coro)


def _make_sleep_recorder():
    recorded = []

    async def _sleep(seconds):
        recorded.append(seconds)

    return _sleep, recorded


@pytest.fixture
def circuit():
    return CircuitBreaker()


# ---------------------------------------------------------------------------
# classify_error 判定逻辑必须和真实 provider 一致(否则这个 fixture 测出来的
# 行为对真实 provider 没有参考价值)。两边都走 classify_http_error。
# ---------------------------------------------------------------------------


def test_classify_error_matches_real_providers_for_rate_limit():
    provider = FaultInjectingProvider(script=[])
    assert provider.classify_error(RateLimitFault()) == "retryable"


def test_classify_error_matches_real_providers_for_server_error():
    provider = FaultInjectingProvider(script=[])
    assert provider.classify_error(ServerErrorFault(status_code=502)) == "retryable"


def test_classify_error_matches_real_providers_for_auth_fault():
    provider = FaultInjectingProvider(script=[])
    assert provider.classify_error(AuthFault()) == "non_retryable"


def test_classify_error_matches_real_providers_for_timeout():
    provider = FaultInjectingProvider(script=[])
    assert provider.classify_error(TimeoutFault("simulated network timeout")) == "retryable"


# ---------------------------------------------------------------------------
# §1.2 重试与退避：429/5xx/超时 → 退避重试后成功；401 → 立即失败不重试
# ---------------------------------------------------------------------------


def test_rate_limit_retries_then_succeeds(circuit):
    provider = FaultInjectingProvider(
        script=[RateLimitFault(), ProviderResponse(text="恢复正常", stop_reason="stop")]
    )
    sleep, delays = _make_sleep_recorder()
    result = _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert result.text == "恢复正常"
    assert len(provider.calls) == 2  # 第一次限流，第二次才成功
    assert len(delays) == 1  # 只退避了一次


def test_server_error_retries_then_succeeds(circuit):
    provider = FaultInjectingProvider(
        script=[ServerErrorFault(), ProviderResponse(text="ok", stop_reason="stop")]
    )
    sleep, _ = _make_sleep_recorder()
    result = _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert result.text == "ok"


def test_timeout_retries_then_succeeds(circuit):
    provider = FaultInjectingProvider(
        script=[TimeoutFault("timed out"), ProviderResponse(text="ok", stop_reason="stop")]
    )
    sleep, _ = _make_sleep_recorder()
    result = _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert result.text == "ok"


def test_auth_fault_fails_immediately_without_retry(circuit):
    """400/401 不重试——ENGINEERING §1.2:"立即失败,重试无意义"。"""
    provider = FaultInjectingProvider(script=[AuthFault()])
    sleep, delays = _make_sleep_recorder()
    with pytest.raises(NonRetryableError):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert len(provider.calls) == 1  # 只调用了一次，没有重试
    assert delays == []  # 没有退避等待


def test_content_filter_fails_immediately_without_retry(circuit):
    """内容策略拒绝不重试——ENGINEERING §1.2:"走 fallback,不重试"。这条不是
    异常，是 `stop_reason == "content_filter"` 的正常响应，`complete()` 自己
    检查这个字段并转成 NonRetryableError，不经过 `classify_error()`。"""
    provider = FaultInjectingProvider(script=[ContentFilterFault.response()])
    sleep, delays = _make_sleep_recorder()
    with pytest.raises(NonRetryableError, match="content_filter"):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert len(provider.calls) == 1
    assert delays == []


def test_exhausting_all_retries_raises_llm_call_error(circuit):
    provider = FaultInjectingProvider(script=[RateLimitFault(), RateLimitFault(), RateLimitFault()])
    sleep, _ = _make_sleep_recorder()
    with pytest.raises(LLMCallError):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit, max_attempts=3))
    assert len(provider.calls) == 3


# ---------------------------------------------------------------------------
# §1.3 熔断与降级：连续失败达到阈值后熔断打开，prod 档调用自动降级到 dev 档
# ---------------------------------------------------------------------------


def test_circuit_opens_after_threshold_and_downgrades_prod_call(monkeypatch, circuit):
    monkeypatch.setenv("LLM_MODEL_PROD", "prod-model")
    monkeypatch.setenv("LLM_MODEL_DEV", "dev-model")
    sleep, _ = _make_sleep_recorder()

    # 连续两次都是不重试类型的失败(用 AuthFault 避免重试逻辑本身也在消耗熔断
    # 计数之外还干扰调用次数断言)，达到 CircuitBreaker 默认阈值(5)之前先手动
    # 记录几次失败，验证的是"熔断打开后 adapter 的降级行为"，不是"熔断多少次
    # 打开"这个阈值本身(那属于 CircuitBreaker 自己的单测范围)。
    for _ in range(5):
        circuit.record_failure()
    assert circuit.is_open

    provider = FaultInjectingProvider(script=[ProviderResponse(text="降级响应", stop_reason="stop")])
    result = _run(
        complete(
            [{"role": "user", "content": "hi"}],
            provider=provider,
            sleep=sleep,
            circuit=circuit,
            force_prod_tier=True,
        )
    )
    assert result.tier == ModelTier.DEV  # 熔断打开时 prod 请求被降级
    assert result.fallback_triggered is True
    assert provider.calls == ["dev-model"]


def test_rate_limit_failures_eventually_open_the_circuit(circuit):
    """不手动摆弄 `CircuitBreaker` 内部状态，走真实路径:反复触发可重试故障、
    每次都被 `complete()` 的重试耗尽拖到最终失败，验证 `circuit.record_failure()`
    确实在每次真正失败时被调用，累积到阈值后熔断打开。"""
    sleep, _ = _make_sleep_recorder()
    assert not circuit.is_open
    for _ in range(5):
        provider = FaultInjectingProvider(script=[RateLimitFault(), RateLimitFault()])
        with pytest.raises(LLMCallError):
            _run(
                complete(
                    [{"role": "user", "content": "hi"}],
                    provider=provider, sleep=sleep, circuit=circuit, max_attempts=2,
                )
            )
    assert circuit.is_open


# ---------------------------------------------------------------------------
# "格式错乱的 JSON"：不是 provider 异常，是模型返回的内容本身不合法——验证
# 真实下游解析函数(backend/agents/routing.py)对这几个真实样本的处理结果，
# 不是只测这个 fixture 模块自己拼出来的字符串对不对。
# ---------------------------------------------------------------------------


def test_route_parser_handles_response_wrapped_in_code_fence():
    text = MALFORMED_JSON_RESPONSES["wrapped_in_code_fence"]
    branch, hint = _parse_route_llm_json(text)
    assert branch.value == "full_recommend"
    assert hint is None


def test_route_parser_handles_response_with_trailing_prose():
    text = MALFORMED_JSON_RESPONSES["trailing_prose"]
    branch, hint = _parse_route_llm_json(text)
    assert branch.value == "log_write"


def test_route_parser_returns_none_for_non_json_response():
    """完全不是 JSON 时必须返回 None(解析失败的正常信号)，不能抛异常——
    调用方靠这个 None 判断"LLM 兜底也没给出可用结果，退回默认分支"，异常会
    直接打断整条请求。"""
    text = MALFORMED_JSON_RESPONSES["not_json_at_all"]
    assert _parse_route_llm_json(text) is None


def test_malformed_json_response_helper_wraps_correctly():
    resp = malformed_json_response("wrapped_in_code_fence")
    assert isinstance(resp, ProviderResponse)
    assert resp.stop_reason == "stop"
    assert resp.text == MALFORMED_JSON_RESPONSES["wrapped_in_code_fence"]


def test_route_llm_call_returning_malformed_json_is_handled_end_to_end(circuit):
    """把"格式错乱的 JSON"接到真实的 `complete()` 调用链路上——模型响应
    本身没有出错(不是异常)，`complete()` 正常返回，问题出在业务代码怎么解析
    这段文本；这里验证的是"这个响应能被塞进正常调用链路，而不会在 adapter
    这一层就出问题"，具体解析行为由上面几条测试覆盖。"""
    provider = FaultInjectingProvider(script=[malformed_json_response("wrapped_in_code_fence")])
    sleep, _ = _make_sleep_recorder()
    result = _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    branch, _ = _parse_route_llm_json(result.text)
    assert branch.value == "full_recommend"
