"""
测试目标：backend/llm/adapter.py —— 超时分层/重试+退避+jitter/熔断器/双档切换/
多 provider 选择。不打真实网络请求：注入假 Provider(实现 backend/llm/providers/base.py
的协议)/sleep 验证逻辑，对应 docs/ENGINEERING.md §6.2 的 record/replay 测试哲学。
对应实现：backend/llm/adapter.py、backend/llm/providers/*.py

不用 pytest-asyncio(项目没装这个依赖)，测试函数本身是同步的，
内部用 asyncio.run() 跑被测的 async 代码——足够用，不需要额外依赖。
"""
import asyncio

import pytest

from backend.llm.adapter import (
    CircuitBreaker,
    LLMCallError,
    ModelTier,
    NonRetryableError,
    _get_tier,
    _model_for_tier,
    _provider_name_for_tier,
    complete,
)
from backend.llm.providers.base import ProviderResponse


class _FakeStatusError(Exception):
    """不需要精确构造某家 SDK 的具体异常子类——分类逻辑只看 status_code 属性，
    任何带这个属性的异常都能拿来测。"""

    def __init__(self, status_code):
        super().__init__(f"fake status {status_code}")
        self.status_code = status_code


class _FakeProvider:
    """实现 backend/llm/providers/base.py 的 Provider 协议，脚本化返回值/异常。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []  # 记录每次调用实际用了哪个 model
        self.closed = False

    async def call(self, messages, *, model, **kwargs):
        self.calls.append(model)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def classify_error(self, exc):
        status = getattr(exc, "status_code", None)
        if status is not None and (status == 429 or 500 <= status < 600):
            return "retryable"
        return "non_retryable"

    async def aclose(self):
        self.closed = True


def _make_sleep_recorder():
    recorded = []

    async def _sleep(seconds):
        recorded.append(seconds)

    return _sleep, recorded


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def circuit():
    """每个测试给一个全新的 CircuitBreaker，不共用 adapter.py 里的模块级单例——
    那个单例是给真实运行时用的（状态本来就该跨调用持久），但测试之间必须隔离，
    否则前一个测试留下的失败计数会泄漏进下一个测试。"""
    return CircuitBreaker()


# ---------- 基本成功路径 ----------

def test_success_first_try(circuit):
    provider = _FakeProvider([ProviderResponse(text="你好", stop_reason="stop")])
    sleep, recorded = _make_sleep_recorder()
    result = _run(
        complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit)
    )
    assert result.text == "你好"
    assert len(provider.calls) == 1
    assert recorded == []


# ---------- 重试与退避(ENGINEERING §1.2) ----------

def test_retries_on_retryable_error_then_succeeds(circuit):
    provider = _FakeProvider(
        [_FakeStatusError(429), ProviderResponse(text="恢复了", stop_reason="stop")]
    )
    sleep, recorded = _make_sleep_recorder()
    result = _run(
        complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit)
    )
    assert result.text == "恢复了"
    assert len(provider.calls) == 2
    assert len(recorded) == 1  # 两次调用之间只睡了一次


def test_backoff_is_exponential_with_jitter(circuit):
    # 3 次全部 500，验证退避序列大致是 1s / 2s 量级(+jitter),不是固定值
    provider = _FakeProvider([_FakeStatusError(500)] * 3)
    sleep, recorded = _make_sleep_recorder()
    with pytest.raises(LLMCallError):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert len(recorded) == 2  # 3 次尝试之间睡了 2 次
    assert 1.0 <= recorded[0] < 1.5   # BACKOFF_BASE_S * 2^0 + jitter[0, 0.5)
    assert 2.0 <= recorded[1] < 2.5   # BACKOFF_BASE_S * 2^1 + jitter[0, 0.5)


def test_exhausts_retries_raises_llm_call_error(circuit):
    provider = _FakeProvider([_FakeStatusError(500)] * 3)
    sleep, _ = _make_sleep_recorder()
    with pytest.raises(LLMCallError):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert len(provider.calls) == 3


# ---------- 不可重试类别(ENGINEERING §1.2) ----------

def test_400_is_non_retryable(circuit):
    provider = _FakeProvider([_FakeStatusError(400)])
    sleep, recorded = _make_sleep_recorder()
    with pytest.raises(NonRetryableError):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert len(provider.calls) == 1  # 立即失败，不重试
    assert recorded == []


def test_401_is_non_retryable(circuit):
    provider = _FakeProvider([_FakeStatusError(401)])
    sleep, _ = _make_sleep_recorder()
    with pytest.raises(NonRetryableError):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert len(provider.calls) == 1


def test_content_filter_is_non_retryable(circuit):
    provider = _FakeProvider([ProviderResponse(text="(被拦截)", stop_reason="content_filter")])
    sleep, recorded = _make_sleep_recorder()
    with pytest.raises(NonRetryableError):
        _run(complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit))
    assert recorded == []


# ---------- 双档模型切换(D19) ----------

def test_get_tier_defaults_to_dev(monkeypatch):
    monkeypatch.delenv("MODEL_TIER", raising=False)
    assert _get_tier() == ModelTier.DEV


def test_get_tier_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_TIER", "prod")
    assert _get_tier() == ModelTier.PROD


def test_model_for_tier_dev_falls_back_to_chat_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_DEV", raising=False)
    monkeypatch.setenv("CHAT_MODEL", "some-dev-model")
    assert _model_for_tier(ModelTier.DEV, "openai") == "some-dev-model"


def test_model_for_tier_ollama_without_explicit_model_raises(monkeypatch):
    # ollama 没有内置默认模型——本地装了哪个因机器而异，不能瞎猜
    monkeypatch.delenv("LLM_MODEL_DEV", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="没有内置默认模型"):
        _model_for_tier(ModelTier.DEV, "ollama")


def test_force_prod_tier_ignores_model_tier_env(monkeypatch, circuit):
    # 显式传 force_prod_tier=True 时不受 MODEL_TIER=dev 影响(通用能力，不再
    # 专属调和层——D19 原本的"调和层永远传 True"例外已撤销，见决策修订记录)
    monkeypatch.setenv("MODEL_TIER", "dev")
    monkeypatch.setenv("LLM_MODEL_PROD", "prod-model")
    provider = _FakeProvider([ProviderResponse(text="ok", stop_reason="stop")])
    sleep, _ = _make_sleep_recorder()
    result = _run(
        complete(
            [{"role": "user", "content": "hi"}],
            provider=provider, sleep=sleep, force_prod_tier=True, circuit=circuit,
        )
    )
    assert result.tier == ModelTier.PROD
    assert provider.calls == ["prod-model"]


# ---------- 多 provider 选择(D29) ----------

def test_provider_name_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_DEV", raising=False)
    assert _provider_name_for_tier(ModelTier.DEV) == "openai"


def test_provider_name_per_tier_can_differ(monkeypatch):
    # dev 用本地 ollama 免费迭代，prod 用 anthropic 交付——这正是要支持的场景
    monkeypatch.setenv("LLM_PROVIDER_DEV", "ollama")
    monkeypatch.setenv("LLM_PROVIDER_PROD", "anthropic")
    assert _provider_name_for_tier(ModelTier.DEV) == "ollama"
    assert _provider_name_for_tier(ModelTier.PROD) == "anthropic"


def test_provider_result_records_which_provider_was_used(monkeypatch, circuit):
    # 同上：显式钉住 dev 档，不依赖真实 .env 当前恰好是哪一档。
    monkeypatch.setenv("MODEL_TIER", "dev")
    monkeypatch.setenv("LLM_PROVIDER_DEV", "ollama")
    monkeypatch.setenv("LLM_MODEL_DEV", "qwen3:0.6b")  # ollama 没有内置默认模型，必须显式给
    provider = _FakeProvider([ProviderResponse(text="来自本地模型", stop_reason="stop")])
    sleep, _ = _make_sleep_recorder()
    result = _run(
        complete([{"role": "user", "content": "hi"}], provider=provider, sleep=sleep, circuit=circuit)
    )
    assert result.provider == "ollama"


# ---------- 熔断器(ENGINEERING §1.3) ----------

def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, half_open_after_s=30)
    assert cb.is_open is False
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False  # 还没到阈值
    cb.record_failure()
    assert cb.is_open is True  # 第 3 次触发打开


def test_circuit_breaker_closes_on_success():
    cb = CircuitBreaker(failure_threshold=2, half_open_after_s=30)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is True
    cb.record_success()
    assert cb.is_open is False
    assert cb.consecutive_failures == 0


def test_circuit_breaker_half_opens_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, half_open_after_s=0.01)
    cb.record_failure()
    assert cb.is_open is True
    import time

    time.sleep(0.02)
    assert cb.is_open is False  # 半开：允许下一次请求探测


def test_open_circuit_downgrades_prod_call_to_dev(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_PROD", "prod-model")
    monkeypatch.setenv("LLM_MODEL_DEV", "dev-model")
    cb = CircuitBreaker(failure_threshold=1, half_open_after_s=999)
    cb.record_failure()  # 打开熔断
    provider = _FakeProvider([ProviderResponse(text="降级响应", stop_reason="stop")])
    sleep, _ = _make_sleep_recorder()
    result = _run(
        complete(
            [{"role": "user", "content": "hi"}],
            provider=provider,
            sleep=sleep,
            force_prod_tier=True,  # 即便显式要 prod，熔断打开时也该降级
            circuit=cb,
        )
    )
    assert result.tier == ModelTier.DEV
    assert provider.calls == ["dev-model"]
    assert result.fallback_triggered is True


def test_complete_records_generation_span(monkeypatch):
    from backend.observability.tracing import use_memory_backend

    # 显式钉住 dev 档——不能只设 LLM_MODEL_DEV 就假设当前是 dev 档，真实 .env
    # 里的 MODEL_TIER 会变(比如切到 prod 调试)，这条测试测的是 dev 档行为，
    # 不该被环境里当前是哪一档悄悄影响。
    monkeypatch.setenv("MODEL_TIER", "dev")
    monkeypatch.setenv("LLM_MODEL_DEV", "dev-model")
    backend = use_memory_backend()
    from backend.llm.providers.base import TokenUsage

    provider = _FakeProvider(
        [
            ProviderResponse(
                text="ok",
                stop_reason="stop",
                usage=TokenUsage(input_tokens=12, output_tokens=8, total_tokens=20),
            )
        ]
    )
    sleep, _ = _make_sleep_recorder()
    result = _run(
        complete(
            [{"role": "user", "content": "hi"}],
            provider=provider,
            sleep=sleep,
            circuit=CircuitBreaker(),
        )
    )
    assert result.usage is not None
    assert result.usage.total_tokens == 20
    gens = [s for s in backend.spans if s.name == "llm.complete"]
    assert len(gens) == 1
    assert gens[0].as_type == "generation"
    assert gens[0].model == "dev-model"
    assert gens[0].usage_details == {"input": 12, "output": 8, "total": 20}
    assert gens[0].metadata["fallback_triggered"] is False
    assert gens[0].metadata["attempts"] == 1
    assert "latency_ms" in gens[0].metadata


# ---------------------------------------------------------------------------
# `_get_provider` 按 (name, event loop) 缓存 —— 2026-08-31 真实撞过的坑：
# 按名字缓存、生命周期等同于进程时，`backend/mcp_server/tools/_retrieval_common.py`
# 的 `_run_coroutine_sync()` 每次调用现造一个用完即销毁的 event loop，会把
# provider 内部的异步 SDK 客户端(httpx 连接池)在互不相同的 event loop 之间
# 反复复用，导致请求静默挂死。这里不打真实网络，用假 `build_provider` 验证
# 缓存本身的键控行为对不对：同一个 loop 内复用同一个实例(保留连接池收益)，
# 不同 loop 各自拿到不同实例(不会跨 loop 复用出问题)。
# ---------------------------------------------------------------------------


def test_get_provider_reuses_same_instance_within_one_event_loop(monkeypatch):
    from backend.llm import adapter

    built = []

    def fake_build_provider(name, *, timeout_s):
        instance = object()
        built.append(instance)
        return instance

    monkeypatch.setattr(adapter, "build_provider", fake_build_provider)

    async def call_twice():
        return adapter._get_provider("fake"), adapter._get_provider("fake")

    first, second = _run(call_twice())
    assert first is second
    assert len(built) == 1


def test_get_provider_returns_different_instance_across_different_event_loops(monkeypatch):
    """回归测试：这条如果失败(两次拿到同一个实例)，说明缓存又变回"按名字、
    跨 loop 共享"了——正是 2026-08-31 那次真实导致检索请求挂死的根因。"""
    from backend.llm import adapter

    built = []

    def fake_build_provider(name, *, timeout_s):
        instance = object()
        built.append(instance)
        return instance

    monkeypatch.setattr(adapter, "build_provider", fake_build_provider)

    async def call_once():
        return adapter._get_provider("fake")

    first = asyncio.run(call_once())
    second = asyncio.run(call_once())  # 一个全新的、独立的 event loop

    assert first is not second
    assert len(built) == 2
