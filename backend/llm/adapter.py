#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型调用统一 adapter：超时分层 / 重试+退避+jitter / 熔断器 / 双档模型切换 /
多 provider(OpenAI · Anthropic · 本地 Ollama · OpenRouter)。可靠性逻辑全部收敛在这一层，
业务代码(路由/SubAgent/调和层/核查pass)只调用 `complete()`，不感知具体
模型名、不感知具体是哪家服务商，也不自己实现重试。

设计依据：docs/ARCHITECTURE.md §7
决策依据：docs/DECISIONS.md D19(双档模型策略，调和层不参与降级)、D29(多 provider 抽象)
可靠性规格：docs/ENGINEERING.md §1
具体调用某一家服务商的代码在 backend/llm/providers/ 下，本文件不知道也不
关心背后是谁——只认 backend/llm/providers/base.py 定义的 Provider 协议。

分层职责边界，写清楚避免以后有人指望这一层做不该它做的事：
  - 单次 LLM 调用超时(20s)：本模块负责，靠各 provider 内部 client 的
    per-request timeout
  - 单个 SubAgent 超时(45s)、整链路超时(90s)：不是本模块的职责，那是
    "多次 adapter 调用叠加起来"的编排层概念（`backend/agents/timeouts.py`，
    挂在 `run_subagent` / `_stream_chat`）
  - 熔断只覆盖"主力模型"这一个依赖；Open-Meteo、向量检索各自的熔断器
    在各自的工具实现里按同样的 CircuitBreaker 模式建，不复用同一个实例

MODEL_TIER / provider 语义(D19 + D29)：
  - "dev"（默认）：开发迭代期低成本档，可以指向本地 Ollama(免费)
  - "prod"：正式跑分/交付档
  - dev/prod 两档可以是**不同的服务商**，不只是同一家的不同模型——比如
    dev 用本地 Ollama 免费迭代，prod 用 Anthropic/OpenAI 交付；这是本次
    改动前(只支持单一 OpenAI 兼容 provider)做不到的事
  - `force_prod_tier` 参数本身仍在,留作业务代码需要"这一次调用必须用交付档,
    不受 MODEL_TIER 影响"时的显式声明;但 D19 原本"调和层永远传 True"这条例外
    已在 2026-08-27 撤销(见 D19 决策修订记录)——调和层现在和其余调用一样跟随
    MODEL_TIER,目前没有任何调用点传 True
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from backend.env import load_env
from backend.exceptions import LLMCallError, NonRetryableError
from backend.llm.providers.base import TokenUsage, ToolCall
from backend.llm.providers.registry import build_provider
from backend.observability.cost import estimate_cost_usd
from backend.observability.redact import redact_messages
from backend.observability.tracing import observation, stage_log, update_current

logger = logging.getLogger("diet_expert.llm_adapter")

# ---- ENGINEERING §1.1 超时分层：本文件只管"单次 LLM 调用"这一层 ----
SINGLE_CALL_TIMEOUT_S = 20.0

# ---- ENGINEERING §1.2 重试与退避：指数退避 1s→2s→4s，最多 3 次，必须带 jitter ----
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0
JITTER_MAX_S = 0.5

# ---- ENGINEERING §1.3 熔断：主力模型连续 5 次失败 → 打开；30s 后半开探测 ----
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_HALF_OPEN_AFTER_S = 30.0

# provider 在各自 tier 上没有显式配置 LLM_MODEL_DEV/PROD 时的兜底默认值。
# ⚠️ ollama 没有默认值——本地装了哪个模型因机器而异，猜一个大概率是错的，
# 宁可显式报错要求用户设置，也不要悄悄用一个可能没被 `ollama pull` 过的名字。
# ⚠️ openrouter 同样没有默认值——它本身是几十家上游模型的路由层，模型名是
# "vendor/model"这种 slug(比如 "anthropic/claude-haiku-4.5")且随目录变化，
# 挑一个当默认既不代表"最合适"也不保证一直有效，同样必须显式配置。
_DEFAULT_MODELS: dict[tuple[str, "ModelTier"], str] = {}


class ModelTier(str, Enum):
    DEV = "dev"
    PROD = "prod"


_DEFAULT_MODELS.update(
    {
        ("openai", ModelTier.DEV): "gpt-4o-mini",
        ("openai", ModelTier.PROD): "gpt-4o",
        ("anthropic", ModelTier.DEV): "claude-haiku-4-5-20251001",
        ("anthropic", ModelTier.PROD): "claude-sonnet-5",
        ("deepseek", ModelTier.DEV): "deepseek-v4-flash",
        ("deepseek", ModelTier.PROD): "deepseek-v4-flash",
    }
)


def _get_tier() -> ModelTier:
    load_env()
    raw = os.environ.get("MODEL_TIER", "dev").strip().lower()
    return ModelTier.PROD if raw == "prod" else ModelTier.DEV


def _provider_name_for_tier(tier: ModelTier) -> str:
    """dev/prod 两档可以指向不同服务商(D29)——比如 LLM_PROVIDER_DEV=ollama、
    LLM_PROVIDER_PROD=anthropic。没单独配就退到通用的 LLM_PROVIDER，
    再没配就默认 openai(向后兼容此前只支持 OpenAI 兼容端点的行为)。"""
    load_env()
    env_key = "LLM_PROVIDER_PROD" if tier == ModelTier.PROD else "LLM_PROVIDER_DEV"
    return os.environ.get(env_key, os.environ.get("LLM_PROVIDER", "openai")).strip().lower()


def _model_for_tier(tier: ModelTier, provider_name: str) -> str:
    load_env()
    env_key = "LLM_MODEL_PROD" if tier == ModelTier.PROD else "LLM_MODEL_DEV"
    explicit = os.environ.get(env_key)
    if explicit:
        return explicit
    if tier == ModelTier.DEV:
        # 兼容已有脚本(naive_rag.py/run_baselines.py)用的 CHAT_MODEL
        chat_model = os.environ.get("CHAT_MODEL")
        if chat_model:
            return chat_model
    default = _DEFAULT_MODELS.get((provider_name, tier))
    if default is None:
        raise RuntimeError(
            f"provider={provider_name!r} 在 {tier.value} 档没有内置默认模型"
            f"（比如本地 ollama 装的模型因机器而异，没法猜），"
            f"必须显式设置 {env_key}"
            + ("（dev 档也可以用 CHAT_MODEL）" if tier == ModelTier.DEV else "")
        )
    return default


@dataclass
class CircuitBreaker:
    """每个外部依赖一个独立实例(ENGINEERING §1.3)。本文件里的模块级单例只覆盖
    "主力模型"这一个依赖；不要把它复用给天气/向量检索——那两个的失败模式和
    恢复条件都不一样，混在一起会导致"天气挂了却把模型也熔断了"这种误联动。
    """

    failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD
    half_open_after_s: float = CIRCUIT_HALF_OPEN_AFTER_S
    consecutive_failures: int = field(default=0, init=False)
    opened_at: float | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        """半开：距打开超过 half_open_after_s 后，视为"不再拦截"，放一个请求探测；
        探测失败会在 record_failure() 里重新盖上打开时间戳，等于重新计时。"""
        if self.opened_at is None:
            return False
        return (time.monotonic() - self.opened_at) < self.half_open_after_s

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.opened_at = time.monotonic()
            logger.warning(
                "circuit breaker OPEN：主力模型连续 %d 次失败", self.consecutive_failures
            )


_circuit = CircuitBreaker()


def get_circuit_breaker() -> CircuitBreaker:
    """暴露给 trace/日志埋点用(ENGINEERING §1.3:"熔断状态必须进 trace")，
    以及测试用来重置/检查状态。"""
    return _circuit


# provider 实例按 (名字, 所在 event loop) 缓存、跨调用复用(避免每次调用都
# 重建 HTTP client)。
#
# ⚠️ 2026-08-31 教训：这里原来只按名字缓存、生命周期等同于进程(和
# `_retrieval_common.py` BGE-M3 embedder 惰性单例同一个模式)——这对一个
# "只有主 FastAPI event loop"的进程是对的，但真实复现过一个坑：
# `backend/mcp_server/tools/_retrieval_common.py` 的 `_run_coroutine_sync()`
# 每次调用都会现造一个跑完即销毁的 event loop 去跑 MQE 的 `complete()`。
# provider 内部的异步 SDK 客户端(如 `AsyncAnthropic`，包着一个 httpx 连接池)
# 一旦在某个 loop 上真正发起过请求，就绑定在那个 loop 上了——换一个 loop
# 复用同一个客户端会导致请求静默卡死收不到响应(不是报错，是真的挂住，直到
# 上层的 45s SubAgent 超时)，`backend/agents/agent_loop.py` 把工具调用改成
# 并发执行后，这个此前很少触发的竞争窗口变成了每次都会撞上。
#
# 按 `(name, loop)` 缓存后：常驻的主 FastAPI event loop 仍然一直复用同一个
# 客户端(该有的连接池收益都在)；`_run_coroutine_sync` 现造的一次性 loop
# 各自拿到自己的客户端，不会跨 loop 复用——这些一次性 loop 本来也没有"下次
# 还能复用"的机会，不算额外浪费。
#
# 用 `id(loop)` 当 key 是错的：CPython 里被垃圾回收的对象，它的 id() 之后
# 可能被一个全新对象复用，会把"某个早就销毁的旧 loop 的客户端"错误地当成
# "当前这个新 loop 的客户端"发回去——等于把这个 bug 原样绕了个圈子重新引入。
# 用 `weakref.WeakKeyDictionary` 直接以 loop 对象本身(不是它的 id)做 key：
# loop 被销毁时对应缓存条目跟着自动清掉，不会有"新 loop 撞上旧 loop 缓存"
# 这种事，也不会因为一直攒着一次性 loop 的引用而内存泄漏。
_PROVIDERS: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = weakref.WeakKeyDictionary()


def _get_provider(name: str) -> Any:
    loop = asyncio.get_running_loop()
    per_loop = _PROVIDERS.get(loop)
    if per_loop is None:
        per_loop = {}
        _PROVIDERS[loop] = per_loop
    if name not in per_loop:
        per_loop[name] = build_provider(name, timeout_s=SINGLE_CALL_TIMEOUT_S)
    return per_loop[name]


@dataclass
class LLMResult:
    text: str
    model: str
    tier: ModelTier
    provider: str
    raw: Any = None
    # None＝模型这一轮没有请求调用任何工具；非空列表＝
    # backend/agents/router.py 的 Agent Loop 靠这个字段的有无判断是否继续
    # (ARCHITECTURE §3.2 步骤 7)，不是靠固定轮数。
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    cost_est: float | None = None
    attempts: int = 1
    fallback_triggered: bool = False
    latency_ms: float = 0.0


# 供 backend/agents/{routing,agent_loop,...}.py 和 SubAgent/调和层/核查pass 共用的
# "LLM 调用函数"签名——原来定义在 backend/agents/router.py 里(Agent Loop 最初写
# 在那个文件，顺手就近定义)，这里才是它真正的归属：类型描述的是 `complete()`
# 本身的签名，和路由判断/Agent Loop 都无关，只是恰好被两者都用到。
CompleteFn = Callable[..., Awaitable["LLMResult"]]


async def complete(
    messages: list[dict],
    *,
    force_prod_tier: bool = False,
    max_attempts: int = MAX_ATTEMPTS,
    provider: Any = None,
    circuit: CircuitBreaker | None = None,
    sleep=None,
    **create_kwargs,
) -> LLMResult:
    """统一的模型调用入口。业务代码只传 messages，不感知具体模型名/服务商(D19/D29)。

    force_prod_tier=True：显式声明"这次调用必须用交付档模型，不受 MODEL_TIER
    环境变量影响"——不是 adapter 自己猜"这是不是该用好模型的调用"，是调用方
    主动声明。D19 原本要求调和层调用永远传 True，这条例外已在 2026-08-27 撤销
    (见 D19 决策修订记录)，目前没有调用点使用这个参数，留作以后确有需要时用。

    `provider`/`circuit`/`sleep` 参数是为了单测能注入假实现，不需要真的打网络
    请求(tests/unit/llm/test_adapter.py)；业务代码调用时不传，用真实默认值。
    """
    load_env()
    circuit = circuit or _circuit
    sleep = sleep or _default_sleep
    t0 = time.perf_counter()

    requested_tier = ModelTier.PROD if force_prod_tier else _get_tier()
    tier = requested_tier
    fallback_triggered = False

    # 熔断打开时切备用档(ENGINEERING §1.3:"打开后行为=切备用模型档")。
    # 这条对 force_prod_tier 调用同样生效——服务真的挂了不该硬撑着用同一个
    # 打不通的模型，那样只会白白多等一轮超时，对可用性没有任何好处。
    circuit_open = circuit.is_open
    if circuit_open and tier == ModelTier.PROD:
        logger.warning("熔断开启，本次调用从 prod 档降级为 dev 档兜底")
        tier = ModelTier.DEV
        fallback_triggered = True

    provider_name = _provider_name_for_tier(tier)
    active_provider = provider or _get_provider(provider_name)
    model = _model_for_tier(tier, provider_name)

    gen_meta = {
        "provider": provider_name,
        "tier": tier.value,
        "requested_tier": requested_tier.value,
        "force_prod_tier": force_prod_tier,
        "circuit_open": circuit_open,
        "fallback_triggered": fallback_triggered,
        "has_tools": "tools" in create_kwargs and bool(create_kwargs.get("tools")),
    }

    with observation(
        "llm.complete",
        as_type="generation",
        model=model,
        input=redact_messages(messages),
        metadata=gen_meta,
    ):
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await active_provider.call(messages, model=model, **create_kwargs)
            except Exception as exc:  # noqa: BLE001 — 分类交给各 provider 的 classify_error
                category = active_provider.classify_error(exc)
                circuit.record_failure()
                if category == "non_retryable":
                    _record_generation_error(
                        t0, model, provider_name, attempt, fallback_triggered, exc
                    )
                    raise NonRetryableError(str(exc)) from exc
                last_exc = exc
                if attempt < max_attempts:
                    delay = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, JITTER_MAX_S)
                    logger.info(
                        "LLM 调用失败(第 %d/%d 次，provider=%s)，%.2fs 后重试：%s",
                        attempt, max_attempts, provider_name, delay, exc,
                    )
                    await sleep(delay)
                    continue
                _record_generation_error(
                    t0, model, provider_name, attempt, fallback_triggered, exc
                )
                raise LLMCallError(f"重试 {max_attempts} 次后仍失败: {exc}") from exc
            else:
                if resp.stop_reason == "content_filter":
                    # ENGINEERING §1.2："内容策略拒绝 → 不重试，走 fallback"
                    # Policy rejection is not an infrastructure outage — do not
                    # trip the circuit breaker.
                    _record_generation_error(
                        t0, model, provider_name, attempt, fallback_triggered,
                        NonRetryableError("模型内容策略拒绝(content_filter)"),
                    )
                    raise NonRetryableError("模型内容策略拒绝(content_filter)")
                circuit.record_success()
                usage = resp.usage
                cost_est = estimate_cost_usd(model, usage, provider=provider_name)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                tool_names = [c.name for c in (resp.tool_calls or [])]
                update_current(
                    as_type="generation",
                    output={
                        "text": redact_messages(
                            [{"role": "assistant", "content": resp.text}]
                        )[0]["content"],
                        "stop_reason": resp.stop_reason,
                        "tool_call_names": tool_names or None,
                    },
                    usage_details=usage.as_details() if usage else None,
                    cost_details={"total": cost_est} if cost_est is not None else None,
                    metadata={
                        **gen_meta,
                        "attempts": attempt,
                        "latency_ms": round(latency_ms, 1),
                        "tokens": usage.total_tokens if usage else None,
                        "cost_est": cost_est,
                    },
                )
                stage_log(
                    logger,
                    "llm",
                    latency_ms=round(latency_ms, 1),
                    tokens=usage.total_tokens if usage else None,
                    cost_est=cost_est,
                    fallback_triggered=fallback_triggered,
                    model=model,
                    provider=provider_name,
                    attempts=attempt,
                    # `text` 走 stage_log 内部的 redact_log_payload，跟 Langfuse 那份
                    # 一样受 LANGFUSE_CAPTURE_IO 控制——之前只有 update_current() 把
                    # 回复内容送去了 Langfuse，本地进程日志里完全看不到 LLM 说了什么。
                    text=resp.text,
                    tool_call_names=tool_names or None,
                )
                return LLMResult(
                    text=resp.text,
                    model=model,
                    tier=tier,
                    provider=provider_name,
                    raw=resp.raw,
                    tool_calls=resp.tool_calls,
                    usage=usage,
                    cost_est=cost_est,
                    attempts=attempt,
                    fallback_triggered=fallback_triggered,
                    latency_ms=latency_ms,
                )
        _record_generation_error(
            t0, model, provider_name, max_attempts, fallback_triggered, last_exc
        )
        raise LLMCallError(f"重试 {max_attempts} 次后仍失败: {last_exc}")


def _record_generation_error(
    t0: float,
    model: str,
    provider_name: str,
    attempts: int,
    fallback_triggered: bool,
    exc: BaseException | None,
) -> None:
    latency_ms = (time.perf_counter() - t0) * 1000.0
    update_current(
        as_type="generation",
        level="ERROR",
        status_message=str(exc) if exc else "LLM call failed",
        metadata={
            "attempts": attempts,
            "latency_ms": round(latency_ms, 1),
            "fallback_triggered": fallback_triggered,
            "model": model,
            "provider": provider_name,
        },
    )
    stage_log(
        logger,
        "llm",
        latency_ms=round(latency_ms, 1),
        fallback_triggered=fallback_triggered,
        model=model,
        provider=provider_name,
        attempts=attempts,
        error=str(exc) if exc else None,
    )


async def _default_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
