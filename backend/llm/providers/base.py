#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provider 抽象——把"该怎么调用某一家模型服务"和 backend/llm/adapter.py 里
"重试/超时/熔断"这套可靠性逻辑分开。adapter.py 只认这个接口，不知道也不关心
背后是 OpenAI、Anthropic 还是本地 Ollama。

新增一个 provider（比如以后要接文心一言/通义千问的原生 API）只需要实现这三个
方法，不需要动 adapter.py 里的重试/熔断循环。

工具调用(tool_use)的归一化格式(backend/agents/router.py 的 Agent Loop 靠这个跑通
ARCHITECTURE §3.2)：
  - `tools` 入参(传给 `call()`)：`[{"name", "description", "input_schema"}, ...]`，
    形状直接照抄 MCP `ToolDefinition`(backend/mcp_server/registry.py)，因为这正好
    也是 Anthropic 原生 tool 定义的形状；OpenAI 需要 `{"type":"function","function":
    {...}}` 包一层，翻译工作留在 openai_compatible.py 内部，不让调用方关心。
  - `messages` 里表达"assistant 发起了工具调用"/"工具执行结果回填"这两类回合，统一用
    OpenAI 的字段命名(`tool_calls` / role="tool" + `tool_call_id`)作为归一化格式——
    调用方(router.py)只按这一种格式拼 messages，翻译成 Anthropic 的 content-block
    形状(tool_use/tool_result 块，且要求同一个 assistant 回合后的多个 tool 结果合并
    进同一条 user 消息)是 anthropic_provider.py 内部的职责，同一模式已经用在
    system 消息的抽取上，不是新引入的分层原则。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def classify_http_error(
    exc: Exception,
    extra_retryable: tuple[type[BaseException], ...] = (),
) -> str:
    """ENGINEERING §1.2: 429 / 5xx / timeout → retryable; 400 / 401 / other → not.

    Shared by the real providers and the fault-injection fixture so tests
    exercise the same classification production uses, not a look-alike copy.
    SDK-specific timeout / connection types are passed as `extra_retryable`
    (constructing those classes in tests needs a fake request object).
    Builtin `TimeoutError` is always retryable — that covers
    `tests.fixtures.fault_injection.TimeoutFault` without a second copy of
    this table.
    """
    if isinstance(exc, TimeoutError):
        return "retryable"
    if extra_retryable and isinstance(exc, extra_retryable):
        return "retryable"
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            status_int = int(status)
        except (TypeError, ValueError):
            return "non_retryable"
        return "retryable" if (status_int == 429 or 500 <= status_int < 600) else "non_retryable"
    return "non_retryable"


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token counts from a provider response.

    Missing usage (Ollama sometimes, stub providers in tests) stays None on
    `ProviderResponse` rather than zeros — zeros would look like a real empty
    call in Langfuse cost rollups.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def as_details(self) -> dict[str, int]:
        total = self.total_tokens or (self.input_tokens + self.output_tokens)
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "total": total,
        }


@dataclass
class ToolCall:
    """一次归一化的工具调用请求(不同厂商字段名不同，这里统一)。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    text: str
    # 归一化后的停止原因，不同厂商叫法不一样，这里统一成：
    #   "stop"（正常结束）/ "max_tokens"（被长度截断）
    #   / "content_filter"（内容策略拒绝，ENGINEERING §1.2 明确"不重试"的那一类）
    #   / "tool_use"（模型请求调用一个或多个工具，ARCHITECTURE §3.2 步骤 1）
    #   / "other"（其余未归类情形）
    stop_reason: str
    raw: Any = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None


class Provider(Protocol):
    async def call(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse: ...

    def classify_error(self, exc: Exception) -> str:
        """归类成 "retryable" / "non_retryable"，见 docs/ENGINEERING.md §1.2 的表。
        每家 SDK 的异常类型不一样，分类逻辑必须留在各自 provider 里，
        不能在 adapter.py 里写一堆 isinstance(exc, openai.XXX)。"""
        ...

    async def aclose(self) -> None: ...
