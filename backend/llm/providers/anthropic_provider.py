#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anthropic 原生 Messages API——和 OpenAI 的 chat completions 格式不兼容，
不能简单换个 base_url 了事，需要单独翻译请求/响应形状：
  - system prompt 是顶层独立参数，不是 messages 列表里的一条，这里从
    messages 里把 role="system" 的条目摘出来单独传
  - max_tokens 是 Anthropic 的必填参数，OpenAI 那边可以不传，这里不传会
    直接报错，所以给了一个默认值(DEFAULT_MAX_TOKENS)
  - 响应内容是 content blocks 列表（可能混杂 text/tool_use 等类型），
    不是 OpenAI 那种单一的 choices[0].message.content 字符串
  - 停止原因叫 stop_reason 不是 finish_reason，且**没有**和 OpenAI
    content_filter 完全对应的值——Anthropic 目前不会把"内容策略拒绝"作为
    一个独立可归类的 stop_reason 暴露出来（拒绝通常表现为模型自己在正文里
    婉拒，走的还是正常的 end_turn），这里如实标注这个不对等，不是漏做。
    如果后续需要检测"模型是不是拒答了"，那是核查 pass 的语义判断范畴
    (backend/agents/verification.py)，不是这里能靠 stop_reason 字段判断的事。
  - 工具调用(tool_use)：`tools` 入参形状(`{"name","description","input_schema"}`)
    和 Anthropic 原生格式完全一致，不用翻译；但 messages 里"assistant 发起调用"/
    "工具结果回填"两类回合用的是归一化(OpenAI 风格)字段名(见 base.py 模块文档)，
    这里要翻译成 Anthropic 的 content-block 形状——tool_use 块进 assistant 消息，
    tool_result 块要求包进一条 **user** 消息（不是独立角色），且同一个 assistant
    回合后连续出现的多条 role="tool" 归一化消息要合并进同一条 user 消息，
    这是 Anthropic API 的硬性要求，不是本文件自己加的规则。
"""
from __future__ import annotations

from typing import Any

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    anthropic = None
    AsyncAnthropic = None

from backend.llm.providers.base import ProviderResponse, TokenUsage, ToolCall, classify_http_error


def _usage_from_anthropic(resp: Any) -> TokenUsage | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    if prompt == 0 and completion == 0:
        return None
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=prompt + completion,
    )

DEFAULT_MAX_TOKENS = 1024
# 2026-09-01 真实跑通时发现：DeepSeek(走这个 provider 的 Anthropic 兼容协议)
# 是带内部推理(thinking)的模型，推理过程本身也计入 max_tokens——预算不够时
# 会出现"token 真的被消耗了，但 content 里一个 type=='text' 的块都没来得及
# 吐出来"，`text` 抽出来是空字符串，下游(比如 MQE 的 `json.loads(text)`)
# 直接炸掉，日志上看是"花了 1000+ token，正文 0 字符"。真的 Anthropic API
# 目前没观察到这个问题(1024 一直够用)，这里保留 1024 不动；DeepSeek 单独
# 给一个大得多的默认值，见 `registry.py` 的 `deepseek` 分支。

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
}


def _translate_messages(messages: list[dict]) -> tuple[list[str], list[dict]]:
    """归一化 messages -> (system 文本片段, Anthropic content-block 消息列表)。"""
    system_parts: list[str] = []
    out: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role == "system":
            system_parts.append(m["content"])
            i += 1
        elif role == "tool":
            blocks = []
            while i < n and messages[i].get("role") == "tool":
                tm = messages[i]
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tm["tool_call_id"],
                        "content": tm.get("content") or "",
                    }
                )
                i += 1
            out.append({"role": "user", "content": blocks})
        elif role == "assistant" and m.get("tool_calls"):
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for call in m["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call.get("arguments") or {},
                    }
                )
            out.append({"role": "assistant", "content": blocks})
            i += 1
        else:
            out.append({"role": role, "content": m["content"]})
            i += 1
    return system_parts, out


def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    # 形状已经是 Anthropic 原生格式({"name","description","input_schema"})，
    # 原样传即可——见 base.py 模块文档为什么两边天然一致。
    return list(tools)


class AnthropicProvider:
    def __init__(
        self,
        api_key: str | None,
        timeout_s: float,
        base_url: str | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if AsyncAnthropic is None:
            raise RuntimeError("需要 anthropic 包：pip install anthropic")
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_s,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**client_kwargs)
        self._default_max_tokens = default_max_tokens

    async def call(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        system_parts, chat_messages = _translate_messages(messages)
        kwargs.setdefault("max_tokens", self._default_max_tokens)
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        translated_tools = _translate_tools(tools)
        if translated_tools:
            kwargs["tools"] = translated_tools

        resp = await self._client.messages.create(model=model, messages=chat_messages, **kwargs)

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        tool_calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
            for block in resp.content
            if getattr(block, "type", None) == "tool_use"
        ] or None
        stop_reason = _STOP_REASON_MAP.get(resp.stop_reason, "other")
        return ProviderResponse(
            text=text,
            stop_reason=stop_reason,
            raw=resp,
            tool_calls=tool_calls,
            usage=_usage_from_anthropic(resp),
        )

    def classify_error(self, exc: Exception) -> str:
        extra: tuple[type, ...] = ()
        if anthropic is not None:
            extra = (anthropic.APITimeoutError, anthropic.APIConnectionError)
        return classify_http_error(exc, extra_retryable=extra)

    async def aclose(self) -> None:
        await self._client.close()
