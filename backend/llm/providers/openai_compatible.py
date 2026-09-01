#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
覆盖 OpenAI 本身、Ollama、以及任何声称"OpenAI 兼容"的服务商——三者用的是
同一套 chat completions 请求/响应格式，区别只在 base_url 和要不要真实 api_key。

Ollama 单独说一句：它在本机跑一个 HTTP 服务（默认 http://localhost:11434），
自带一个 OpenAI 兼容层，挂在 /v1 路径下——所以不需要给 Ollama 单独写一个
provider 实现，复用这个类、把 base_url 指过去就行；api_key 随便填一个非空
字符串即可，Ollama 不校验它，只是 OpenAI SDK 要求这个参数不能是 None。

工具调用(tool_use)：归一化格式(backend/llm/providers/base.py 模块文档)在
messages/tools 两处都需要翻译成 OpenAI 的 wire 格式——`tools` 从
`{"name","description","input_schema"}` 包成 `{"type":"function","function":
{...,"parameters":input_schema}}`；assistant 的 `tool_calls` 字段里
`arguments` 要序列化成 JSON 字符串（OpenAI 要求，不接受 dict）；工具结果的
role="tool" 消息只保留 OpenAI 认的 `tool_call_id`/`content` 两个字段。
"""
from __future__ import annotations

import json
from typing import Any

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    openai = None
    AsyncOpenAI = None

from backend.llm.providers.base import ProviderResponse, TokenUsage, ToolCall, classify_http_error


def _usage_from_openai(resp: Any) -> TokenUsage | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
    completion = int(
        getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
    )
    total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
    if prompt == 0 and completion == 0 and total == 0:
        return None
    return TokenUsage(input_tokens=prompt, output_tokens=completion, total_tokens=total)

_FINISH_REASON_MAP = {
    "content_filter": "content_filter",
    "length": "max_tokens",
    "stop": "stop",
    "tool_calls": "tool_use",
}


def _translate_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content"),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                            },
                        }
                        for call in m["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m.get("content") or "",
                }
            )
        else:
            out.append(m)
    return out


def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class OpenAICompatibleProvider:
    def __init__(self, api_key: str | None, base_url: str, timeout_s: float):
        if AsyncOpenAI is None:
            raise RuntimeError("需要 openai 包：pip install openai")
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed", base_url=base_url, timeout=timeout_s
        )

    async def call(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        translated_tools = _translate_tools(tools)
        if translated_tools:
            kwargs["tools"] = translated_tools
        resp = await self._client.chat.completions.create(
            model=model, messages=_translate_messages(messages), **kwargs
        )
        choice = resp.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        stop_reason = _FINISH_REASON_MAP.get(finish_reason, "other")
        raw_tool_calls = getattr(choice.message, "tool_calls", None) or []
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
            )
            for tc in raw_tool_calls
        ] or None
        return ProviderResponse(
            text=choice.message.content or "",
            stop_reason=stop_reason,
            raw=resp,
            tool_calls=tool_calls,
            usage=_usage_from_openai(resp),
        )

    def classify_error(self, exc: Exception) -> str:
        extra: tuple[type, ...] = ()
        if openai is not None:
            extra = (openai.APITimeoutError, openai.APIConnectionError)
        return classify_http_error(exc, extra_retryable=extra)

    async def aclose(self) -> None:
        await self._client.close()
