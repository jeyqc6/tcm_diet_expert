#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record/Replay provider——docs/ENGINEERING.md §7.2、docs/BUILD_PLAN.md 阶段6
"Record/Replay fixture"这一项的实现。

目的：集成测试里"真实跑一次录制,CI 回放时离线、零成本"。这是一个 Provider
(和 anthropic_provider.py/openai_compatible.py 同一协议,见 base.py)，不是新的
调用路径——测试代码通过 `backend/llm/adapter.py` `complete()` 已有的 `provider=`
注入点传进去，业务代码(router.py/reconciliation.py/两个 subagent 等)一行都不用改。

两种模式：
  - record：真实打一次网络请求(需要包一层真实 provider + 有效凭据)，把
    `(请求指纹, 响应)` 存成 fixture 文件。
  - replay：不打网络，按指纹在 fixture 目录里查，查不到直接报错——**这本身
    就是"prompt 被意外改动"的检测器**(ENGINEERING §7.2 原文)：如果业务代码
    改了 system prompt 或 messages 结构，指纹会变，旧 fixture 对不上，测试
    立刻失败提示"需要重新录制"，而不是默默返回一个不再对应真实调用的响应。

指纹只由请求内容决定(model/messages/tools/其余 kwargs 的规范化 JSON 摘要)，
不含 caller 标签——caller 只用来给 fixture 文件命名，方便人在
`tests/fixtures/llm_replay/` 目录里一眼看出这是哪个调用点录的，不参与查找逻辑
本身(同一个 caller 标签下,不同请求内容天然落到不同文件)。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.exceptions import DietExpertError
from backend.llm.providers.base import Provider, ProviderResponse, ToolCall, TokenUsage

DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "llm_replay"

_CALLER_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"


class ReplayFixtureMissing(DietExpertError):
    """replay 模式下指纹查不到对应 fixture——要么从没录过，要么请求内容(prompt/
    messages/tools)变了导致指纹对不上。两种情况处理方式一样：用
    `LLM_REPLAY_MODE=record`(+ 真实凭据)重新跑一次录制，把新 fixture 提交。"""

    http_status = 500
    error_type = "replay_fixture_missing"


def _sanitize_caller(caller: str) -> str:
    """caller 是自由文本(测试作者随手起的标签)，落进文件名前做一次保守清理——
    避免路径穿越或者把奇怪字符写进文件系统。"""
    cleaned = "".join(ch if ch in _CALLER_CHARS else "_" for ch in caller)
    return cleaned or "unlabeled"


def compute_fingerprint(model: str, messages: list[dict], tools: list[dict] | None = None, **extra: Any) -> str:
    """请求内容的确定性指纹——只由"这次调用实际会发送什么"决定，不含时间戳/
    caller 这类和"是不是同一次请求"无关的信息。`sort_keys=True` 保证字段顺序
    不同但内容相同的调用落到同一个指纹上。"""
    canonical = json.dumps(
        {"model": model, "messages": messages, "tools": tools, "extra": extra},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _tool_calls_to_json(tool_calls: list[ToolCall] | None) -> list[dict] | None:
    if tool_calls is None:
        return None
    return [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]


def _tool_calls_from_json(data: list[dict] | None) -> list[ToolCall] | None:
    if data is None:
        return None
    return [ToolCall(id=d["id"], name=d["name"], arguments=d["arguments"]) for d in data]


def _usage_to_json(usage: TokenUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _usage_from_json(data: dict[str, int] | None) -> TokenUsage | None:
    if data is None:
        return None
    return TokenUsage(**data)


@dataclass
class ReplayProvider:
    """`mode="record"` 时必须传 `inner`(真实 provider 实例，负责真的打网络)；
    `mode="replay"` 时不需要 `inner`——这正是"CI 零成本"的关键，回放路径完全
    不构造任何需要凭据的真实 provider。"""

    mode: str
    caller: str
    inner: Provider | None = None
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR

    def __post_init__(self) -> None:
        if self.mode not in ("record", "replay"):
            raise ValueError(f"mode 必须是 'record' 或 'replay'，收到 {self.mode!r}")
        if self.mode == "record" and self.inner is None:
            raise ValueError("record 模式必须传 inner=真实 provider 实例，否则没有东西可录")
        self._safe_caller = _sanitize_caller(self.caller)

    def _fixture_path(self, fingerprint: str) -> Path:
        return self.fixtures_dir / f"{self._safe_caller}__{fingerprint}.json"

    async def call(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        fingerprint = compute_fingerprint(model, messages, tools, **kwargs)
        path = self._fixture_path(fingerprint)

        if self.mode == "replay":
            if not path.exists():
                raise ReplayFixtureMissing(
                    f"caller={self.caller!r} 指纹={fingerprint} 没有对应的 fixture"
                    f"({path})——要么从没录过，要么这次请求的内容(prompt/messages/tools)"
                    f"和录制时不一样了。用 LLM_REPLAY_MODE=record 重新跑一次录制。"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            resp = data["response"]
            return ProviderResponse(
                text=resp["text"],
                stop_reason=resp["stop_reason"],
                raw=None,
                tool_calls=_tool_calls_from_json(resp.get("tool_calls")),
                usage=_usage_from_json(resp.get("usage")),
            )

        # record 模式：真的打一次网络请求，然后把结果连同请求一起存下来。
        assert self.inner is not None  # __post_init__ 已经保证，这里只是给类型检查器一个交代
        resp = await self.inner.call(messages, model=model, tools=tools, **kwargs)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "caller": self.caller,
                    "fingerprint": fingerprint,
                    "request": {"model": model, "messages": messages, "tools": tools, **kwargs},
                    "response": {
                        "text": resp.text,
                        "stop_reason": resp.stop_reason,
                        "tool_calls": _tool_calls_to_json(resp.tool_calls),
                        "usage": _usage_to_json(resp.usage),
                    },
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return resp

    def classify_error(self, exc: Exception) -> str:
        if self.inner is not None:
            return self.inner.classify_error(exc)
        # replay 模式下正常路径不会抛 provider 异常(fixture 缺失走
        # ReplayFixtureMissing，不是"网络错误"语义)，这里只是满足协议。
        return "non_retryable"

    async def aclose(self) -> None:
        if self.inner is not None:
            await self.inner.aclose()


def replay_provider_for(
    caller: str,
    *,
    real_provider_name: str = "anthropic",
    timeout_s: float = 20.0,
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
) -> ReplayProvider:
    """测试代码的标准入口——同一份测试代码在两种模式下都能跑，靠
    `LLM_REPLAY_MODE` 环境变量切换，默认 "replay"(CI/日常跑测试不需要任何
    凭据)。要重新录制时本地设 `LLM_REPLAY_MODE=record` + 对应 provider 的真实
    API key 再跑一次这条测试，生成的 fixture 提交进仓库。"""
    mode = os.environ.get("LLM_REPLAY_MODE", "replay").strip().lower()
    inner: Provider | None = None
    if mode == "record":
        from backend.llm.providers.registry import build_provider

        inner = build_provider(real_provider_name, timeout_s=timeout_s)
    return ReplayProvider(mode=mode, caller=caller, inner=inner, fixtures_dir=fixtures_dir)
