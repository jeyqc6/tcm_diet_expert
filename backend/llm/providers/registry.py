#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
provider 名字 -> 具体实例。env 变量的读取集中在这里，adapter.py 不直接读
OPENAI_API_KEY/ANTHROPIC_API_KEY 这些具体名字，只知道"给我 dev/prod 档
对应的 provider"——新增/换一家服务商只用改这一个文件。
"""
from __future__ import annotations

import os

from backend.env import load_env
from backend.llm.providers.anthropic_provider import AnthropicProvider
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider

SUPPORTED_PROVIDERS = ("openai", "anthropic", "deepseek", "ollama", "openrouter")

_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
# 2026-09-01：DeepSeek 是带内部推理的模型，推理本身也吃 max_tokens 预算——
# 沿用真的 Anthropic 那份 1024 默认值时，真实观察到预算被推理吃光、
# `text` 抽出来是空字符串（见 anthropic_provider.py DEFAULT_MAX_TOKENS 注释）。
# 只调大 DeepSeek 这一路的默认值，不动真的 Anthropic（1024 那边一直够用，
# 没有理由跟着一起改）。可以用 DEEPSEEK_MAX_TOKENS 覆盖，不用改代码重新实测。
_DEEPSEEK_DEFAULT_MAX_TOKENS = 8192


def _deepseek_max_tokens() -> int:
    load_env()
    raw = (os.environ.get("DEEPSEEK_MAX_TOKENS") or "").strip()
    if not raw:
        return _DEEPSEEK_DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _DEEPSEEK_DEFAULT_MAX_TOKENS
    return value if value > 0 else _DEEPSEEK_DEFAULT_MAX_TOKENS


def build_provider(name: str, *, timeout_s: float):
    load_env()
    name = name.strip().lower()
    if name == "openai":
        return OpenAICompatibleProvider(
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            timeout_s=timeout_s,
        )
    if name == "ollama":
        # 本地跑，不需要真实鉴权；OLLAMA_BASE_URL 默认对应 `ollama serve` 的默认端口。
        return OpenAICompatibleProvider(
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            timeout_s=timeout_s,
        )
    if name == "anthropic":
        return AnthropicProvider(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
            timeout_s=timeout_s,
        )
    if name == "deepseek":
        # DeepSeek exposes an Anthropic-compatible Messages API — reuse the same
        # provider implementation; only base_url / api_key / model name differ.
        return AnthropicProvider(
            api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", _DEEPSEEK_ANTHROPIC_BASE_URL),
            timeout_s=timeout_s,
            default_max_tokens=_deepseek_max_tokens(),
        )
    if name == "openrouter":
        # OpenRouter 本身就是一层 OpenAI 兼容代理(转发到几十家上游模型)，复用
        # OpenAICompatibleProvider、换 base_url/api_key 即可，和 ollama 是同一套
        # 复用逻辑，不需要单独的 provider 实现。
        return OpenAICompatibleProvider(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout_s=timeout_s,
        )
    raise ValueError(f"未知 provider {name!r}，可选：{', '.join(SUPPORTED_PROVIDERS)}")
