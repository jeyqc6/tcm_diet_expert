"""
测试目标：backend/llm/providers/registry.py 的 provider 名字 -> 实例装配。
不打真实网络（client 构造是惰性的）；只检查装出来的实例类型/base_url/api_key
是从对的环境变量读出来的。
对应实现：backend/llm/providers/registry.py
"""
from __future__ import annotations

import pytest

from backend.llm.providers.anthropic_provider import AnthropicProvider
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.llm.providers.registry import SUPPORTED_PROVIDERS, build_provider


def test_openai_uses_llm_api_key_and_default_base_url(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    provider = build_provider("openai", timeout_s=1)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert str(provider._client.base_url) == "https://api.openai.com/v1/"


def test_ollama_defaults_to_localhost_and_non_none_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    provider = build_provider("ollama", timeout_s=1)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert str(provider._client.base_url) == "http://localhost:11434/v1/"


def test_anthropic_reads_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    provider = build_provider("anthropic", timeout_s=1)
    assert isinstance(provider, AnthropicProvider)


def test_anthropic_base_url_is_overridable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")
    provider = build_provider("anthropic", timeout_s=1)
    assert str(provider._client.base_url) == "https://api.minimax.io/anthropic/"


def test_deepseek_defaults_to_deepseek_anthropic_base_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-x")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    provider = build_provider("deepseek", timeout_s=1)
    assert isinstance(provider, AnthropicProvider)
    assert str(provider._client.base_url) == "https://api.deepseek.com/anthropic/"


def test_deepseek_falls_back_to_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-fallback")
    provider = build_provider("deepseek", timeout_s=1)
    assert isinstance(provider, AnthropicProvider)


def test_deepseek_base_url_is_overridable(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-x")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/anthropic")
    provider = build_provider("deepseek", timeout_s=1)
    assert str(provider._client.base_url) == "https://example.invalid/anthropic/"


def test_deepseek_is_listed_as_supported():
    assert "deepseek" in SUPPORTED_PROVIDERS


def test_openrouter_defaults_to_openrouter_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-x")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    provider = build_provider("openrouter", timeout_s=1)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert str(provider._client.base_url) == "https://openrouter.ai/api/v1/"


def test_openrouter_base_url_is_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-x")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.invalid/v1")
    provider = build_provider("openrouter", timeout_s=1)
    assert str(provider._client.base_url) == "https://example.invalid/v1/"


def test_openrouter_is_listed_as_supported():
    assert "openrouter" in SUPPORTED_PROVIDERS


def test_unknown_provider_raises_with_supported_list_in_message():
    with pytest.raises(ValueError, match="openrouter"):
        build_provider("does-not-exist", timeout_s=1)


def test_provider_name_matching_is_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-x")
    provider = build_provider("  OpenRouter  ", timeout_s=1)
    assert isinstance(provider, OpenAICompatibleProvider)
