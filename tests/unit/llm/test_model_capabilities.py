"""
测试目标：backend/llm/model_capabilities.py —— 模型上下文窗口的前缀匹配表 +
环境变量覆盖(D13"能力档案不锁定具体模型"，压缩阈值按实际模型窗口缩放这条
需求的配置入口)。
对应实现：backend/llm/model_capabilities.py
"""
from __future__ import annotations

from backend.llm.model_capabilities import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    context_window_for_model,
)


def test_known_model_prefix_matches():
    assert context_window_for_model("claude-haiku-4-5-20251001") == 200_000
    assert context_window_for_model("gpt-4o-mini") == 128_000


def test_unknown_model_falls_back_to_default():
    assert context_window_for_model("some-brand-new-model-nobody-registered") == DEFAULT_CONTEXT_WINDOW_TOKENS


def test_none_or_empty_model_falls_back_to_default():
    assert context_window_for_model(None) == DEFAULT_CONTEXT_WINDOW_TOKENS
    assert context_window_for_model("") == DEFAULT_CONTEXT_WINDOW_TOKENS


def test_case_insensitive_prefix_match():
    assert context_window_for_model("Claude-Haiku-4-5-20251001") == 200_000


def test_env_override_wins_over_known_model_table():
    """本地跑一个上下文窗口比较小的模型(比如某个量化过的 Ollama 模型)时，
    这是不用改代码就能生效的配置入口——环境变量优先级高于前缀匹配表，
    哪怕传进来的模型名在表里能匹配到一个更大的值。"""
    import os

    os.environ["LLM_CONTEXT_WINDOW_OVERRIDE"] = "8192"
    try:
        assert context_window_for_model("claude-haiku-4-5-20251001") == 8192
    finally:
        del os.environ["LLM_CONTEXT_WINDOW_OVERRIDE"]


def test_env_override_applies_even_to_unknown_models():
    import os

    os.environ["LLM_CONTEXT_WINDOW_OVERRIDE"] = "4096"
    try:
        assert context_window_for_model("some-local-model") == 4096
    finally:
        del os.environ["LLM_CONTEXT_WINDOW_OVERRIDE"]


def test_malformed_env_override_falls_back_gracefully():
    """配置写错了(比如手滑打成非数字)不该炸掉整条请求链路——按"没配"处理。"""
    import os

    os.environ["LLM_CONTEXT_WINDOW_OVERRIDE"] = "not-a-number"
    try:
        assert context_window_for_model("gpt-4o") == 128_000
    finally:
        del os.environ["LLM_CONTEXT_WINDOW_OVERRIDE"]
