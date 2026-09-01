"""
测试目标：Record/Replay fixture 端到端演示(docs/ENGINEERING.md §7.2、
docs/BUILD_PLAN.md 阶段6"Record/Replay fixture")——真实调用录制一次，之后
默认(LLM_REPLAY_MODE 未设置 => "replay")离线回放，不打网络、不需要任何
LLM API key，CI 环境能直接跑过。
对应实现：backend/llm/providers/replay.py
覆盖要求：同一份测试代码，两种模式都能跑——重新录制时本地设
`LLM_REPLAY_MODE=record` + `ANTHROPIC_API_KEY` 再跑一次这个文件，把新生成的
fixture(`tests/fixtures/llm_replay/demo_router_call__*.json`)提交进仓库。
"""
from __future__ import annotations

import asyncio

import pytest

from backend.llm.adapter import NonRetryableError, complete
from backend.llm.providers.replay import ReplayFixtureMissing, replay_provider_for


def _run(coro):
    return asyncio.run(coro)


# 固定 model/provider，不受本地 .env 影响——指纹由 (model, messages, tools) 决定，
# 录制时和回放时如果用了不同的模型/服务商，指纹对不上，测试会报
# ReplayFixtureMissing(这正是设计意图：环境漂移也应该被这条检测器捕捉到)。
_MESSAGES = [
    {"role": "system", "content": "你是 diet_expert 的路由分类器，用一句话说明你的职责。"},
    {"role": "user", "content": "今天该吃什么"},
]


def _pin_model_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_TIER", "dev")
    monkeypatch.setenv("LLM_PROVIDER_DEV", "anthropic")
    monkeypatch.setenv("LLM_MODEL_DEV", "claude-haiku-4-5-20251001")


def test_replay_fixture_round_trip(monkeypatch):
    """默认(replay)模式：这条测试本身就是"CI 回放时离线、零成本"的可运行证明
    ——没有配置任何 LLM API key 也应该能跑过，因为完全不碰网络。"""
    _pin_model_env(monkeypatch)
    provider = replay_provider_for("demo_router_call")

    result = _run(complete(_MESSAGES, provider=provider))

    assert isinstance(result.text, str)
    assert result.text.strip()  # 真实录制的响应，不是空字符串占位符
    assert result.provider == "anthropic"


def test_replay_fixture_missing_when_prompt_drifts(monkeypatch, tmp_path):
    """ENGINEERING §7.2:"指纹对不上就报错——这本身就是 prompt 被意外改动的
    检测器"。用一份和已录制内容不同的消息去查同一个 caller 标签，验证确实
    查不到、而不是静默返回一份对不上的旧响应。

    `complete()` 会把 `ReplayFixtureMissing` 归类成 non_retryable(重试一次
    没有意义——fixture 不会因为重试就出现)，按 adapter.py 的既有约定包装成
    `NonRetryableError` 再往外抛，和其余 provider 异常走同一条路径，不是
    Replay 机制专属的例外行为。原始的 `ReplayFixtureMissing` 保留在
    `__cause__` 里，方便调试时看清具体是"缺 fixture"而不是别的网络错误。"""
    _pin_model_env(monkeypatch)
    provider = replay_provider_for("demo_router_call")
    drifted_messages = _MESSAGES + [{"role": "user", "content": "这条消息在录制时不存在"}]

    with pytest.raises(NonRetryableError) as exc_info:
        _run(complete(drifted_messages, provider=provider))
    assert isinstance(exc_info.value.__cause__, ReplayFixtureMissing)
