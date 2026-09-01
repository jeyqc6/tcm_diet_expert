#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型能力档案——目前只有一项：上下文窗口大小（tokens）。

决策依据：`docs/DECISIONS.md` D13("PRD 定义能力档案(上下文长度、结构化输出、
推理复杂度、语言约束)，具体模型作为可替换的实现注记记录于本文档，不在 PRD
里锁定模型名")。这个原则目前只有"上下文长度"这一项有真正的消费者
(`backend/memory/compression.py` 的压缩触发阈值需要知道"这个模型实际能装
多少")，结构化输出/推理复杂度/语言约束这几项还没有代码需要读取，不在这里
提前建自己不知道具体形状的字段——真需要时再加，不是这次顺带发明。

和 `backend/observability/cost.py` 的 `_PRICE_PER_MILLION` 同一种模式(前缀
匹配表 + 未知模型走保守兜底，不是要求精确型号的注册表)：模型名字符串经常带
日期后缀(`claude-haiku-4-5-20251001`)或版本变体，前缀匹配比要求调用方传
"精确型号"更贴近真实调用场景，新增一个模型只用改这一张表。

**为什么需要这个模块，而不是把 PRD §12.3 的 32k/10k/12k 这些数字当成放之
四海而皆准的绝对值**：那张预算表是"这个项目希望消耗多少上下文"这一层主动
的成本/延迟决策，数字本身假设了一个"足够大"的模型上下文窗口。如果实际配置
的模型窗口比这个假设小(常见于本地小模型——某些 Ollama 拉下来的量化模型
只有 4k-8k 上下文)，沿用没有缩放过的固定阈值会让"压缩逻辑觉得还没到该触发
的时候"和"这次请求其实已经超出模型物理上限"这两件事脱节。
`backend/memory/compression.py` 用这里给出的窗口大小，按比例收紧 PRD 假设
的预算(只收紧、不放大——见该模块 `_effective_budget()` 的说明)。
"""
from __future__ import annotations

import os

# PRD §12.3 的预算表按"中枢 agent ≤32k tokens"这个上限设计；不认识的模型
# (且没有 LLM_CONTEXT_WINDOW_OVERRIDE 覆盖)时假设窗口"恰好够用"，不缩放——
# 这是保守的默认行为(不无端收紧未知模型的可用预算)，不是精确值。
DEFAULT_CONTEXT_WINDOW_TOKENS = 32_000

# 前缀匹配，和 backend/observability/cost.py 同一套写法；新增模型只改这一张表，
# 不用碰下面的函数逻辑。数字是官方文档公开的上下文窗口上限，不是本项目的
# 预算决策(那是上面 DEFAULT_CONTEXT_WINDOW_TOKENS 和 PRD §12.3 的事)。
_CONTEXT_WINDOW_PER_MODEL: list[tuple[str, int]] = [
    ("claude-haiku-4-5", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3-5-haiku", 200_000),
    ("claude-sonnet-5", 200_000),
    ("claude-sonnet-4-5", 200_000),
    ("claude-3-5-sonnet", 200_000),
    ("claude-3-haiku", 200_000),
    ("claude-3-sonnet", 200_000),
    ("claude-3-opus", 200_000),
    ("gpt-4o-mini", 128_000),
    ("gpt-4o", 128_000),
    ("gpt-4.1-mini", 1_047_576),
    ("gpt-4.1", 1_047_576),
]


def context_window_for_model(model: str | None) -> int:
    """`model` 为空、不认识、或没配 override 时，返回
    `DEFAULT_CONTEXT_WINDOW_TOKENS`(保守假设，不缩放预算)。

    **这就是"不同 LLM 上下文窗口不同、需要一个可以改动的地方"这条要求的落点**：
    本地跑一个上下文窗口比较小的模型(比如某个量化过的 Ollama 模型)，不需要
    改这个文件本身——设 `LLM_CONTEXT_WINDOW_OVERRIDE` 环境变量就够了，这个值
    对所有模型统一生效，优先级高于下面的前缀匹配表(和 `LLM_MODEL_DEV`/
    `LLM_PROVIDER_DEV` 这类按环境变量切换配置的既有模式一致，不是这次新发明
    的配置方式)。真的要给某个具体模型登记准确数字(比如新出的模型)，才需要
    编辑 `_CONTEXT_WINDOW_PER_MODEL` 这张表。
    """
    override = os.environ.get("LLM_CONTEXT_WINDOW_OVERRIDE", "").strip()
    if override:
        try:
            return int(override)
        except ValueError:
            pass  # 配置错误按"没配"处理，不让一个打错的环境变量炸掉整条请求链路

    lowered = (model or "").strip().lower()
    for prefix, window in _CONTEXT_WINDOW_PER_MODEL:
        if lowered.startswith(prefix):
            return window
    return DEFAULT_CONTEXT_WINDOW_TOKENS
