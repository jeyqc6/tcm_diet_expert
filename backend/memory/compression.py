#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层压缩：压缩优先级表 + 结构化归档摘要 + 两级触发时机（SubAgent内同步/中枢异步+同步兜底）。

设计依据：docs/ARCHITECTURE.md §4.4/§4.4.1
决策依据：docs/DECISIONS.md D8/D27
roadmap：阶段 7（本项目技术制高点）

## 这个模块管什么、不管什么

只处理 §4.4 明确划定的压缩对象——**会话历史 + 检索结果**。`user_profile`
(Tier 0)永不参与压缩，靠**类型层面**保证，不是靠约定：本文件定义的
`TurnRecord`/`ArchivedSummary` 两个数据结构都没有"用户画像"这个字段，压缩
逻辑物理上碰不到它，同 `backend/agents/reconciliation.py` 用函数签名强制
D14 边界的做法一脉相承。

**依赖方向**：本文件在 `backend/memory/` 下，只允许依赖 `backend/llm/`（已有
先例，见 `dish_decomposition.py` 导入 `CompleteFn`），不依赖 `backend/agents/`
或 `backend/mcp_server/`——那两个包才是依赖 `backend/memory/` 的一方(见
`_subagent_common.py`/`log_write.py`)，反过来会成环。这带来两个具体设计
选择：
  1. `cited_source_ids` 作为参数直接传入(`list[str]`)，不在本文件里调用
     `backend.agents.citation.extract_cited_ids()` 解析原文——那个函数在
     `backend/agents/`，解析这一步由调用方(以后接线中枢编排层时)先做。
  2. 压缩优先级表操作的是本文件自己定义的 `CompressibleChunk`(只有
     `source_id`/`text` 两个字段)，不是 `backend/mcp_server/tools/
     _retrieval_common.py` 的 `RetrievedChunk`——后者不但在另一个包，文件名
     前缀下划线本身就表明"仅供本包内部使用"，从 `backend/memory/` 跨包导入
     一个私有模块违反它自己的命名意图。调用方把 `RetrievedChunk` 转成
     `CompressibleChunk(source_id=c.source_id, text=c.text)` 是一行代码，
     不值得为了省这一行牺牲包边界的清晰度。

## 接线状态（2026-08-29）

会话历史压缩已由 `backend/memory/session_store.py` 接进 `/api/chat`。
SubAgent Level 1 检索压缩已接进 `run_agent_loop`：每次 retrieve_*
`tool_result` 追加后，`fifo_drop_oldest_chunks` 按 FIFO 丢掉最旧 chunk
直到 12k 预算成立。不能在循环中途按「未引用」丢弃——那时 `final_text`
还不存在（ARCHITECTURE §4.4.1）。`compress_retrieved_chunks`（引用感知）
仍留给循环结束后的归档路径，不在本轮 mid-loop 使用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from backend.llm.model_capabilities import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    context_window_for_model,
)

# ---------------------------------------------------------------------------
# token 估算——粗略字符数估算，不是精确分词
# ---------------------------------------------------------------------------

# 中文字符密集文本比纯英文更接近"1 字符≈1 token"，混合内容用一个比纯英文
# 保守(更小)的比例，宁可压缩触发早一点，也不要因为低估 token 数导致真的
# 超出模型窗口——和其余压缩阈值一样是"可以改动的地方"，不是精确校准值，
# 阶段 7/8 有真实数据后可以按需调整。
CHARS_PER_TOKEN_ESTIMATE = 1.8


def estimate_tokens(text: str) -> int:
    """粗略估算，不引入 tiktoken 这类依赖——压缩触发判断只需要"大概是不是
    超预算了"这个粒度，不需要和真实计费对齐到个位数(真实计费靠 provider
    返回的 usage，见 `backend/observability/cost.py`)。"""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


# ---------------------------------------------------------------------------
# PRD §12.3 预算 + 按模型上下文窗口缩放(backend/llm/model_capabilities.py)
# ---------------------------------------------------------------------------

# 以下四个是 PRD §12.3 / DECISIONS.md D27 修订二给出的具体数字，按"模型窗口
# 足够大(≥32k)"这个假设设计——真正生效的阈值是 `_effective_budget()` 按
# `context_window_for_model()` 缩放之后的结果，不是直接用这里的原始数字。
SUBAGENT_RETRIEVAL_BUDGET_TOKENS = 12_000
SUBAGENT_RETRIEVAL_TRIGGER_RATIO = 0.8  # ≈9.6k

SESSION_HISTORY_BUDGET_TOKENS = 10_000
TIER1_ARCHIVE_TRIGGER_RATIO = 0.6  # ≈6k，给 Tier2/3 留空间

# 会话空闲超过这个时长判定结束，Tier2 摘要整体折叠进 Tier3——阈值待实测调整
# (ARCHITECTURE §4.4.1 原文明确标注"阈值待实测调整"，这里不是钉死的数字)。
SESSION_IDLE_THRESHOLD_SECONDS: float = 30 * 60

# compression_tier 列值约定——定义在这里(不是 session_store.py)是因为
# `ArchivedSummary.tier` 需要用到它，而 session_store.py 已经依赖本文件，
# 反过来才会成环(见模块文档"依赖方向")；session_store.py 从这里 re-export。
TIER_RAW = 0
TIER_ARCHIVED_ACTIVE = 1
TIER_ARCHIVED_IDLE = 3


def _effective_budget(nominal_budget_tokens: int, *, model: str | None) -> int:
    """按实际模型的上下文窗口缩放 PRD 假设的预算——**只收紧，不放大**。

    模型窗口比 `DEFAULT_CONTEXT_WINDOW_TOKENS`(32k)大是常态(Claude/GPT 主力
    模型的真实窗口远超 32k)，但预算数字本身是主动的成本/延迟决策，不是
    "模型能装多少就该塞多少"——窗口更大不代表应该多用，所以只在窗口**小于**
    假设值时才按比例收紧，避免小窗口模型收到一个它物理上装不下的请求。
    """
    window = context_window_for_model(model)
    if window >= DEFAULT_CONTEXT_WINDOW_TOKENS:
        return nominal_budget_tokens
    scale = window / DEFAULT_CONTEXT_WINDOW_TOKENS
    return max(1, int(nominal_budget_tokens * scale))


# ---------------------------------------------------------------------------
# 压缩优先级表(D27 第 1 点)——检索结果/工具调用原始内容这一层
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressibleChunk:
    """压缩逻辑需要的最小 chunk 形状——只有 `source_id`/`text`，不是
    `RetrievedChunk` 的别名(见模块文档"依赖方向"一节，两者故意解耦)。"""

    source_id: str
    text: str


def should_compress_retrieval(
    chunks: Sequence[CompressibleChunk], *, model: str | None = None
) -> bool:
    """SubAgent 内 Level 1 触发检查(ARCHITECTURE §4.4.1)——检索结果子分区
    估算 token 数是否超过预算的 `SUBAGENT_RETRIEVAL_TRIGGER_RATIO`。"""
    total = sum(estimate_tokens(c.text) for c in chunks)
    budget = _effective_budget(SUBAGENT_RETRIEVAL_BUDGET_TOKENS, model=model)
    return total > budget * SUBAGENT_RETRIEVAL_TRIGGER_RATIO


def fifo_drop_oldest_chunks(
    chunks: Sequence[CompressibleChunk], *, model: str | None = None
) -> list[CompressibleChunk]:
    """Level 1 mid-loop trim (ARCHITECTURE §4.4.1 timing).

    Citation-aware drop is impossible here: final_text does not exist yet.
    Drop the oldest retrieved chunks until the 12k budget holds.
    """
    kept = list(chunks)
    budget = _effective_budget(SUBAGENT_RETRIEVAL_BUDGET_TOKENS, model=model)
    while kept and sum(estimate_tokens(c.text) for c in kept) > budget:
        kept.pop(0)
    return kept


def compress_retrieved_chunks(
    chunks: Sequence[CompressibleChunk], cited_source_ids: Iterable[str]
) -> list[CompressibleChunk]:
    """应用压缩优先级表第 1/2 条(D27)：

    - 未被引用的 chunk：直接从结果中删除，不摘要——一条检索结果如果最终
      没进结论，说明它对当前判断没有价值，不值得先花一次 LLM 调用去总结它。
    - 被引用的 chunk：只保留 `source_id`，`text` 清空——溯源展开
      (ARCHITECTURE §5.2 步骤 8)靠 `source_id` 回查数据库，会话历史里留一份
      原文副本是重复存储。

    `cited_source_ids` 由调用方从 SubAgent 的 `final_text` 里解析好再传进来
    (`backend.agents.citation.extract_cited_ids()`)，本函数不做文本解析。
    """
    cited = set(cited_source_ids)
    return [
        CompressibleChunk(source_id=c.source_id, text="")
        for c in chunks
        if c.source_id in cited
    ]


# ---------------------------------------------------------------------------
# 结构化归档摘要(D27 第 2 点)——Tier 2/3 固定模板，不是自由文本 LLM 摘要
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnRecord:
    """Tier 1 的一条原文轮次——中枢会话历史的最小单元。

    `branch` 是路由分支的字符串值(如 `"full_recommend"`)，不是
    `backend.agents.routing.RouteBranch` 枚举本身——见模块文档"依赖方向"。
    `conclusion` 由调用方在生成阶段就已经产出(调和层/核查 pass 的最终结论
    文本，或 log_review/log_write 这类不经过调和层分支自己的确定性结论)，
    本文件不负责从 `raw_text` 里提炼结论。
    """

    turn_id: str
    branch: str
    raw_text: str
    conclusion: str
    cited_source_ids: tuple[str, ...] = ()
    rejected_suggestions: tuple[str, ...] = ()
    triggered_guardrails: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.raw_text)


@dataclass(frozen=True)
class ArchivedSummary:
    """Tier 2/3 的结构化归档摘要——D27 固定模板，每条是独立的结构化记录
    (类比"像 `git log` 不是 `git squash`")，不是自由文本摘要，因此可以被
    结构化查询(见下面 `find_summaries_mentioning()`)。"""

    turn_id: str
    branch: str
    conclusion: str
    cited_source_ids: tuple[str, ...] = ()
    rejected_suggestions: tuple[str, ...] = ()
    triggered_guardrails: tuple[str, ...] = ()
    # Tier2(仍在进行的会话)还是 Tier3(已判定空闲/结束)——只影响
    # `drop_oldest_until_within_budget()` 的丢弃优先级，不影响 `render()`
    # 的输出内容(两个 tier 用的是同一个模板)。新摘要默认落在 Tier2，
    # `session_store.maybe_fold_idle_session()` 折叠时才会变成 Tier3。
    tier: int = TIER_ARCHIVED_ACTIVE

    def render(self) -> str:
        """D27 原文固定模板：
        `{turn_id} | {branch} | 结论:{一句话} | 引用:{source_id列表} | 被拒建议:{若有} | 触发的guardrail:{若有}`
        """
        cited = "、".join(self.cited_source_ids) if self.cited_source_ids else "无"
        rejected = "；".join(self.rejected_suggestions) if self.rejected_suggestions else "无"
        guardrails = "、".join(self.triggered_guardrails) if self.triggered_guardrails else "无"
        return (
            f"{self.turn_id} | {self.branch} | 结论:{self.conclusion} | "
            f"引用:{cited} | 被拒建议:{rejected} | 触发的guardrail:{guardrails}"
        )


# 归档摘要"结论"字段的长度上限——纯确定性截断，不调用 LLM 做语义压缩。
# D27 第 2 点的核心主张是"用固定模板取代自由文本 LLM 摘要"，模板本身已经
# 是压缩后的结构化数据，`conclusion` 字段的值理应在生成阶段就是一句话(调和
# 层/核查 pass 本来就是"一次性给结论"，不是长篇大论)；这里的截断只是防止
# 极端情况(比如调用方传入了没有事先浓缩过的长文本)撑爆归档记录，不是常规
# 路径依赖的行为。
MAX_ARCHIVED_CONCLUSION_CHARS = 80


def build_archived_summary(
    turn: TurnRecord, *, max_conclusion_chars: int = MAX_ARCHIVED_CONCLUSION_CHARS
) -> ArchivedSummary:
    """把一条 `TurnRecord` 装进 D27 固定模板——纯确定性组装，不调用 LLM。"""
    conclusion = turn.conclusion.strip()
    if len(conclusion) > max_conclusion_chars:
        conclusion = conclusion[: max_conclusion_chars - 1].rstrip() + "…"
    return ArchivedSummary(
        turn_id=turn.turn_id,
        branch=turn.branch,
        conclusion=conclusion,
        cited_source_ids=turn.cited_source_ids,
        rejected_suggestions=turn.rejected_suggestions,
        triggered_guardrails=turn.triggered_guardrails,
    )


def find_summaries_mentioning(
    summaries: Sequence[ArchivedSummary], keyword: str
) -> list[ArchivedSummary]:
    """D27 原文举的例子："上次因为过敏原被拒绝的建议是什么"——结构化摘要的
    价值就是能被这样查询，自由文本摘要做不到。这里只做最朴素的子串匹配
    (在 结论/被拒建议/guardrail 三个字段里找)，不是语义检索；真的需要语义
    检索时这些摘要该发进向量库，不是这个函数的职责。"""
    hits = []
    for s in summaries:
        haystack = " ".join([s.conclusion, *s.rejected_suggestions, *s.triggered_guardrails])
        if keyword in haystack:
            hits.append(s)
    return hits


# ---------------------------------------------------------------------------
# 两级触发(D27 修订二)——中枢会话历史 Tier1→Tier2、Tier2→Tier3、紧急兜底
# ---------------------------------------------------------------------------


def should_archive_tier1(
    turns: Sequence[TurnRecord], *, model: str | None = None
) -> bool:
    """步骤 9(会话落库，响应已发出之后，异步)的触发检查：Tier1 累计估算
    token 是否超过预算的 `TIER1_ARCHIVE_TRIGGER_RATIO`。"""
    total = sum(t.estimated_tokens() for t in turns)
    budget = _effective_budget(SESSION_HISTORY_BUDGET_TOKENS, model=model)
    return total > budget * TIER1_ARCHIVE_TRIGGER_RATIO


def select_turns_to_archive(
    turns: Sequence[TurnRecord], *, model: str | None = None, keep_recent: int = 1
) -> tuple[list[TurnRecord], list[TurnRecord]]:
    """决定 Tier1 里哪些最旧的轮次该被归档成 Tier2 摘要——从最旧的开始丢，
    直到剩余部分回落到阈值以下，但至少保留 `keep_recent` 轮原文(Tier1 的
    定位是"当前会话最近 N 轮，原文"，不能被这个函数整个清空，否则 Tier1
    就没有存在的意义了)。

    `turns` 必须按时间正序传入(旧→新)。返回 `(待归档, 仍留在Tier1)`。
    """
    ordered = list(turns)
    if len(ordered) <= keep_recent:
        return [], ordered
    budget = _effective_budget(SESSION_HISTORY_BUDGET_TOKENS, model=model)
    threshold_tokens = budget * TIER1_ARCHIVE_TRIGGER_RATIO
    to_archive: list[TurnRecord] = []
    remaining = ordered
    while len(remaining) > keep_recent and sum(t.estimated_tokens() for t in remaining) > threshold_tokens:
        to_archive.append(remaining[0])
        remaining = remaining[1:]
    return to_archive, remaining


def is_session_idle(
    last_activity_ts: float,
    *,
    now: float | None = None,
    idle_threshold_s: float = SESSION_IDLE_THRESHOLD_SECONDS,
) -> bool:
    """会话是否该判定结束(Tier2 整体折叠进 Tier3)。折叠本身不改变数据——
    `ArchivedSummary` 在 Tier2/Tier3 是同一种结构，区别只是"这个会话是不是
    还在进行中"这个存储层面的生命周期状态，由调用方(以后接线持久层时)决定
    要不要把这些记录从"当前会话"的分区移到"历史会话"的分区，不是这个模块
    要做数据变换的事。"""
    effective_now = now if now is not None else time.time()
    return (effective_now - last_activity_ts) > idle_threshold_s


def drop_oldest_until_within_budget(
    summaries: Sequence[ArchivedSummary],
    *,
    model: str | None = None,
    budget_tokens: int | None = None,
) -> tuple[ArchivedSummary, ...]:
    """步骤 2(准备派发下一轮请求上下文，同步)的紧急兜底(D27 修订二)——
    只在异步归档任务没赶上、Tier1 还没来得及被摘要的紧急情况下调用；正常
    路径下步骤 9 的异步归档应该已经让 Tier1 回落，不需要这里介入。

    不从 Tier1 丢——丢 Tier1 原文等于丢了本该异步生成、还没来得及生成的
    摘要，信息损失比丢一条已经归档过的摘要更大(`summaries` 参数本身就
    只该传 Tier2/3，Tier1 原文不经过这个函数)。两个已归档的 tier 之间，
    优先丢 Tier3(已判定会话空闲/结束)里最旧的，Tier3 丢完仍超预算才动
    Tier2(会话仍在进行中)——同样从最旧的开始丢；两轮丢弃都不打乱剩余
    记录原本的时间顺序。不等待、不调用 LLM——正在被回答的这一轮不能被
    压缩逻辑拖慢(PRD §11 fallback 表"丢弃最旧的低价值记录"这条的具体
    实现)。
    """
    budget = (
        budget_tokens
        if budget_tokens is not None
        else _effective_budget(SESSION_HISTORY_BUDGET_TOKENS, model=model)
    )

    def _total_tokens(items: list[ArchivedSummary]) -> int:
        return sum(estimate_tokens(s.render()) for s in items)

    kept = list(summaries)
    for target_tier in (TIER_ARCHIVED_IDLE, TIER_ARCHIVED_ACTIVE):
        i = 0
        while _total_tokens(kept) > budget and i < len(kept):
            if kept[i].tier == target_tier:
                kept.pop(i)
            else:
                i += 1
    return tuple(kept)
