"""
测试目标：压缩优先级表分支逻辑、结构化归档摘要模板字段完整性、Tier阈值判断
对应实现：backend/memory/compression.py
覆盖要求：常规——纯函数，不碰数据库/LLM，构造数据直接验证。
"""
from __future__ import annotations


import pytest

from backend.memory.compression import (
    TIER_ARCHIVED_ACTIVE,
    TIER_ARCHIVED_IDLE,
    CompressibleChunk,
    ArchivedSummary,
    TurnRecord,
    build_archived_summary,
    compress_retrieved_chunks,
    drop_oldest_until_within_budget,
    estimate_tokens,
    fifo_drop_oldest_chunks,
    find_summaries_mentioning,
    is_session_idle,
    select_turns_to_archive,
    should_archive_tier1,
    should_compress_retrieval,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("红烧肉" * 20)
    long = estimate_tokens("红烧肉" * 2000)
    assert long > short
    assert long == pytest.approx(short * 100, rel=0.05)


def test_estimate_tokens_never_zero_for_nonempty_text():
    assert estimate_tokens("a") >= 1


# ---------------------------------------------------------------------------
# 压缩优先级表：检索结果(D27 第1点)
# ---------------------------------------------------------------------------


def test_compress_retrieved_chunks_drops_uncited_entirely():
    chunks = [
        CompressibleChunk(source_id="t1", text="阳虚质忌生冷" * 20),
        CompressibleChunk(source_id="t2", text="没有被引用的检索结果" * 20),
    ]
    result = compress_retrieved_chunks(chunks, cited_source_ids=["t1"])
    assert [c.source_id for c in result] == ["t1"]


def test_compress_retrieved_chunks_keeps_cited_but_clears_text():
    chunks = [CompressibleChunk(source_id="t1", text="阳虚质忌生冷" * 20)]
    result = compress_retrieved_chunks(chunks, cited_source_ids=["t1"])
    assert result[0].source_id == "t1"
    assert result[0].text == ""


def test_compress_retrieved_chunks_empty_input():
    assert compress_retrieved_chunks([], cited_source_ids=["t1"]) == []


def test_compress_retrieved_chunks_no_citations_drops_everything():
    chunks = [CompressibleChunk(source_id="t1", text="x"), CompressibleChunk(source_id="t2", text="y")]
    assert compress_retrieved_chunks(chunks, cited_source_ids=[]) == []


def test_should_compress_retrieval_true_when_over_trigger_ratio():
    # 预算 12000 tokens，触发比例 0.8 → 9600；构造明显超过的检索结果。
    big_chunk = CompressibleChunk(source_id="t1", text="字" * 30_000)
    assert should_compress_retrieval([big_chunk]) is True


def test_fifo_drop_oldest_chunks_keeps_newest_within_budget():
    # Mid-loop cannot drop "uncited" — drop oldest first until the 12k budget holds.
    oldest = CompressibleChunk(source_id="old", text="旧" * 30_000)
    newest = CompressibleChunk(source_id="new", text="新" * 100)
    kept = fifo_drop_oldest_chunks([oldest, newest])
    assert [c.source_id for c in kept] == ["new"]


def test_fifo_drop_oldest_chunks_noop_when_under_budget():
    chunks = [CompressibleChunk(source_id="t1", text="短文本")]
    assert fifo_drop_oldest_chunks(chunks) == chunks


def test_should_compress_retrieval_false_when_under_budget():
    small_chunk = CompressibleChunk(source_id="t1", text="阳虚质忌生冷")
    assert should_compress_retrieval([small_chunk]) is False


def test_should_compress_retrieval_scales_down_for_small_context_window_model(monkeypatch):
    """不同 LLM 上下文窗口不同这条要求的具体验证——同样的检索结果，配置一个
    小窗口模型时应该更容易触发压缩(预算被按比例收紧)。"""
    monkeypatch.setenv("LLM_CONTEXT_WINDOW_OVERRIDE", "4000")
    # 4000/32000 的比例下，12000 * (4000/32000) * 0.8 = 1200 tokens 就该触发。
    chunk = CompressibleChunk(source_id="t1", text="字" * 3000)  # 约 1666 tokens
    assert should_compress_retrieval([chunk], model="claude-haiku-4-5") is True


def test_should_compress_retrieval_large_context_window_does_not_expand_budget(monkeypatch):
    """窗口比 32k 假设更大时不放大预算——预算是主动的成本决策，不是"能装多少
    就该塞多少"。"""
    monkeypatch.setenv("LLM_CONTEXT_WINDOW_OVERRIDE", "200000")
    # 未缩放/未放大情况下，9600 tokens 边界之内不该触发。
    chunk = CompressibleChunk(source_id="t1", text="字" * 15_000)  # 约 8333 tokens
    assert should_compress_retrieval([chunk], model="claude-haiku-4-5") is False


# ---------------------------------------------------------------------------
# 结构化归档摘要(D27 第2点)
# ---------------------------------------------------------------------------


def test_archived_summary_render_matches_d27_template():
    summary = ArchivedSummary(
        turn_id="t42",
        branch="candidate_eval",
        conclusion="红烧肉可以适量食用",
        cited_source_ids=("t1", "n1"),
        rejected_suggestions=("建议2:含虾过敏原",),
        triggered_guardrails=("allergen_hit",),
    )
    rendered = summary.render()
    assert rendered == (
        "t42 | candidate_eval | 结论:红烧肉可以适量食用 | 引用:t1、n1 | "
        "被拒建议:建议2:含虾过敏原 | 触发的guardrail:allergen_hit"
    )


def test_archived_summary_render_uses_placeholder_when_fields_empty():
    summary = ArchivedSummary(turn_id="t1", branch="log_review", conclusion="今天没有饮食记录")
    rendered = summary.render()
    assert "引用:无" in rendered
    assert "被拒建议:无" in rendered
    assert "触发的guardrail:无" in rendered


def test_build_archived_summary_preserves_short_conclusion_verbatim():
    turn = TurnRecord(turn_id="t1", branch="fact_query", raw_text="...", conclusion="红枣性温")
    summary = build_archived_summary(turn)
    assert summary.conclusion == "红枣性温"


def test_build_archived_summary_truncates_long_conclusion_deterministically():
    turn = TurnRecord(turn_id="t1", branch="full_recommend", raw_text="...", conclusion="很长的结论" * 50)
    summary = build_archived_summary(turn, max_conclusion_chars=20)
    assert len(summary.conclusion) == 20
    assert summary.conclusion.endswith("…")


def test_build_archived_summary_does_not_call_llm():
    """D27 第2点的核心主张——固定模板取代自由文本 LLM 摘要，这个函数应该是
    纯确定性组装：不接受、也不需要 complete 参数。"""
    import inspect

    sig = inspect.signature(build_archived_summary)
    assert "complete" not in sig.parameters


def test_find_summaries_mentioning_matches_conclusion_field():
    summaries = [
        ArchivedSummary(turn_id="t1", branch="log_write", conclusion="记录了番茄炒蛋"),
        ArchivedSummary(turn_id="t2", branch="candidate_eval", conclusion="已记录您对虾过敏"),
    ]
    hits = find_summaries_mentioning(summaries, "过敏")
    assert [s.turn_id for s in hits] == ["t2"]


def test_find_summaries_mentioning_matches_rejected_and_guardrail_fields():
    summaries = [
        ArchivedSummary(
            turn_id="t1", branch="full_recommend", conclusion="综合建议",
            rejected_suggestions=("含虾过敏原被拒",),
        ),
        ArchivedSummary(
            turn_id="t2", branch="full_recommend", conclusion="综合建议2",
            triggered_guardrails=("过敏原硬阻断",),
        ),
    ]
    hits = find_summaries_mentioning(summaries, "过敏")
    assert {s.turn_id for s in hits} == {"t1", "t2"}


def test_find_summaries_mentioning_no_match_returns_empty():
    summaries = [ArchivedSummary(turn_id="t1", branch="log_review", conclusion="今天没有记录")]
    assert find_summaries_mentioning(summaries, "过敏") == []


# ---------------------------------------------------------------------------
# 两级触发：中枢会话历史 Tier1→Tier2
# ---------------------------------------------------------------------------


def test_should_archive_tier1_false_for_short_session():
    turns = [TurnRecord(turn_id="t1", branch="fact_query", raw_text="很短", conclusion="c")]
    assert should_archive_tier1(turns) is False


def test_should_archive_tier1_true_when_over_trigger_ratio():
    turns = [TurnRecord(turn_id="t1", branch="full_recommend", raw_text="字" * 20_000, conclusion="c")]
    assert should_archive_tier1(turns) is True


def test_select_turns_to_archive_keeps_at_least_keep_recent():
    """即使全部轮次加起来远超预算，也不能把 Tier1 清空——至少留
    `keep_recent` 轮原文，这是 Tier1"最近N轮，原文"这条定位的底线。"""
    turns = [
        TurnRecord(turn_id=f"t{i}", branch="full_recommend", raw_text="字" * 20_000, conclusion=f"c{i}")
        for i in range(5)
    ]
    to_archive, remaining = select_turns_to_archive(turns, keep_recent=2)
    assert len(remaining) == 2
    assert [t.turn_id for t in remaining] == ["t3", "t4"]
    assert [t.turn_id for t in to_archive] == ["t0", "t1", "t2"]


def test_select_turns_to_archive_archives_oldest_first_until_under_threshold():
    turns = [
        TurnRecord(turn_id="old", branch="full_recommend", raw_text="字" * 15_000, conclusion="c"),
        TurnRecord(turn_id="new", branch="full_recommend", raw_text="短", conclusion="c"),
    ]
    to_archive, remaining = select_turns_to_archive(turns, keep_recent=1)
    assert [t.turn_id for t in to_archive] == ["old"]
    assert [t.turn_id for t in remaining] == ["new"]


def test_select_turns_to_archive_no_archiving_needed_when_under_budget():
    turns = [TurnRecord(turn_id="t1", branch="fact_query", raw_text="短", conclusion="c")]
    to_archive, remaining = select_turns_to_archive(turns)
    assert to_archive == []
    assert remaining == turns


# ---------------------------------------------------------------------------
# 会话空闲判定(Tier2→Tier3)
# ---------------------------------------------------------------------------


def test_is_session_idle_true_after_threshold():
    assert is_session_idle(last_activity_ts=0.0, now=2000.0, idle_threshold_s=1800) is True


def test_is_session_idle_false_within_threshold():
    assert is_session_idle(last_activity_ts=0.0, now=100.0, idle_threshold_s=1800) is False


# ---------------------------------------------------------------------------
# 步骤2同步紧急兜底：从 Tier3 最旧的丢，不从 Tier1 丢
# ---------------------------------------------------------------------------


def test_drop_oldest_until_within_budget_removes_oldest_first():
    summaries = [
        ArchivedSummary(turn_id=f"t{i}", branch="full_recommend", conclusion="字" * 500)
        for i in range(20)
    ]
    result = drop_oldest_until_within_budget(summaries, budget_tokens=1000)
    # 应该只剩最新的几条，且保留的是列表尾部(最新)，不是随机丢弃。
    assert result == tuple(summaries[len(summaries) - len(result):])
    assert sum(estimate_tokens(s.render()) for s in result) <= 1000


def test_drop_oldest_until_within_budget_noop_when_already_under_budget():
    summaries = [ArchivedSummary(turn_id="t1", branch="fact_query", conclusion="短")]
    result = drop_oldest_until_within_budget(summaries, budget_tokens=10_000)
    assert result == tuple(summaries)


def test_drop_oldest_until_within_budget_can_drop_everything_if_still_over():
    summaries = [ArchivedSummary(turn_id="t1", branch="full_recommend", conclusion="字" * 10_000)]
    result = drop_oldest_until_within_budget(summaries, budget_tokens=1)
    assert result == ()


def test_drop_oldest_until_within_budget_prefers_tier3_even_if_positioned_earlier_in_list():
    # tier2 排在列表最前面(时间上更旧)，tier3 排在后面(时间上更新)——按 tier
    # 优先级丢弃时，即便 tier3 更新，也该先丢它，tier2 保留下来。
    tier2 = ArchivedSummary(turn_id="t1", branch="x", conclusion="老会话仍在进行", tier=TIER_ARCHIVED_ACTIVE)
    tier3 = ArchivedSummary(turn_id="t2", branch="x", conclusion="字" * 500, tier=TIER_ARCHIVED_IDLE)
    budget = estimate_tokens(tier2.render()) + 1
    result = drop_oldest_until_within_budget([tier2, tier3], budget_tokens=budget)
    assert result == (tier2,)


def test_drop_oldest_until_within_budget_falls_through_to_tier2_when_tier3_exhausted():
    tier3 = ArchivedSummary(turn_id="t1", branch="x", conclusion="字" * 500, tier=TIER_ARCHIVED_IDLE)
    tier2 = ArchivedSummary(turn_id="t2", branch="x", conclusion="字" * 500, tier=TIER_ARCHIVED_ACTIVE)
    # 预算小到两条都装不下——丢完全部 tier3 之后还超预算，必须继续丢 tier2。
    result = drop_oldest_until_within_budget([tier3, tier2], budget_tokens=1)
    assert result == ()


def test_drop_oldest_until_within_budget_drops_multiple_tier3_oldest_first():
    tier3_old = ArchivedSummary(turn_id="t1", branch="x", conclusion="字" * 500, tier=TIER_ARCHIVED_IDLE)
    tier3_new = ArchivedSummary(turn_id="t2", branch="x", conclusion="字" * 500, tier=TIER_ARCHIVED_IDLE)
    tier2 = ArchivedSummary(turn_id="t3", branch="x", conclusion="短", tier=TIER_ARCHIVED_ACTIVE)
    budget = estimate_tokens(tier2.render()) + 1
    result = drop_oldest_until_within_budget([tier3_old, tier3_new, tier2], budget_tokens=budget)
    assert result == (tier2,)


def test_drop_oldest_until_within_budget_keeps_chronological_order_among_survivors():
    tier2_a = ArchivedSummary(turn_id="t1", branch="x", conclusion="a", tier=TIER_ARCHIVED_ACTIVE)
    tier3 = ArchivedSummary(turn_id="t2", branch="x", conclusion="字" * 500, tier=TIER_ARCHIVED_IDLE)
    tier2_b = ArchivedSummary(turn_id="t3", branch="x", conclusion="b", tier=TIER_ARCHIVED_ACTIVE)
    budget = estimate_tokens(tier2_a.render()) + estimate_tokens(tier2_b.render()) + 1
    result = drop_oldest_until_within_budget([tier2_a, tier3, tier2_b], budget_tokens=budget)
    # 丢掉中间的 tier3 之后，剩下两条 tier2 仍按原来的时间顺序排列。
    assert result == (tier2_a, tier2_b)


# ---------------------------------------------------------------------------
# 端到端场景(BUILD_PLAN 阶段7完成判据)：20 轮对话，第2轮说过敏，
# 第20轮问推荐，过敏原相关记录不因压缩而消失。
#
# 范围说明：这条测试验证的是 compression.py 自己的职责——结构化摘要在
# Tier1→Tier2→Tier3 的多轮流转中不丢失、可被结构化查询到。user_profile
# (Tier0)才是"过敏原"这条事实真正永久生效的地方(靠类型层面完全不参与压缩
# 保证，见模块文档)，那是 backend/memory/critical_fact_scanner.py 的职责，
# 不在这个文件的范围内——这条测试不依赖也不验证那部分。
# ---------------------------------------------------------------------------


def test_twenty_turn_conversation_allergen_summary_survives_compression():
    turns = []
    for i in range(20):
        if i == 1:  # 第2轮(0-indexed)
            turns.append(
                TurnRecord(
                    turn_id="turn-2",
                    branch="log_write",
                    raw_text="对了我对虾过敏" + "，补充说明" * 400,
                    conclusion="已记录您对虾过敏，后续推荐会自动避开",
                    triggered_guardrails=("allergen_disclosed",),
                )
            )
        else:
            turns.append(
                TurnRecord(
                    turn_id=f"turn-{i + 1}",
                    branch="full_recommend",
                    raw_text="今天该吃什么" + "，日常闲聊内容" * 400,
                    conclusion=f"第{i + 1}轮的综合建议",
                )
            )

    # 模拟"结束一轮就检查一次要不要归档"：每追加一轮，都跑一次触发检查+归档，
    # 直到跑完全部20轮(对应步骤9在每轮响应后异步执行一次的既有设计)。
    tier1: list[TurnRecord] = []
    tier3: list[ArchivedSummary] = []
    for turn in turns:
        tier1.append(turn)
        if should_archive_tier1(tier1):
            to_archive, tier1 = select_turns_to_archive(tier1, keep_recent=2)
            tier3.extend(build_archived_summary(t) for t in to_archive)

    # 20 轮全部处理完之后，每一轮要么还在 Tier1(原文)，要么已经归档进
    # Tier3(摘要)，不多不少，没有轮次在这个过程里凭空消失。
    all_turn_ids = {t.turn_id for t in tier1} | {s.turn_id for s in tier3}
    assert all_turn_ids == {f"turn-{i + 1}" for i in range(20)}
    assert len(tier1) + len(tier3) == 20

    # 第2轮(过敏原披露)这时候大概率已经被压缩进 Tier3 了(20轮显然超过了
    # Tier1"最近几轮"的容量)——用 D27 承诺的"结构化摘要可查询"这条能力，
    # 在第20轮这个时间点，仍然能查到这条记录，没有被当成"低价值内容"丢弃。
    assert "turn-2" in {s.turn_id for s in tier3}, "过敏原披露这一轮应该已经被归档，不是还留在 Tier1 原文里"
    hits = find_summaries_mentioning(tier3, "过敏")
    assert any(s.turn_id == "turn-2" for s in hits)
    assert "虾过敏" in next(s for s in hits if s.turn_id == "turn-2").conclusion
