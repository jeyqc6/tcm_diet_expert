"""
测试目标：backend/agents/dispatch.py `_compose_task_input()` —— D27 补充
(backend/memory/session_store.py 接线)把会话历史接进 SubAgent task_input
的组装逻辑。
对应实现：backend/agents/dispatch.py
"""
from __future__ import annotations

import asyncio

from backend.agents.dispatch import _compose_task_input, _should_retry_insufficient_evidence
from backend.agents.dispatch import _stream_verification_result
from backend.agents.verification import RejectedItem, SuggestionItem, VerificationResult


def test_no_history_returns_message_unchanged():
    """没有历史(空字符串)时行为和这条接线加入之前完全一样，不引入任何回归——
    这是所有既有调用点(单测里没配置会话历史的场景)保持通过的前提。"""
    assert _compose_task_input("今天该吃什么", "") == "今天该吃什么"


def test_history_is_prepended_with_labeled_sections():
    result = _compose_task_input("今天该吃什么", "turn-0 | fact_query | 结论:红枣性温 | 引用:tcm_001 | 被拒建议:无 | 触发的guardrail:无")
    assert "近期对话历史" in result
    assert "本次用户消息" in result
    assert "红枣性温" in result
    assert result.endswith("今天该吃什么")


def test_history_and_message_both_fully_present():
    history = "一些历史文本"
    message = "这次的问题"
    result = _compose_task_input(message, history)
    assert history in result
    assert message in result


def test_should_retry_insufficient_evidence_on_flag():
    result = VerificationResult(
        accepted=[],
        rejected=[],
        needs_reconciliation_retry=True,
    )
    assert _should_retry_insufficient_evidence(result) is True


def test_should_retry_insufficient_evidence_on_missing_source():
    result = VerificationResult(
        accepted=[],
        rejected=[
            RejectedItem(
                item=SuggestionItem(text="无依据"),
                check_number=1,
                reason="missing source",
            )
        ],
    )
    assert _should_retry_insufficient_evidence(result) is True


def test_should_retry_insufficient_evidence_on_citation_relevance_soft_reject():
    """check_number=2(LLM 判定"引用内容不真的支持该建议")和 check_number=1
    并列进"依据不足"重试——两者的病因和补救方式相同(再检索一次拿更贴切的
    依据)，不该只有 1 能重试、2 不能。真实场景见 2026-08-31 那次事故:调和层
    整条回复被 check 2 拒了，但内容本身没有安全问题，只是引用没扣准。"""
    result = VerificationResult(
        accepted=[],
        rejected=[
            RejectedItem(
                item=SuggestionItem(text="引用没扣准"),
                check_number=2,
                reason="引用的 source_id 仅支持某个细节，不支持整体结论",
            )
        ],
    )
    assert _should_retry_insufficient_evidence(result) is True


def test_should_not_retry_on_allergen_or_ed_or_diagnostic_rejection():
    """check 4(过敏原)/6(ED)/8(诊断性表述)不该触发"依据不足"重试——这几条的
    病因是内容本身违规，"再检索一次"解决不了，必须走各自专门的硬阻断路径，
    不能被这条通用重试机制误吸收。"""
    for check_number in (4, 6, 8):
        result = VerificationResult(
            accepted=[],
            rejected=[
                RejectedItem(item=SuggestionItem(text="违规内容"), check_number=check_number, reason="硬阻断")
            ],
        )
        assert _should_retry_insufficient_evidence(result) is False, f"check_number={check_number}"


def test_should_not_retry_when_accepted():
    result = VerificationResult(
        accepted=[SuggestionItem(text="ok [source: t1]")],
        rejected=[],
        needs_reconciliation_retry=False,
    )
    assert _should_retry_insufficient_evidence(result) is False


def test_allergen_rejection_emits_safe_fallback_message():
    """A blocked recommendation gets a safe notice without exposing its draft."""
    result = VerificationResult(
        accepted=[],
        rejected=[
            RejectedItem(
                item=SuggestionItem(text="不安全的原始建议：加蚝油。"),
                check_number=4,
                reason="命中用户过敏原(虾→虾)，硬阻断",
                action="hard_block",
                allergen_names=("虾",),
            )
        ],
    )

    async def collect():
        return [event async for event in _stream_verification_result(result, "trace-1")]

    events = asyncio.run(collect())
    rendered = "".join(events)
    assert "安全提示" in rendered
    assert "虾" in rendered
    assert "不安全的原始建议" not in rendered
    assert "蚝油" not in rendered
    assert '"event": "done"' not in rendered
    assert "event: done" in rendered
