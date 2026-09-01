"""
测试目标：九类体质转化分计算、体质夹杂识别、边界分值（恰好40/30分）、
平和质单独判定规则（含"基本是"档位）。
对应实现：backend/onboarding/ccmq_scoring.py
覆盖要求：常规
"""
from __future__ import annotations

import pytest

from backend.onboarding.ccmq_scoring import (
    CONSTITUTIONS,
    PING_HE,
    VERDICT_BASICALLY_YES,
    VERDICT_LEANING_YES,
    VERDICT_NO,
    VERDICT_YES,
    raw_to_transformed,
    score_ccmq,
)

_PATHOLOGICAL = [c for c in CONSTITUTIONS if c != PING_HE]


def _answers_for_raw(raw: int) -> list[int]:
    """Build 5 item scores (1-5 each) summing to `raw`, for raw in [5, 25]."""
    assert 5 <= raw <= 25
    items = [1, 1, 1, 1, 1]
    remaining = raw - 5
    i = 0
    while remaining > 0:
        bump = min(4, remaining)
        items[i] += bump
        remaining -= bump
        i += 1
    return items


def _all_answers(overrides: dict[str, int]) -> dict[str, list[int]]:
    """All nine constitutions default to raw=5 (lowest), overridden by raw sums given."""
    answers = {c: _answers_for_raw(5) for c in CONSTITUTIONS}
    for c, raw in overrides.items():
        answers[c] = _answers_for_raw(raw)
    return answers


def test_raw_to_transformed_known_boundaries():
    assert raw_to_transformed(11) == 30.0
    assert raw_to_transformed(13) == 40.0
    assert raw_to_transformed(17) == 60.0
    assert raw_to_transformed(5) == 0.0
    assert raw_to_transformed(25) == 100.0


def test_raw_to_transformed_rejects_non_positive_item_count():
    with pytest.raises(ValueError):
        raw_to_transformed(10, item_count=0)


def test_pathological_boundary_exactly_40_is_yes():
    result = score_ccmq(_all_answers({"qi_xu": 13}))
    assert result.scores["qi_xu"].transformed_score == 40.0
    assert result.scores["qi_xu"].verdict == VERDICT_YES
    assert result.primary == "qi_xu"


def test_pathological_boundary_exactly_30_is_leaning_yes():
    result = score_ccmq(_all_answers({"qi_xu": 11}))
    assert result.scores["qi_xu"].transformed_score == 30.0
    assert result.scores["qi_xu"].verdict == VERDICT_LEANING_YES
    # "倾向是" alone (nothing reaches "是") must not be promoted to primary.
    assert result.primary is None
    assert result.secondary == ("qi_xu",)


def test_pathological_just_below_30_is_no():
    result = score_ccmq(_all_answers({"qi_xu": 10}))
    assert result.scores["qi_xu"].verdict == VERDICT_NO
    assert result.primary is None
    assert result.secondary == ()


def test_multiple_constitutions_above_40_are_all_detected_as_candidates():
    """体质夹杂：多个体质同时 >=40 分是正常结果，不能只保留最高分那个。"""
    result = score_ccmq(_all_answers({"qi_xu": 20, "yang_xu": 15, "tan_shi": 13}))
    assert result.scores["qi_xu"].verdict == VERDICT_YES
    assert result.scores["yang_xu"].verdict == VERDICT_YES
    assert result.scores["tan_shi"].verdict == VERDICT_YES
    # Highest score becomes primary, the rest land in secondary (order: score desc).
    assert result.primary == "qi_xu"
    assert result.secondary == ("yang_xu", "tan_shi")


def test_leaning_yes_enters_secondary_not_dropped():
    result = score_ccmq(_all_answers({"qi_xu": 20, "yang_xu": 11}))
    assert result.scores["yang_xu"].verdict == VERDICT_LEANING_YES
    assert result.primary == "qi_xu"
    assert "yang_xu" in result.secondary


def test_ping_he_yes_requires_others_all_below_30():
    answers = _all_answers({PING_HE: 17})  # all others default to raw=5 -> transformed 0
    result = score_ccmq(answers)
    assert result.scores[PING_HE].transformed_score == 60.0
    assert result.scores[PING_HE].verdict == VERDICT_YES
    assert result.primary == PING_HE
    assert result.secondary == ()


def test_ping_he_basically_yes_when_one_other_between_30_and_40():
    answers = _all_answers({PING_HE: 17, "qi_xu": 11})  # qi_xu transformed = 30 (leaning)
    result = score_ccmq(answers)
    assert result.scores[PING_HE].verdict == VERDICT_BASICALLY_YES
    assert result.primary == PING_HE
    assert "qi_xu" in result.secondary


def test_ping_he_no_when_another_constitution_reaches_40():
    answers = _all_answers({PING_HE: 17, "qi_xu": 13})  # qi_xu transformed = 40 (yes)
    result = score_ccmq(answers)
    assert result.scores[PING_HE].verdict == VERDICT_NO
    assert result.scores["qi_xu"].verdict == VERDICT_YES
    assert result.primary == "qi_xu"


def test_ping_he_no_when_score_below_60():
    answers = _all_answers({PING_HE: 16})  # transformed = 55
    result = score_ccmq(answers)
    assert result.scores[PING_HE].verdict == VERDICT_NO


def test_all_no_yields_no_primary_or_secondary():
    result = score_ccmq(_all_answers({}))  # everything at raw=5 -> transformed 0
    assert result.primary is None
    assert result.secondary == ()
    assert all(s.verdict == VERDICT_NO for s in result.scores.values())


def test_missing_constitution_raises():
    answers = _all_answers({})
    del answers["qi_xu"]
    with pytest.raises(ValueError, match="missing"):
        score_ccmq(answers)


def test_unknown_constitution_key_raises():
    answers = _all_answers({})
    answers["not_a_real_constitution"] = [3, 3, 3, 3, 3]
    with pytest.raises(ValueError, match="unknown"):
        score_ccmq(answers)


def test_wrong_item_count_raises():
    answers = _all_answers({})
    answers["qi_xu"] = [3, 3, 3]
    with pytest.raises(ValueError, match="expected 5"):
        score_ccmq(answers)


def test_item_score_out_of_range_raises():
    answers = _all_answers({})
    answers["qi_xu"] = [1, 2, 3, 4, 6]
    with pytest.raises(ValueError, match="out of range"):
        score_ccmq(answers)


def test_item_score_zero_raises():
    answers = _all_answers({})
    answers["qi_xu"] = [0, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="out of range"):
        score_ccmq(answers)
