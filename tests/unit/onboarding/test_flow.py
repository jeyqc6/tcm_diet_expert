"""
测试目标：backend/onboarding/flow.py
——§11.2 对话步骤的确定性状态机：触发条件、过敏原/偏好抽取、city→timezone
建议+显式确认(D30 不静默写入)、体质自述 vs CCMQ 问卷两条路径、结果写回字段。
对应实现：backend/onboarding/flow.py
"""
from __future__ import annotations

import pytest

from backend.onboarding.ccmq_scoring import CONSTITUTIONS, ITEMS_PER_CONSTITUTION
from backend.onboarding.flow import (
    ITEM_TEXTS,
    OnboardingResult,
    OnboardingStep,
    _parse_likert,
    advance_chat_onboarding,
    apply_answer,
    should_trigger,
    start_onboarding,
)
from backend.onboarding.session_store import InMemoryOnboardingSessionStore


class _FakeProfile:
    def __init__(self, constitution=None, allergens=(), constitution_source=None, onboarding_done=False):
        self.constitution = constitution
        self.allergens = allergens
        self.constitution_source = constitution_source
        self.onboarding_done = onboarding_done


_LIKERT_STEM_WORDS = ("没有", "很少", "有时", "经常", "总是")


def test_item_texts_match_scoring_shape():
    assert set(ITEM_TEXTS) == set(CONSTITUTIONS)
    for c in CONSTITUTIONS:
        assert len(ITEM_TEXTS[c]) == ITEMS_PER_CONSTITUTION


def test_item_texts_avoid_likert_option_words():
    """Stems must not reuse the answer scale, or 「经常」 reads as 「经常很少疲惫」."""
    for constitution, items in ITEM_TEXTS.items():
        for text in items:
            for word in _LIKERT_STEM_WORDS:
                assert word not in text, f"{constitution}: {text!r} contains {word!r}"


def test_should_trigger_when_no_profile():
    assert should_trigger(None) is True


def test_should_trigger_for_create_user_stub():
    """POST /api/users inserts user_id + display_name only — still first conversation."""
    assert should_trigger(_FakeProfile()) is True


def test_should_not_trigger_when_onboarding_done_even_if_empty():
    """Skip-all / empty constitution must not hide a finished intro, or re-ask."""
    assert should_trigger(_FakeProfile(onboarding_done=True)) is False


def test_should_trigger_when_constitution_known_but_intro_not_done():
    """Having a constitution (scanner / PATCH) is not 'already onboarded'."""
    assert should_trigger(_FakeProfile(constitution="qi_xu")) is True


def test_should_trigger_when_only_allergens_without_intro():
    assert should_trigger(_FakeProfile(allergens=("花生",))) is True


def test_start_onboarding_returns_allergens_step():
    step = start_onboarding()
    assert isinstance(step, OnboardingStep)
    assert step.step_id == "allergens"


def test_allergens_step_extracts_list():
    step = apply_answer("allergens", "花生、虾，芒果", {})
    assert step.step_id == "preferences"
    assert step.state["allergens"] == ["花生", "虾", "芒果"]


def test_allergens_step_recognizes_negation():
    step = apply_answer("allergens", "没有", {})
    assert step.state["allergens"] == []


def test_preferences_step_stores_notes():
    step = apply_answer("preferences", "不吃香菜", {"allergens": []})
    assert step.step_id == "city"
    assert step.state["preferences"] == {"notes": "不吃香菜"}


def test_preferences_step_negation_yields_empty_dict():
    step = apply_answer("preferences", "没有", {"allergens": []})
    assert step.state["preferences"] == {}


def test_city_step_suggests_known_timezone():
    step = apply_answer("city", "上海", {})
    assert step.step_id == "timezone_confirm"
    assert step.state["timezone_suggested"] == "Asia/Shanghai"
    assert "Asia/Shanghai" in step.prompt


def test_city_step_unknown_city_no_suggestion():
    step = apply_answer("city", "无名小镇", {})
    assert step.step_id == "timezone_confirm"
    assert step.state["timezone_suggested"] is None


def test_timezone_confirm_accepts_suggestion():
    state = {"timezone_suggested": "Asia/Shanghai"}
    step = apply_answer("timezone_confirm", "对", state)
    assert step.step_id == "constitution_known"
    assert step.state["timezone"] == "Asia/Shanghai"


def test_timezone_confirm_skip_leaves_timezone_none():
    state = {"timezone_suggested": "Asia/Shanghai"}
    step = apply_answer("timezone_confirm", "跳过", state)
    assert step.state["timezone"] is None


def test_timezone_confirm_override_with_explicit_iana_name():
    state = {"timezone_suggested": "Asia/Shanghai"}
    step = apply_answer("timezone_confirm", "America/New_York", state)
    assert step.state["timezone"] == "America/New_York"


def test_timezone_confirm_override_with_another_city_name():
    state = {"timezone_suggested": "Asia/Shanghai"}
    step = apply_answer("timezone_confirm", "纽约", state)
    assert step.state["timezone"] == "America/New_York"


def test_constitution_known_self_reported_finishes_immediately():
    state = {"allergens": ["花生"], "preferences": {}, "city": "上海", "timezone": "Asia/Shanghai"}
    result = apply_answer("constitution_known", "我是气虚质", state)
    assert isinstance(result, OnboardingResult)
    assert result.profile_updates["constitution"] == "qi_xu"
    assert result.profile_updates["constitution_source"] == "self_reported"
    assert result.profile_updates["onboarding_done"] is True
    assert result.profile_updates["constitution_secondary"] == []
    assert result.profile_updates["allergens"] == ["花生"]


def test_constitution_known_unsure_starts_ccmq_batch():
    step = apply_answer("constitution_known", "不知道", {})
    assert step.step_id == "ccmq_batch"
    assert step.state["ccmq_flat_index"] == 0
    assert "1-5" in step.prompt
    assert "4比较符合" in step.prompt
    assert "您平时精力充沛吗？" in step.prompt


def test_parse_likert_recognizes_words_and_digits():
    assert _parse_likert("没有") == 1
    assert _parse_likert("很少") == 2
    assert _parse_likert("有时") == 3
    assert _parse_likert("经常") == 4
    assert _parse_likert("总是") == 5
    assert _parse_likert("不确定") == 3
    assert _parse_likert("4") == 4
    assert _parse_likert("完全不像") == 1
    assert _parse_likert("不太像") == 2
    assert _parse_likert("有些像") == 3
    assert _parse_likert("比较符合") == 4
    assert _parse_likert("非常符合") == 5
    assert _parse_likert("４") == 4


def test_parse_likert_rejects_unrecognized_text():
    with pytest.raises(ValueError):
        _parse_likert("随便啦")


def test_ccmq_batch_wrong_count_reasks_same_batch():
    state = {"ccmq_flat_index": 0, "ccmq_answers": {c: [] for c in CONSTITUTIONS}}
    step = apply_answer("ccmq_batch", "没有，很少", state)  # only 2 answers, batch size is 3
    assert step.step_id == "ccmq_batch"
    assert step.state["ccmq_flat_index"] == 0  # unchanged, re-asks


def test_ccmq_batch_unrecognized_word_reasks_same_batch():
    state = {"ccmq_flat_index": 0, "ccmq_answers": {c: [] for c in CONSTITUTIONS}}
    step = apply_answer("ccmq_batch", "没有，随便，有时", state)
    assert step.step_id == "ccmq_batch"
    assert step.state["ccmq_flat_index"] == 0


def test_ccmq_batch_advances_and_records_scores():
    state = {"ccmq_flat_index": 0, "ccmq_answers": {c: [] for c in CONSTITUTIONS}}
    step = apply_answer("ccmq_batch", "4,2,5", state)
    assert step.step_id == "ccmq_batch"
    assert step.state["ccmq_flat_index"] == 3
    assert step.state["ccmq_answers"]["ping_he"] == [4, 2, 5]


def _complete_ccmq_all_low(state):
    """Drive every remaining batch with '没有' (score=1) answers until done."""
    from backend.onboarding.flow import _CCMQ_BATCH_SIZE, _TOTAL_ITEMS

    result = None
    while state.get("ccmq_flat_index", 0) < _TOTAL_ITEMS:
        idx = state["ccmq_flat_index"]
        remaining = min(_CCMQ_BATCH_SIZE, _TOTAL_ITEMS - idx)
        answer = "，".join(["没有"] * remaining)
        result = apply_answer("ccmq_batch", answer, state)
        state = result.state
    return result


def test_ccmq_all_low_scores_reach_confirm_with_no_strong_candidate():
    state = {"ccmq_flat_index": 0, "ccmq_answers": {c: [] for c in CONSTITUTIONS}}
    step = _complete_ccmq_all_low(state)
    assert step.step_id == "constitution_confirm"
    assert step.state["ccmq_primary"] is None
    assert "未见明显偏颇体质倾向" in step.prompt


def test_ccmq_high_scores_for_one_constitution_becomes_primary_candidate():
    from backend.onboarding.flow import _CCMQ_BATCH_SIZE, _FLAT_ITEMS, _TOTAL_ITEMS

    state = {"ccmq_flat_index": 0, "ccmq_answers": {c: [] for c in CONSTITUTIONS}}
    result = None
    while state.get("ccmq_flat_index", 0) < _TOTAL_ITEMS:
        idx = state["ccmq_flat_index"]
        batch = _FLAT_ITEMS[idx : idx + _CCMQ_BATCH_SIZE]
        words = ["总是" if c == "qi_xu" else "没有" for c, _ in batch]
        result = apply_answer("ccmq_batch", "，".join(words), state)
        state = result.state
    assert result.step_id == "constitution_confirm"
    assert result.state["ccmq_primary"] == "qi_xu"


def test_constitution_confirm_accept_writes_ccmq_result():
    state = {
        "allergens": [], "preferences": {}, "city": None, "timezone": None,
        "ccmq_primary": "qi_xu", "ccmq_secondary": ["yang_xu"],
    }
    result = apply_answer("constitution_confirm", "确认", state)
    assert result.profile_updates["constitution"] == "qi_xu"
    assert result.profile_updates["constitution_secondary"] == ["yang_xu"]
    assert result.profile_updates["constitution_source"] == "ccmq_computed"
    assert result.profile_updates["onboarding_done"] is True


def test_constitution_confirm_skip_leaves_constitution_unset():
    state = {"allergens": [], "preferences": {}, "city": None, "timezone": None,
             "ccmq_primary": "qi_xu", "ccmq_secondary": []}
    result = apply_answer("constitution_confirm", "跳过", state)
    assert result.profile_updates["constitution"] is None
    assert result.profile_updates["constitution_source"] == "unconfirmed"
    assert result.profile_updates["onboarding_done"] is True


def test_constitution_confirm_override_with_different_constitution():
    state = {"allergens": [], "preferences": {}, "city": None, "timezone": None,
             "ccmq_primary": "qi_xu", "ccmq_secondary": []}
    result = apply_answer("constitution_confirm", "其实是阳虚质", state)
    assert result.profile_updates["constitution"] == "yang_xu"
    assert result.profile_updates["constitution_source"] == "self_reported"


def test_unknown_step_id_raises():
    with pytest.raises(ValueError):
        apply_answer("not_a_real_step", "x", {})


def test_allergens_skip_is_empty_list_not_an_allergen_named_skip():
    step = apply_answer("allergens", "跳过", {})
    assert step.step_id == "preferences"
    assert step.state["allergens"] == []


def test_constitution_known_skip_finishes_without_starting_ccmq():
    result = apply_answer("constitution_known", "跳过", {"allergens": []})
    assert isinstance(result, OnboardingResult)
    assert result.profile_updates["constitution"] is None
    assert result.profile_updates["constitution_source"] == "unconfirmed"
    assert result.profile_updates["onboarding_done"] is True


def test_abort_all_finishes_from_any_step():
    result = apply_answer("preferences", "全部跳过", {"allergens": ["花生"]})
    assert isinstance(result, OnboardingResult)
    assert result.profile_updates["allergens"] == ["花生"]
    assert result.profile_updates["constitution"] is None
    assert result.profile_updates["constitution_source"] == "unconfirmed"
    assert result.profile_updates["onboarding_done"] is True


def test_advance_chat_starts_on_first_message_without_consuming_it():
    store = InMemoryOnboardingSessionStore()
    turn = advance_chat_onboarding("今天吃什么", None, store, user_id="u1")
    assert turn is not None
    assert turn.started is True
    assert turn.done is False
    assert "过敏" in turn.prompt
    # First message was a real question, not an allergen answer.
    assert store.get("u1").step_id == "allergens"
    assert store.get("u1").state == {}


def test_advance_chat_second_message_is_the_answer():
    store = InMemoryOnboardingSessionStore()
    advance_chat_onboarding("今天吃什么", None, store, user_id="u1")
    turn = advance_chat_onboarding("花生", None, store, user_id="u1")
    assert turn.started is False
    assert turn.done is False
    assert store.get("u1").step_id == "preferences"
    assert store.get("u1").state["allergens"] == ["花生"]


def test_advance_chat_abort_clears_store_and_does_not_retrigger_with_profile():
    store = InMemoryOnboardingSessionStore()
    advance_chat_onboarding("今天吃什么", None, store, user_id="u1")
    turn = advance_chat_onboarding("全部跳过", None, store, user_id="u1")
    assert turn.done is True
    assert turn.profile_updates is not None
    assert store.get("u1") is None
    later = advance_chat_onboarding(
        "红枣性味",
        _FakeProfile(onboarding_done=True),
        store,
        user_id="u1",
    )
    assert later is None


def test_item_texts_en_match_scoring_shape():
    from backend.onboarding.flow import ITEM_TEXTS_EN

    assert set(ITEM_TEXTS_EN) == set(CONSTITUTIONS)
    for c in CONSTITUTIONS:
        assert len(ITEM_TEXTS_EN[c]) == ITEMS_PER_CONSTITUTION


def test_english_skip_and_confirm_aliases():
    step = apply_answer("allergens", "skip", {}, locale="en")
    assert step.step_id == "preferences"
    assert step.state["allergens"] == []
    assert "taste" in step.prompt.lower() or "prefer" in step.prompt.lower()

    result = apply_answer("preferences", "skip all", {"allergens": ["peanut"]}, locale="en")
    assert isinstance(result, OnboardingResult)
    assert result.profile_updates["onboarding_done"] is True
    assert result.profile_updates["locale"] == "en"
    assert result.profile_updates["allergens"] == ["peanut"]

    tz = apply_answer("timezone_confirm", "yes", {"timezone_suggested": "Asia/Shanghai"}, locale="en")
    assert tz.state["timezone"] == "Asia/Shanghai"

    confirm = apply_answer(
        "constitution_confirm",
        "confirm",
        {"ccmq_primary": "qi_xu", "ccmq_secondary": []},
        locale="en",
    )
    assert confirm.profile_updates["constitution"] == "qi_xu"
    assert confirm.profile_updates["constitution_source"] == "ccmq_computed"


def test_parse_likert_english_aliases():
    assert _parse_likert("never") == 1
    assert _parse_likert("rarely") == 2
    assert _parse_likert("sometimes") == 3
    assert _parse_likert("often") == 4
    assert _parse_likert("always") == 5
    assert _parse_likert("unsure") == 3
    assert _parse_likert("unknown") == 3


def test_start_onboarding_locale_en_is_english():
    step = start_onboarding(locale="en")
    assert "skip" in step.prompt.lower()
    assert "过敏" not in step.prompt


def test_constitution_matches_english_display_name():
    result = apply_answer("constitution_known", "Qi deficiency", {}, locale="en")
    assert isinstance(result, OnboardingResult)
    assert result.profile_updates["constitution"] == "qi_xu"


def test_advance_chat_starts_for_create_user_stub():
    store = InMemoryOnboardingSessionStore()
    turn = advance_chat_onboarding("今天吃什么", _FakeProfile(), store, user_id="u1")
    assert turn is not None
    assert turn.started is True


def test_advance_chat_does_not_start_when_already_offered():
    store = InMemoryOnboardingSessionStore()
    turn = advance_chat_onboarding(
        "今天吃什么",
        _FakeProfile(onboarding_done=True),
        store,
        user_id="u1",
    )
    assert turn is None
