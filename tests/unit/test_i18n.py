"""Locale helpers: default zh, language instruction only for en, t() lookups."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import ChatRequest, OnboardingAnswerRequest
from backend.i18n import (
    LANGUAGE_INSTRUCTION_EN,
    apply_language_instruction,
    language_instruction,
    meal_type_label,
    normalize_locale,
    t,
)
from backend.onboarding.flow import ITEM_TEXTS, ITEM_TEXTS_EN, start_onboarding


def test_normalize_locale_defaults_and_accepts_zh_en():
    assert normalize_locale(None) == "zh"
    assert normalize_locale("") == "zh"
    assert normalize_locale("zh") == "zh"
    assert normalize_locale("en") == "en"
    assert normalize_locale("EN") == "en"
    assert normalize_locale("fr") == "zh"


def test_language_instruction_present_for_en_empty_for_zh():
    assert language_instruction("en") == LANGUAGE_INSTRUCTION_EN
    assert language_instruction("zh") == ""
    assert language_instruction(None) == ""
    assert "English" in apply_language_instruction("base", "en")
    assert apply_language_instruction("base", "zh") == "base"


def test_t_lookups_zh_and_en():
    assert "跳过" in t("onboarding.start", "zh")
    assert "skip" in t("onboarding.start", "en").lower()
    assert t("missing.key", "en") == "missing.key"
    assert "HTTP" not in t("api.chain_timeout", "zh")
    assert "timed out" in t("api.chain_timeout", "en").lower()
    assert meal_type_label("午餐", "zh") == "午餐"
    assert meal_type_label("午餐", "en") == "lunch"


def test_chat_request_defaults_locale_zh():
    req = ChatRequest(session_id="s1", message="今天吃什么")
    assert req.locale == "zh"


def test_chat_request_accepts_en_and_rejects_other():
    req = ChatRequest(session_id="s1", message="hi", locale="en")
    assert req.locale == "en"
    req = ChatRequest(session_id="s1", message="hi", locale="EN")
    assert req.locale == "en"
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="hi", locale="fr")


def test_onboarding_answer_request_accepts_locale():
    req = OnboardingAnswerRequest(step_id="allergens", answer="none", locale="en")
    assert req.locale == "en"


def test_item_texts_en_has_9x5():
    assert set(ITEM_TEXTS_EN) == set(ITEM_TEXTS)
    for key, stems in ITEM_TEXTS_EN.items():
        assert len(stems) == 5, key
        assert len(ITEM_TEXTS[key]) == 5


def test_start_onboarding_en_prompt_is_english():
    step = start_onboarding(locale="en")
    assert step.step_id == "allergens"
    lowered = step.prompt.lower()
    assert "skip" in lowered
    assert "allerg" in lowered
    assert "过敏" not in step.prompt
