"""
测试目标：过敏原/补剂关键词命中的穷举用例；跨分支触发（本模块本身不区分
分支——它是分支无关的纯文本扫描，"跨分支"这条约束由 api/main.py 在路由判断
之前调用它来保证，见该文件的集成测试）；误报排除（否定语境、无"过敏"declaration
的单纯提及）。
对应实现：backend/memory/critical_fact_scanner.py
覆盖要求：**要求 100%**
"""
from __future__ import annotations

import pytest

from backend.agents.user_context import UserProfileContext
from backend.memory.critical_fact_scanner import (
    ALLERGEN_CATEGORY_KEYWORDS,
    SUPPLEMENT_KEYWORDS,
    AllergenMention,
    CriticalFactScanResult,
    merge_into_profile,
    scan_allergen_mentions,
    scan_critical_facts,
    scan_supplement_mentions,
)

# ---------------------------------------------------------------------------
# 过敏原声明：逐条穷举 ALLERGEN_CATEGORY_KEYWORDS 每个类别的每一个关键词，
# 分别用三种句式验证
# ---------------------------------------------------------------------------

_ALL_ALLERGEN_TERMS = [
    (category, term)
    for category, terms in ALLERGEN_CATEGORY_KEYWORDS.items()
    for term in terms
]


@pytest.mark.parametrize("category,term", _ALL_ALLERGEN_TERMS)
def test_every_allergen_keyword_detected_via_dui_guo_min_pattern(category: str, term: str) -> None:
    mentions = scan_allergen_mentions(f"我对{term}过敏，要注意一下")
    assert any(m.category == category for m in mentions), f"{term} -> {category} 未命中"


@pytest.mark.parametrize("category,term", _ALL_ALLERGEN_TERMS)
def test_every_allergen_keyword_detected_via_bare_guo_min_pattern(category: str, term: str) -> None:
    mentions = scan_allergen_mentions(f"{term}过敏")
    assert any(m.category == category for m in mentions), f"{term} -> {category} 未命中"


@pytest.mark.parametrize("category,term", _ALL_ALLERGEN_TERMS)
def test_every_allergen_keyword_detected_via_guo_min_yuan_shi_pattern(category: str, term: str) -> None:
    mentions = scan_allergen_mentions(f"过敏原是{term}")
    assert any(m.category == category for m in mentions), f"{term} -> {category} 未命中"


def test_empty_text_never_matches() -> None:
    assert scan_allergen_mentions("") == ()


def test_plain_mention_without_allergy_declaration_does_not_match() -> None:
    """单纯提到食材(比如在聊今天吃了什么)不该被当成过敏声明——那会把日常
    对话大量误判成关键事实。"""
    assert scan_allergen_mentions("我今天吃了很多虾") == ()
    assert scan_allergen_mentions("鱼香肉丝真好吃") == ()
    assert scan_allergen_mentions("花生米配啤酒") == ()


@pytest.mark.parametrize("marker", ("没有", "不是", "并不", "不对", "并非"))
def test_every_negation_marker_suppresses_allergy_declaration(marker: str) -> None:
    assert scan_allergen_mentions(f"{marker}对虾过敏") == ()


def test_negation_far_from_declaration_does_not_suppress() -> None:
    long_prefix = "今天天气不错，" * 5
    text = f"{long_prefix}我对虾过敏"
    mentions = scan_allergen_mentions(text)
    assert any(m.category == "甲壳类" for m in mentions)


def test_multiple_categories_in_one_message_all_detected() -> None:
    mentions = scan_allergen_mentions("过敏原是花生和芝麻")
    assert {m.category for m in mentions} == {"花生", "芝麻"}


def test_same_category_only_reported_once() -> None:
    mentions = scan_allergen_mentions("我对虾过敏，对螃蟹也过敏")
    assert len([m for m in mentions if m.category == "甲壳类"]) == 1


def test_mention_matched_term_is_the_literal_word_not_the_category() -> None:
    mentions = scan_allergen_mentions("我对虾过敏")
    assert mentions == (AllergenMention(category="甲壳类", matched_term="虾"),)


# ---------------------------------------------------------------------------
# 英文过敏声明——归一化后落进 user_profile.allergens 的仍然是中文类别名
# (见模块文档"扫描到的是类别而不是原文用词"一节)，这里只测"用户用英文声明
# 过敏，能不能被识别成对应的中文类别"。
# ---------------------------------------------------------------------------


def test_english_allergic_to_pattern_is_detected() -> None:
    mentions = scan_allergen_mentions("I'm allergic to shrimp.")
    assert any(m.category == "甲壳类" and m.matched_term == "shrimp" for m in mentions)


def test_english_bare_allergy_pattern_is_detected() -> None:
    mentions = scan_allergen_mentions("I have a peanut allergy.")
    assert any(m.category == "花生" for m in mentions)


def test_english_allergy_is_pattern_is_detected() -> None:
    mentions = scan_allergen_mentions("My allergy is sesame.")
    assert any(m.category == "芝麻" for m in mentions)


def test_english_allergy_declaration_is_case_insensitive() -> None:
    assert scan_allergen_mentions("I'm ALLERGIC TO Shellfish.") != ()


def test_english_not_negation_suppresses_allergy_declaration() -> None:
    assert scan_allergen_mentions("I am not allergic to shrimp.") == ()


def test_english_contraction_negation_suppresses_allergy_declaration() -> None:
    """"n't"是"isn't"/"aren't"/"doesn't"这类缩写的公共子串——窗口检查的是
    "命中位置前 N 个字符里有没有出现这个子串"，不要求语法完整匹配，见
    _ALLERGY_WINDOW 的注释。"""
    assert scan_allergen_mentions("She isn't allergic to shrimp.") == ()


def test_english_plain_mention_without_allergy_declaration_does_not_match() -> None:
    """单纯提到食材(比如聊今天吃了什么)不该被当成过敏声明——同中文那条同名
    测试的理由。"""
    assert scan_allergen_mentions("I had shrimp for lunch today.") == ()


def test_english_lactose_intolerant_maps_to_dairy_category() -> None:
    mentions = scan_allergen_mentions("I am slightly lactose intolerant.")
    assert any(m.category == "乳制品" and m.matched_term == "lactose" for m in mentions)


def test_english_allergic_to_mango_maps_to_mango_literal() -> None:
    mentions = scan_allergen_mentions("allergic to mangoes")
    assert any(m.category == "芒果" and m.matched_term == "mangoes" for m in mentions)


def test_user_log_health_restriction_message_is_detected() -> None:
    msg = "Please log that I am slightly lactose intolerant and allergic to mangoes."
    result = scan_critical_facts(msg, None)
    assert result.hit is True
    assert "乳制品" in result.new_allergens
    assert "芒果" in result.new_allergens


def test_chinese_lactose_intolerance_maps_to_dairy_category() -> None:
    mentions = scan_allergen_mentions("我有乳糖不耐受")
    assert any(m.category == "乳制品" for m in mentions)


# ---------------------------------------------------------------------------
# 补剂提及：逐条穷举 SUPPLEMENT_KEYWORDS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUPPLEMENT_KEYWORDS)
def test_every_supplement_keyword_detected_via_chi_pattern(name: str) -> None:
    mentions = scan_supplement_mentions(f"我最近在吃{name}")
    assert any(m.name == name for m in mentions), f"{name} 未命中"


@pytest.mark.parametrize("name", SUPPLEMENT_KEYWORDS)
def test_every_supplement_keyword_detected_via_fuyong_pattern(name: str) -> None:
    mentions = scan_supplement_mentions(f"正在服用{name}")
    assert any(m.name == name for m in mentions), f"{name} 未命中"


def test_supplement_longer_keyword_wins_over_shorter_substring() -> None:
    """"维生素D"应该整体命中,不该被切成"维生素"+"D"两半(维生素 是 维生素D 的
    子串,长词优先排列保证这一点)。"""
    mentions = scan_supplement_mentions("我在吃维生素D")
    names = {m.name for m in mentions}
    assert "维生素D" in names
    assert "维生素" not in names


def test_empty_text_never_matches_supplements() -> None:
    assert scan_supplement_mentions("") == ()


def test_plain_mention_without_intake_verb_does_not_match() -> None:
    assert scan_supplement_mentions("鱼油对心血管有好处") == ()


@pytest.mark.parametrize("marker", ("没", "不再", "已经不", "停了", "戒掉", "停用"))
def test_every_supplement_negation_marker_suppresses_match(marker: str) -> None:
    assert scan_supplement_mentions(f"{marker}吃鱼油") == ()


def test_same_supplement_only_reported_once() -> None:
    mentions = scan_supplement_mentions("我在吃鱼油，每天都吃鱼油")
    assert len(mentions) == 1


# ---------------------------------------------------------------------------
# 英文补剂提及
# ---------------------------------------------------------------------------


def test_english_taking_pattern_is_detected() -> None:
    mentions = scan_supplement_mentions("I'm taking fish oil every day.")
    assert any(m.name == "fish oil" for m in mentions)


def test_english_take_pattern_is_detected() -> None:
    mentions = scan_supplement_mentions("I take vitamin D in winter.")
    assert any(m.name == "vitamin D" for m in mentions)


def test_english_supplement_pattern_is_case_insensitive() -> None:
    assert scan_supplement_mentions("I'm Taking Fish Oil.") != ()


def test_english_plain_mention_without_intake_verb_does_not_match() -> None:
    assert scan_supplement_mentions("Fish oil is good for your heart.") == ()


@pytest.mark.parametrize("marker", ("not", "n't", "stopped", "quit"))
def test_english_supplement_negation_marker_suppresses_match(marker: str) -> None:
    text = {
        "not": "I am not taking fish oil.",
        "n't": "I don't take fish oil.",
        "stopped": "I stopped taking fish oil.",
        "quit": "I quit taking fish oil.",
    }[marker]
    assert scan_supplement_mentions(text) == ()


def test_english_on_is_not_a_supplement_trigger_word() -> None:
    """故意不把"on"当触发动词——"on"太常见，会把"focused on calcium
    absorption"这种完全无关的句子也误判成补剂提及，见 _build_supplement_pattern
    的注释。"""
    assert scan_supplement_mentions("This meal plan is focused on calcium absorption.") == ()


# ---------------------------------------------------------------------------
# scan_critical_facts()：相对现有画像的增量
# ---------------------------------------------------------------------------


def test_scan_critical_facts_no_profile_reports_everything_as_new() -> None:
    result = scan_critical_facts("我对虾过敏，在吃鱼油", None)
    assert result.new_allergens == ("甲壳类",)
    assert result.new_supplements == ("鱼油",)
    assert result.hit is True


def test_scan_critical_facts_already_known_allergen_not_reported_again() -> None:
    profile = UserProfileContext(user_id="u1", allergens=("甲壳类",))
    result = scan_critical_facts("我对虾过敏", profile)
    assert result.new_allergens == ()
    assert result.hit is False


def test_scan_critical_facts_already_known_supplement_not_reported_again() -> None:
    profile = UserProfileContext(user_id="u1", supplements=({"name": "鱼油", "dose": None},))
    result = scan_critical_facts("我在吃鱼油", profile)
    assert result.new_supplements == ()
    assert result.hit is False


def test_scan_critical_facts_no_hit_returns_empty_result() -> None:
    result = scan_critical_facts("今天该吃什么好呢", None)
    assert result == CriticalFactScanResult()
    assert result.hit is False


def test_scan_critical_facts_partial_overlap_only_reports_the_new_one() -> None:
    profile = UserProfileContext(user_id="u1", allergens=("花生",))
    result = scan_critical_facts("我对花生过敏，对芝麻也过敏", profile)
    assert result.new_allergens == ("芝麻",)


# ---------------------------------------------------------------------------
# merge_into_profile()：写入 payload 是完整合并后的列表(UPSERT 覆盖语义)，
# 不是只有新增的那部分
# ---------------------------------------------------------------------------


def test_merge_into_profile_no_existing_profile_creates_new_context() -> None:
    result = CriticalFactScanResult(new_allergens=("甲壳类",), new_supplements=("鱼油",))
    payload, updated = merge_into_profile(result, None, user_id="u1")
    assert payload == {"allergens": ["甲壳类"], "supplements": [{"name": "鱼油", "dose": None}]}
    assert updated.user_id == "u1"
    assert updated.allergens == ("甲壳类",)
    assert updated.supplements == ({"name": "鱼油", "dose": None},)


def test_merge_into_profile_preserves_existing_allergens_not_just_new_ones() -> None:
    """`write_memory(critical)` 是整列覆盖，不是数组追加——payload 必须带上
    已经记过的过敏原，不能只传新增的这一个，否则会把旧的覆盖掉。"""
    profile = UserProfileContext(user_id="u1", allergens=("花生",))
    result = CriticalFactScanResult(new_allergens=("甲壳类",))
    payload, updated = merge_into_profile(result, profile)
    assert set(payload["allergens"]) == {"花生", "甲壳类"}
    assert set(updated.allergens) == {"花生", "甲壳类"}


def test_merge_into_profile_preserves_existing_supplements_not_just_new_ones() -> None:
    profile = UserProfileContext(user_id="u1", supplements=({"name": "钙片", "dose": "600mg"},))
    result = CriticalFactScanResult(new_supplements=("鱼油",))
    payload, updated = merge_into_profile(result, profile)
    assert {"name": "钙片", "dose": "600mg"} in payload["supplements"]
    assert {"name": "鱼油", "dose": None} in payload["supplements"]
    assert updated.supplements == ({"name": "钙片", "dose": "600mg"}, {"name": "鱼油", "dose": None})


def test_merge_into_profile_omits_allergens_key_when_nothing_new() -> None:
    """只带真的有新增的那个字段——不该无谓地把 allergens 也塞进 payload 里
    (即便值和数据库里已有的一样，多写一次也不是本函数该做的事，调用方按
    result.hit 决定要不要调这个函数)。"""
    profile = UserProfileContext(user_id="u1", allergens=("花生",))
    result = CriticalFactScanResult(new_supplements=("鱼油",))
    payload, _ = merge_into_profile(result, profile)
    assert "allergens" not in payload
    assert "supplements" in payload


def test_merge_into_profile_preserves_other_profile_fields() -> None:
    profile = UserProfileContext(user_id="u1", constitution="qi_xu", city="上海")
    result = CriticalFactScanResult(new_allergens=("甲壳类",))
    _, updated = merge_into_profile(result, profile)
    assert updated.constitution == "qi_xu"
    assert updated.city == "上海"


def test_merge_into_profile_merges_preferences_lists() -> None:
    profile = UserProfileContext(
        user_id="u1",
        preferences={"忌口": ["花生"]},
    )
    result = CriticalFactScanResult(new_preferences={"忌口": ["香菜"]})
    payload, updated = merge_into_profile(result, profile)
    assert payload["preferences"] == {"忌口": ["花生", "香菜"]}
    assert updated.preferences == {"忌口": ["花生", "香菜"]}
