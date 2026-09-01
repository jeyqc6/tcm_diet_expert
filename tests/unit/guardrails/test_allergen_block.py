"""
测试目标：过敏原命中的穷举用例，逐条穷举而非抽样
对应实现：backend/guardrails/output_filters.py
覆盖要求：**要求 100%**
"""
from __future__ import annotations

import pytest

from backend.guardrails.output_filters import (
    HIDDEN_ALLERGEN_SOURCES,
    NEGATION_MARKERS,
    check_allergens,
    filter_output,
)

# ---------------------------------------------------------------------------
# 直接命中：文本里出现过敏原类别本身的字面词
# ---------------------------------------------------------------------------


def test_direct_allergen_mention_is_blocked() -> None:
    findings = check_allergens("这道菜含花生，注意避开。", user_allergens=["花生"])
    assert len(findings) == 1
    assert findings[0].matched_term == "花生"
    assert findings[0].allergen == "花生"


def test_no_allergen_configured_never_blocks() -> None:
    assert check_allergens("这道菜含花生。", user_allergens=[]) == []
    assert check_allergens("这道菜含花生。", user_allergens=None) == []


def test_empty_text_never_blocks() -> None:
    assert check_allergens("", user_allergens=["花生"]) == []


def test_allergen_not_present_in_text_does_not_block() -> None:
    assert check_allergens("这道菜含鸡蛋。", user_allergens=["花生"]) == []


def test_multiple_configured_allergens_only_matched_ones_returned() -> None:
    findings = check_allergens("这道菜含花生和鸡蛋。", user_allergens=["花生", "牛奶"])
    assert {f.allergen for f in findings} == {"花生"}


# ---------------------------------------------------------------------------
# 隐藏来源命中：逐条穷举 HIDDEN_ALLERGEN_SOURCES 里的每一条映射
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term,category", sorted(HIDDEN_ALLERGEN_SOURCES.items()))
def test_every_hidden_allergen_source_is_detected(term: str, category: str) -> None:
    text = f"关键调味：{term}，其余家常处理。"
    findings = check_allergens(text, user_allergens=[category])
    assert any(f.matched_term == term and f.allergen == category for f in findings)


def test_hidden_source_word_without_matching_user_allergen_does_not_block() -> None:
    # 用户过敏原是"花生"，文本含"蚝油"(→甲壳类)，两者不相关，不应命中
    findings = check_allergens("加一勺蚝油提鲜。", user_allergens=["花生"])
    assert findings == []


def test_direct_and_hidden_source_can_both_fire() -> None:
    findings = check_allergens(
        "含蚝油，另外也直接加了花生碎。", user_allergens=["甲壳类", "花生"]
    )
    matched_allergens = {f.allergen for f in findings}
    assert matched_allergens == {"甲壳类", "花生"}


def test_hidden_allergen_sources_map_is_non_empty_seed_list() -> None:
    """蚝油→甲壳类是 verification_checklist.md/recipe_and_shopping_list.md
    两份 Skill 反复举的例子，必须在种子表里。"""
    assert HIDDEN_ALLERGEN_SOURCES.get("蚝油") == "甲壳类"
    assert len(HIDDEN_ALLERGEN_SOURCES) >= 5


# ---------------------------------------------------------------------------
# filter_output() 聚合结果：blocked 属性
# ---------------------------------------------------------------------------


def test_filter_output_blocked_true_when_allergen_hit() -> None:
    result = filter_output("含花生碎。", user_allergens=["花生"])
    assert result.blocked is True
    assert len(result.allergens) == 1


def test_filter_output_blocked_false_when_clean() -> None:
    result = filter_output("阳虚质应少食生冷。", user_allergens=["花生"])
    assert result.blocked is False
    assert result.allergens == []
    assert result.diagnostic is None


# ---------------------------------------------------------------------------
# 否定语境：模型主动声明"不含/已避开"不应该被当成命中——逐条穷举
# NEGATION_MARKERS 每一个否定词
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", NEGATION_MARKERS)
def test_every_negation_marker_suppresses_direct_match(marker: str) -> None:
    text = f"蛋白质来源已换成{marker}甲壳类的选项。"
    assert check_allergens(text, user_allergens=["甲壳类"]) == []


@pytest.mark.parametrize("marker", NEGATION_MARKERS)
def test_every_negation_marker_suppresses_hidden_source_match(marker: str) -> None:
    text = f"这道菜{marker}蚝油。"
    assert check_allergens(text, user_allergens=["甲壳类"]) == []


def test_negation_only_suppresses_the_negated_occurrence_not_others() -> None:
    """"已经避开甲壳类了，但是我还是加了蚝油"——"蚝油"这次出现前面没有否定词，
    仍然应该被正常命中；不能因为句子里出现过一次否定词，就连着别的、真实存在
    的命中也一起放过。"""
    findings = check_allergens("已经避开甲壳类了，但是我还是加了蚝油。", user_allergens=["甲壳类"])
    assert len(findings) == 1
    assert findings[0].matched_term == "蚝油"


def test_negation_marker_far_from_match_does_not_suppress() -> None:
    """否定词离命中位置太远(超出窗口)不应该压制命中——不是"这句话里出现过
    某个否定词"就全句免疫。"""
    long_prefix = "这道菜口味丰富，" * 5  # 远远超过 _NEGATION_WINDOW
    text = f"不推荐，{long_prefix}含花生碎。"
    findings = check_allergens(text, user_allergens=["花生"])
    assert len(findings) == 1


def test_unrelated_word_containing_wu_does_not_suppress_match() -> None:
    """"无花果"这类词本身带"无"字——之所以不把单字"无"列进否定词表，就是为了
    避免这种词把后面真实的过敏原命中误判成"被否定过"。"""
    findings = check_allergens("无花果和芝麻很搭配。", user_allergens=["芝麻"])
    assert len(findings) == 1
    assert findings[0].matched_term == "芝麻"


# ---------------------------------------------------------------------------
# Avoidance section headers: listing allergens under "今天避开" is not a hit
# ---------------------------------------------------------------------------


def test_today_avoid_list_is_not_an_allergen_hit() -> None:
    """Live false positive: the model listed allergens under avoidance copy."""
    text = (
        "硬约束：全程不出现花生、花生油、花生酱、虾、虾皮、虾油、香菜。\n"
        "今天避开：花生类、虾类、香菜。"
    )
    assert check_allergens(text, user_allergens=["花生", "虾"]) == []


def test_avoidance_section_also_covers_hidden_sources() -> None:
    assert check_allergens("今天避开：蚝油、虾皮。", user_allergens=["甲壳类"]) == []


def test_avoidance_section_does_not_suppress_later_recommendation() -> None:
    """A header must not immunize the rest of the reply."""
    text = "今天避开：花生、虾。\n\n午餐推荐：宫保鸡丁配花生碎。"
    findings = check_allergens(text, user_allergens=["花生"])
    assert len(findings) == 1
    assert findings[0].matched_term == "花生"


# ---------------------------------------------------------------------------
# 英文回复(i18n.py locale=en)：`user_profile.allergens` 永远存中文类别名，
# 英文文本要靠 HIDDEN_ALLERGEN_SOURCES 里新补的英文条目才能命中——见
# output_filters.py 2026-09-01 那段注释。
# ---------------------------------------------------------------------------


def test_english_direct_allergen_word_is_blocked() -> None:
    """用户过敏原是中文类别"甲壳类"，回复是英文——"shellfish"要能命中，
    这是英文回复唯一能触发过敏原硬阻断的路径(直接命中层比对的是中文类别名
    本身，英文文本里不会出现"甲壳类"这几个字)。"""
    findings = check_allergens("This dish contains shellfish, please avoid.", user_allergens=["甲壳类"])
    assert len(findings) == 1
    assert findings[0].matched_term == "shellfish"
    assert findings[0].allergen == "甲壳类"


def test_english_hidden_source_word_is_blocked() -> None:
    findings = check_allergens("Season with a spoon of oyster sauce.", user_allergens=["甲壳类"])
    assert len(findings) == 1
    assert findings[0].matched_term == "oyster sauce"


def test_english_allergen_match_is_case_insensitive() -> None:
    """英文句首/标题惯例会大写首字母——"Shellfish"要和小写表里的
    "shellfish"一样能命中。"""
    findings = check_allergens("Shellfish is one of the ingredients.", user_allergens=["甲壳类"])
    assert len(findings) == 1


@pytest.mark.parametrize(
    "marker", ("not contain", "not include", "free of", "without", "excluding", "excludes", "avoided", "removed", "not used", "omitted"),
)
def test_english_negation_marker_suppresses_direct_match(marker: str) -> None:
    text = f"This recipe is {marker} shellfish."
    assert check_allergens(text, user_allergens=["甲壳类"]) == []


def test_english_negation_is_case_insensitive() -> None:
    assert check_allergens("Without shellfish in this dish.", user_allergens=["甲壳类"]) == []


def test_english_avoidance_section_is_not_a_hit() -> None:
    text = "To avoid: peanuts, shrimp.\nEnjoy the rest of the meal freely."
    assert check_allergens(text, user_allergens=["花生", "甲壳类"]) == []


def test_english_avoidance_section_does_not_suppress_later_recommendation() -> None:
    """"peanut" 是 "peanuts" 的子串，两个都在表里，命中"crushed peanuts"时
    两条各自都会算一次——这和中文表里"花生"/"花生油"同时命中"加了花生油"是
    同一种情况(HIDDEN_ALLERGEN_SOURCES 本来就允许词条互为子串，下游只关心
    命中的 allergen 集合，不关心条数，见 `test_direct_and_hidden_source_can_both_fire`
    的同一个断言写法)，这里跟着用集合断言，不断言具体条数。"""
    text = "To avoid: peanuts.\n\nLunch: Kung Pao chicken with crushed peanuts."
    findings = check_allergens(text, user_allergens=["花生"])
    assert findings  # 避开小节没有连坐后面真实的命中
    assert {f.allergen for f in findings} == {"花生"}
