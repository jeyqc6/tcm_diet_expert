"""
测试目标：诊断性表述 / "多多益善"式表述检测（过敏原部分见 test_allergen_block.py）
对应实现：backend/guardrails/output_filters.py
覆盖要求：常规
"""
from __future__ import annotations

from backend.guardrails.output_filters import (
    HIDDEN_ALLERGEN_SOURCES,
    check_diagnostic_statement,
    check_unlimited_good_statement,
    filter_output,
    hidden_sources_for_allergens,
)


def test_diagnostic_statement_is_detected() -> None:
    finding = check_diagnostic_statement("根据你的描述，你患有慢性胃炎。")
    assert finding is not None
    assert "患有" in finding.matched_text


def test_diagnosed_as_phrasing_is_detected() -> None:
    finding = check_diagnostic_statement("你可能确诊为糖尿病。")
    assert finding is not None


def test_normal_advice_is_not_flagged_as_diagnostic() -> None:
    assert check_diagnostic_statement("阳虚质建议少食生冷。") is None


def test_english_diagnostic_statement_is_detected() -> None:
    finding = check_diagnostic_statement("Based on your symptoms, you have chronic gastritis.")
    assert finding is not None


def test_english_diagnosed_with_phrasing_is_detected() -> None:
    assert check_diagnostic_statement("You might have diabetes.") is not None
    assert check_diagnostic_statement("You are diagnosed with hypertension.") is not None
    assert check_diagnostic_statement("Your diagnosis is IBS.") is not None


def test_english_diagnostic_pattern_is_case_insensitive() -> None:
    assert check_diagnostic_statement("YOU HAVE a chronic condition.") is not None


def test_english_normal_advice_is_not_flagged_as_diagnostic() -> None:
    assert check_diagnostic_statement("For a yang-deficient constitution, avoid cold and raw food.") is None


def test_unlimited_good_statement_is_detected() -> None:
    finding = check_unlimited_good_statement("红枣多吃有益，多多益善。")
    assert finding is not None


def test_bare_unlimited_good_phrase_is_detected() -> None:
    assert check_unlimited_good_statement("这类食物越多越好。") is not None


def test_normal_moderate_advice_is_not_flagged() -> None:
    assert check_unlimited_good_statement("建议适量摄入优质蛋白。") is None


def test_english_unlimited_good_statement_is_detected() -> None:
    assert check_unlimited_good_statement("Red dates are great — the more the better.") is not None


def test_english_eat_as_much_as_you_want_is_detected() -> None:
    assert check_unlimited_good_statement("Feel free to eat as much as you want.") is not None
    assert check_unlimited_good_statement("You can eat as much as you like.") is not None


def test_english_unlimited_amounts_is_detected() -> None:
    assert check_unlimited_good_statement("There's no limit on how much you can have.") is not None


def test_english_pattern_is_case_insensitive() -> None:
    assert check_unlimited_good_statement("The More The Better.") is not None


def test_english_normal_moderate_advice_is_not_flagged() -> None:
    assert check_unlimited_good_statement("Eat lean protein in moderation.") is None


def test_filter_output_aggregates_all_three_checks() -> None:
    result = filter_output(
        "你患有胃炎，建议多吃山药，多多益善，且含花生。", user_allergens=["花生"]
    )
    assert result.diagnostic is not None
    assert result.unlimited_good is not None
    assert len(result.allergens) == 1
    assert result.blocked is True  # 过敏原命中即 blocked


# ---------------------------------------------------------------------------
# hidden_sources_for_allergens()：反查隐藏来源，供生成阶段提示词使用
# ---------------------------------------------------------------------------


def test_hidden_sources_for_allergens_returns_matching_category_only() -> None:
    result = hidden_sources_for_allergens(["甲壳类"])
    assert set(result.keys()) == {"甲壳类"}
    assert "蚝油" in result["甲壳类"]


def test_hidden_sources_for_allergens_covers_multiple_categories() -> None:
    result = hidden_sources_for_allergens(["甲壳类", "芝麻"])
    assert set(result.keys()) == {"甲壳类", "芝麻"}
    assert "麻酱" in result["芝麻"]


def test_hidden_sources_for_allergens_empty_when_no_match() -> None:
    # 2026-09-01 之前这里用的是"坚果"——当时 HIDDEN_ALLERGEN_SOURCES 还没有
    # 这个类别的任何条目(中文隐藏来源表的已知缺口，见 critical_fact_scanner.py
    # 模块文档)。补了英文条目后"坚果"已经有真实映射了，这里换成一个确定不
    # 存在的类别名，测的是"查无此类别"这件事本身，不依赖某个类别当下是否
    # 恰好还没被覆盖。
    assert hidden_sources_for_allergens(["不存在的类别"]) == {}


def test_hidden_sources_for_allergens_empty_input() -> None:
    assert hidden_sources_for_allergens([]) == {}
    assert hidden_sources_for_allergens(None) == {}


def test_hidden_sources_reverse_index_matches_forward_table() -> None:
    """反向索引和正向表(HIDDEN_ALLERGEN_SOURCES)必须是同一份数据的两个视角，
    不能各自维护、慢慢漂移——这里断言两者字段数一致。"""
    all_categories = set(HIDDEN_ALLERGEN_SOURCES.values())
    result = hidden_sources_for_allergens(all_categories)
    total_terms = sum(len(v) for v in result.values())
    assert total_terms == len(HIDDEN_ALLERGEN_SOURCES)
