"""
测试目标：backend/agents/conflict_rules_lookup.py
——`select_matched_rules` 是纯函数(不连数据库,构造好的规则行来测)；
`fetch_matched_conflict_rules` 只测失败即静默降级。真实数据库端到端验证
（对着真实灌了40条 conflict_rules 的 Postgres 跑）见对话记录。
对应实现：backend/agents/conflict_rules_lookup.py
"""
from __future__ import annotations

from backend.agents.conflict_rules_lookup import (
    fetch_matched_conflict_rules,
    select_matched_rules,
)

_W01 = {
    "rule_id": "W01",
    "topic": "生菜沙拉",
    "relation": "conflict",
    "resolution": "改变烹饪方式",
    "confidence": "high",
    "applicable_constitutions": ["yang_xu", "qi_xu", "tan_shi"],
    "applicable_goals": ["weight_management"],
}
_W02 = {
    "rule_id": "W02",
    "topic": "低碳水",
    "relation": "conflict",
    "resolution": "不断主食",
    "confidence": "medium",
    "applicable_constitutions": ["qi_xu", "yang_xu", "ping_he"],
    "applicable_goals": ["weight_management", "improve_energy"],
}
_W03 = {
    "rule_id": "W03",
    "topic": "高蛋白",
    "relation": "partial_conflict",
    "resolution": "换蛋白来源",
    "confidence": "low",
    "applicable_constitutions": ["tan_shi", "shi_re"],
    "applicable_goals": ["weight_management"],
}
_UNRELATED = {
    "rule_id": "X99",
    "topic": "不相关",
    "relation": "aligned",
    "resolution": "无",
    "confidence": "high",
    "applicable_constitutions": ["yin_xu"],
    "applicable_goals": ["improve_sleep"],
}

_ALL_RULES = [_W01, _W02, _W03, _UNRELATED]


def test_matches_by_constitution_overlap():
    result = select_matched_rules(_ALL_RULES, constitutions=["qi_xu"], goal_tags=[])
    ids = {r["rule_id"] for r in result}
    assert ids == {"W01", "W02"}


def test_matches_by_goal_overlap():
    result = select_matched_rules(_ALL_RULES, constitutions=[], goal_tags=["improve_energy"])
    ids = {r["rule_id"] for r in result}
    assert ids == {"W02"}


def test_no_match_returns_empty():
    result = select_matched_rules(_ALL_RULES, constitutions=["te_bing"], goal_tags=["improve_sleep_quality"])
    assert result == []


def test_empty_constitutions_and_goals_returns_empty_not_everything():
    """既没有体质也没有目标(全新用户,画像完全空):不能把 40 条规则全塞进去。"""
    result = select_matched_rules(_ALL_RULES, constitutions=[], goal_tags=[])
    assert result == []


def test_sorted_by_confidence_high_first():
    result = select_matched_rules(_ALL_RULES, constitutions=["tan_shi"], goal_tags=[])
    ids = [r["rule_id"] for r in result]
    # W01(high) matches via tan_shi, W03(low) also matches via tan_shi.
    assert ids == ["W01", "W03"]


def test_limit_truncates():
    result = select_matched_rules(_ALL_RULES, constitutions=["qi_xu", "tan_shi", "shi_re"], goal_tags=[], limit=1)
    assert len(result) == 1
    assert result[0]["rule_id"] == "W01"  # highest confidence among matches


def test_matches_via_secondary_constitution_too():
    """体质夹杂(D28):次要体质命中的规则也该出现,不只按主体质过滤。"""
    result = select_matched_rules(_ALL_RULES, constitutions=["ping_he", "shi_re"], goal_tags=[])
    ids = {r["rule_id"] for r in result}
    assert ids == {"W02", "W03"}


def test_fetch_matched_conflict_rules_returns_empty_without_profile():
    assert fetch_matched_conflict_rules(constitutions=None, goal_tags=None) == []


def test_fetch_matched_conflict_rules_returns_empty_on_connection_failure():
    result = fetch_matched_conflict_rules(
        constitutions=["qi_xu"],
        goal_tags=None,
        dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist",
    )
    assert result == []
