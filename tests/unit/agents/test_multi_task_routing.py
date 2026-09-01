"""
测试目标：一句话包含多个意图时的确定性切分与激活条件(D32,ARCHITECTURE §5.1.1)
——`segment_intents` 的切分规则本身、`classify_multi_task` 的保守激活条件
(全部片段必须 rule_matched 且分支互不相同才算真的多任务)。
对应实现：backend/agents/routing.py
"""
from __future__ import annotations

from backend.agents.routing import RouteBranch, classify_multi_task, segment_intents


def test_single_intent_message_returns_one_segment():
    assert segment_intents("今天该吃什么") == ["今天该吃什么"]


def test_natural_use_of_connector_word_without_real_split_stays_one_segment():
    """"还有什么其他推荐吗"里"还有"前面没有实质内容，不该切出一个空片段。"""
    assert segment_intents("还有什么其他推荐吗") == ["还有什么其他推荐吗"]


def test_splits_on_explicit_connector():
    segments = segment_intents("帮我记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么")
    assert segments == ["帮我记录一下中午吃了麻婆豆腐", "另外阳虚质应该吃什么"]


def test_adjacent_connectors_merge_into_one_split_point():
    """"顺便再问一下"是"顺便"+"再问一下"相邻——不该被拆成一个空洞的"顺便"片段
    加上"再问一下..."两段，见 D32 关于连接词按长度优先+连续匹配的说明。"""
    segments = segment_intents("记录一下吃了番茄炒蛋，顺便再问一下减肥期间能不能吃甜食")
    assert segments == ["记录一下吃了番茄炒蛋", "顺便再问一下减肥期间能不能吃甜食"]


def test_three_way_split():
    segments = segment_intents(
        "帮我记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么，还有我今天吃了什么"
    )
    assert segments == [
        "帮我记录一下中午吃了麻婆豆腐",
        "另外阳虚质应该吃什么",
        "还有我今天吃了什么",
    ]


def test_classify_multi_task_activates_for_distinct_rule_matched_branches():
    candidates = classify_multi_task("帮我记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么")
    assert candidates is not None
    assert len(candidates) == 2
    assert candidates[0].decision.branch is RouteBranch.LOG_WRITE
    assert candidates[1].decision.branch is RouteBranch.SINGLE_DOMAIN
    assert candidates[0].text == "帮我记录一下中午吃了麻婆豆腐"


def test_classify_multi_task_three_way():
    candidates = classify_multi_task(
        "帮我记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么，还有我今天吃了什么"
    )
    assert candidates is not None
    assert [c.decision.branch for c in candidates] == [
        RouteBranch.LOG_WRITE,
        RouteBranch.SINGLE_DOMAIN,
        RouteBranch.LOG_REVIEW,
    ]


def test_classify_multi_task_returns_none_for_single_intent():
    assert classify_multi_task("今天该吃什么") is None


def test_classify_multi_task_returns_none_when_segments_share_same_branch():
    """两句话都是"记录"——"记录"分支自己的三级查找本来就能处理一次输入里的
    多个菜品，不需要多任务路径重复处理。"""
    candidates = classify_multi_task("帮我记录一下昨天吃了红烧肉，还有帮我记录一下今天吃了宫保鸡丁")
    assert candidates is None


def test_splits_on_english_connector():
    segments = segment_intents(
        "Please log that I ate a pizza for lunch, by the way what should a yang deficiency constitution eat"
    )
    assert segments == [
        "Please log that I ate a pizza for lunch",
        "by the way what should a yang deficiency constitution eat",
    ]


def test_classify_multi_task_activates_for_english_message():
    candidates = classify_multi_task(
        "Please log that I ate a pizza for lunch, by the way what should a yang deficiency constitution eat"
    )
    assert candidates is not None
    assert candidates[0].decision.branch is RouteBranch.LOG_WRITE
    assert candidates[1].decision.branch is RouteBranch.SINGLE_DOMAIN


def test_english_connector_word_is_case_insensitive():
    segments = segment_intents("What did I eat today, BY THE WAY what should I eat tonight")
    assert len(segments) == 2


def test_natural_use_of_english_connector_word_without_real_split_falls_back():
    """"also"是常见英文词，单意图消息里自然出现也不该被误判成多任务——
    即便切出了两段，第一段("I")规则匹配不到任何分支，激活条件会挡住它。"""
    assert classify_multi_task("I also want to know what to eat") is None


def test_connector_word_does_not_match_inside_another_word():
    """"salsa"不该因为含有"als"这类子串被误判成命中了"also"——`\\b` 词边界。"""
    segments = segment_intents("log that I ate chips and salsa for lunch")
    assert segments == ["log that I ate chips and salsa for lunch"]


def test_classify_multi_task_returns_none_when_any_segment_is_unmatched():
    """第二段规则匹配不到任何分支(退化成"规则没命中"的模糊状态)——宁可整条
    消息退回单分支路径，也不基于一个不确定的切分往下走。"""
    candidates = classify_multi_task("帮我记录一下中午吃了麻婆豆腐，另外我今天心情不好")
    assert candidates is None
