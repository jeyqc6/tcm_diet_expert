"""
测试目标：RouteBranch.OTHER 的确定性快速通道(D33/PRD §17)——整句锚定匹配，
不是子串搜索，混了真实请求的问候不该被误判成 other。
对应实现：backend/agents/routing.py(_OTHER_GREETINGS, classify_route)
"""
from __future__ import annotations

from backend.agents.routing import RouteBranch, classify_route


def test_pure_chinese_greetings_are_other():
    for text in ("你好", "您好！", "嗨", "哈喽"):
        assert classify_route(text).branch is RouteBranch.OTHER, text


def test_pure_chinese_thanks_and_farewell_are_other():
    for text in ("谢谢", "谢谢你", "多谢", "感谢", "拜拜", "再见"):
        assert classify_route(text).branch is RouteBranch.OTHER, text


def test_pure_english_greetings_are_other():
    for text in ("hi", "Hello!", "hey", "thanks", "thank you", "bye", "goodbye"):
        assert classify_route(text).branch is RouteBranch.OTHER, text


def test_greeting_combined_with_real_request_is_not_other():
    """真实请求即便夹了一句问候，不该被误判成纯寒暄——整句锚定匹配这时候
    根本不会命中，前面更具体的分支会先拦下真实请求。"""
    assert classify_route("谢谢你的建议，那我该吃什么").branch is not RouteBranch.OTHER
    assert classify_route("你好，今天该吃什么").branch is RouteBranch.FULL_RECOMMEND
    assert classify_route("hello, what should I eat today").branch is RouteBranch.FULL_RECOMMEND


def test_other_is_rule_matched_not_llm_fallback():
    """纯寒暄走确定性快速通道，不该被标记成"规则没命中"。"""
    result = classify_route("你好")
    assert result.rule_matched is True
