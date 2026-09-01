"""
测试目标：backend/agents/clarification.py 的存取/清除逻辑（仿 onboarding
session_store 的既有测试风格），以及 citation.py 里追问标记的解析。
对应实现：backend/agents/clarification.py、backend/agents/citation.py
"""
from __future__ import annotations

from backend.agents.citation import CLARIFICATION_MARKER, extract_clarification_question
from backend.agents.clarification import InMemoryClarificationStore, PendingClarification
from backend.agents.routing import RouteBranch


def test_get_returns_none_when_nothing_pending():
    store = InMemoryClarificationStore()
    assert store.get("s1") is None


def test_put_then_get_round_trips():
    store = InMemoryClarificationStore()
    pending = PendingClarification(original_text="这个能不能吃", branch=RouteBranch.CANDIDATE_EVAL)
    store.put("s1", pending)
    assert store.get("s1") == pending


def test_clear_removes_pending():
    store = InMemoryClarificationStore()
    store.put("s1", PendingClarification(original_text="x", branch=RouteBranch.LOG_WRITE))
    store.clear("s1")
    assert store.get("s1") is None


def test_sessions_are_independent():
    store = InMemoryClarificationStore()
    store.put("s1", PendingClarification(original_text="x", branch=RouteBranch.LOG_WRITE))
    store.put("s2", PendingClarification(original_text="y", branch=RouteBranch.CANDIDATE_EVAL))
    assert store.get("s1").original_text == "x"
    assert store.get("s2").original_text == "y"
    store.clear("s1")
    assert store.get("s1") is None
    assert store.get("s2") is not None


def test_domain_hint_round_trips():
    store = InMemoryClarificationStore()
    pending = PendingClarification(
        original_text="阳虚质该吃什么", branch=RouteBranch.SINGLE_DOMAIN, domain_hint="tcm"
    )
    store.put("s1", pending)
    assert store.get("s1").domain_hint == "tcm"


def test_extract_clarification_question_matches_marker():
    text = f"{CLARIFICATION_MARKER} 你说的是哪一道菜？"
    assert extract_clarification_question(text) == "你说的是哪一道菜？"


def test_extract_clarification_question_none_for_normal_answer():
    assert extract_clarification_question("阳虚质忌生冷 [source: t1]") is None


def test_extract_clarification_question_requires_prefix_not_substring():
    """标记必须在开头，不是随便出现在文本某处——否则一段正常回答里恰好提到
    这几个字也会被误判成追问。"""
    text = f"我们不会输出 {CLARIFICATION_MARKER} 这种东西 [source: t1]"
    assert extract_clarification_question(text) is None


def test_extract_clarification_question_strips_whitespace():
    text = f"  {CLARIFICATION_MARKER}   你指的是哪个？  "
    assert extract_clarification_question(text) == "你指的是哪个？"
