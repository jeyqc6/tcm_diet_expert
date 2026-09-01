"""
测试目标：classify_turn() 的 LLM 兜底(D32 补充,2026-08-27)——纯规则切分覆盖
不到的两类情况：①连接词存在但某个片段规则没命中；②压根没有连接词的隐式
多意图。同时验证"确定性能处理的情况不打 LLM"这条零回归约束没被破坏。
对应实现：backend/agents/routing.py(classify_turn/_classify_turn_inner/_llm_classify_turn)
覆盖要求：mock complete()，不打真实网络。
"""
from __future__ import annotations

import asyncio

from backend.agents.routing import RouteBranch, classify_turn
from backend.llm.adapter import LLMResult, ModelTier


def _run(coro):
    return asyncio.run(coro)


def _llm(text: str) -> LLMResult:
    return LLMResult(text=text, model="m", tier=ModelTier.DEV, provider="fake")


class _ScriptedComplete:
    def __init__(self, script: list[LLMResult]):
        self._script = list(script)
        self.call_count = 0

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.call_count += 1
        if not self._script:
            raise AssertionError(f"complete() 被调用第 {self.call_count} 次，但脚本已经用完")
        return self._script.pop(0)


# --- 零回归：确定性能处理的情况完全不打 LLM ---------------------------------


def test_single_intent_rule_matched_never_calls_llm():
    complete = _ScriptedComplete([])
    tasks = _run(classify_turn("今天该吃什么", complete=complete))
    assert len(tasks) == 1
    assert tasks[0].decision.branch is RouteBranch.FULL_RECOMMEND
    assert complete.call_count == 0


def test_weather_constrained_eating_is_rule_matched_full_recommend():
    complete = _ScriptedComplete([])
    tasks = _run(classify_turn("今天的天气适合吃什么", complete=complete))
    assert len(tasks) == 1
    assert tasks[0].decision.branch is RouteBranch.FULL_RECOMMEND
    assert tasks[0].decision.rule_matched is True
    assert complete.call_count == 0


def test_deterministic_multi_task_never_calls_llm():
    complete = _ScriptedComplete([])
    tasks = _run(
        classify_turn("帮我记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么", complete=complete)
    )
    assert len(tasks) == 2
    assert complete.call_count == 0


def test_same_branch_segments_use_whole_message_without_llm():
    """两句话都是"记录"——不算"规则拿不准"，不该多打一次 LLM。"""
    complete = _ScriptedComplete([])
    tasks = _run(
        classify_turn(
            "帮我记录一下昨天吃了红烧肉，还有帮我记录一下今天吃了宫保鸡丁", complete=complete
        )
    )
    assert len(tasks) == 1
    assert tasks[0].decision.branch is RouteBranch.LOG_WRITE
    assert complete.call_count == 0


# --- LLM 兜底覆盖的两类新情况 -------------------------------------------------


def test_llm_fallback_for_fully_unmatched_message_can_return_single_task():
    complete = _ScriptedComplete(
        [_llm('{"tasks":[{"text":"totally unmatched utterance xyz","branch":"log_write","domain_hint":null}]}')]
    )
    tasks = _run(classify_turn("totally unmatched utterance xyz", complete=complete))
    assert len(tasks) == 1
    assert tasks[0].decision.branch is RouteBranch.LOG_WRITE
    assert tasks[0].decision.rule_matched is False
    assert complete.call_count == 1


def test_llm_fallback_for_fully_unmatched_message_can_return_multiple_tasks():
    """隐式多意图(没有连接词)——纯规则切分完全看不出这里有两个请求。"""
    complete = _ScriptedComplete(
        [
            _llm(
                '{"tasks":['
                '{"text":"麻婆豆腐好吃吗","branch":"fact_query","domain_hint":"tcm"},'
                '{"text":"我中午吃了","branch":"log_write","domain_hint":null}'
                "]}"
            )
        ]
    )
    tasks = _run(classify_turn("麻婆豆腐好吃吗我中午吃了", complete=complete))
    assert len(tasks) == 2
    assert tasks[0].decision.branch is RouteBranch.FACT_QUERY
    assert tasks[0].decision.domain_hint == "tcm"
    assert tasks[1].decision.branch is RouteBranch.LOG_WRITE


def test_llm_fallback_triggered_when_connector_present_but_one_segment_unmatched():
    """连接词存在(切出2段)，但第二段规则匹配不到任何分支——这正是纯规则版本
    会漏掉的情况，现在应该交给 LLM 判断，而不是直接吞掉第二个意图。"""
    complete = _ScriptedComplete(
        [
            _llm(
                '{"tasks":['
                '{"text":"帮我记录一下中午吃了麻婆豆腐","branch":"log_write","domain_hint":null},'
                '{"text":"我今天心情不好","branch":"full_recommend","domain_hint":null}'
                "]}"
            )
        ]
    )
    tasks = _run(
        classify_turn("帮我记录一下中午吃了麻婆豆腐，另外我今天心情不好", complete=complete)
    )
    assert len(tasks) == 2
    assert tasks[0].decision.branch is RouteBranch.LOG_WRITE
    assert tasks[1].decision.branch is RouteBranch.FULL_RECOMMEND
    assert complete.call_count == 1


def test_llm_failure_falls_back_to_rule_matched_single_result():
    """连接词存在、第二段没命中，LLM 调用本身失败——退回"整句当单任务"，
    不是让请求直接崩掉。"""
    async def failing_complete(messages, *, tools=None, **kwargs):
        raise RuntimeError("network error")

    tasks = _run(
        classify_turn(
            "帮我记录一下中午吃了麻婆豆腐，另外我今天心情不好", complete=failing_complete
        )
    )
    assert len(tasks) == 1
    assert tasks[0].decision.branch is RouteBranch.LOG_WRITE


def test_llm_unparseable_output_falls_back_to_full_recommend_when_nothing_rule_matched():
    complete = _ScriptedComplete([_llm("not json at all")])
    tasks = _run(classify_turn("totally unmatched utterance xyz", complete=complete))
    assert len(tasks) == 1
    assert tasks[0].decision.branch is RouteBranch.FULL_RECOMMEND
    assert tasks[0].decision.rule_matched is False


def test_llm_task_missing_text_falls_back_to_original_query():
    complete = _ScriptedComplete(
        [_llm('{"tasks":[{"branch":"full_recommend","domain_hint":null}]}')]
    )
    tasks = _run(classify_turn("totally unmatched utterance xyz", complete=complete))
    assert len(tasks) == 1
    assert tasks[0].text == "totally unmatched utterance xyz"
