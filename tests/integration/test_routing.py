"""
测试目标：六条分支各自命中对应示例问法；易混淆分支边界用例（记录回顾vs事实查询、候选评估vs完整推荐）
对应实现：backend/agents/routing.py
覆盖要求：规则路由不依赖 LLM；LLM 兜底用 mock complete，不打真实网络
"""
from __future__ import annotations

import asyncio

import pytest

from backend.agents.routing import (
    RouteBranch,
    classify_route,
    classify_route_async,
    _parse_route_llm_json,
)
from backend.llm.adapter import LLMResult, ModelTier


def _run(coro):
    return asyncio.run(coro)


def _llm(text: str) -> LLMResult:
    return LLMResult(text=text, model="m", tier=ModelTier.DEV, provider="fake")


# 六条分支各 5 条中文（BUILD_PLAN 完成判据）
CASES: dict[RouteBranch, list[str]] = {
    RouteBranch.LOG_WRITE: [
        "帮我记录一下，中午吃了麻婆豆腐和米饭",
        "刚才吃了蚝油炒芥蓝，记一下",
        "今天午餐吃了黄焖鸡，请帮我记下",
        "写入饮食记录：晚餐小米粥",
        "刚刚吃了红烧肉，保存到记录里",
    ],
    RouteBranch.LOG_REVIEW: [
        "我昨天晚上吃了什么，我都忘了",
        "上周我吃了什么？",
        "回顾一下我这几天的饮食",
        "查一下我的饮食记录",
        "前天中午我吃了什么",
    ],
    RouteBranch.FACT_QUERY: [
        "红枣是什么性味？",
        "生姜归经是什么？",
        "蚝油里面有什么过敏原？",
        "什么是药食同源？",
        "牛奶含不含乳糖？",
    ],
    RouteBranch.CANDIDATE_EVAL: [
        "楼下有黄焖鸡、米线，我选哪一个吃",
        "这个我现在能不能吃？",
        "今天晚上这些可以吃吗？",
        "我今天已经吃了某个大补的东西，还能吃什么、不能吃什么？",
        "黄焖鸡和米线选哪个更好",
    ],
    RouteBranch.SINGLE_DOMAIN: [
        "气虚质春季该吃什么？",
        "从中医角度，阳虚质忌什么？",
        "缺铁性贫血怎么补？",
        "红枣能不能纠正缺铁性贫血？",
        "我在吃华法林，菠菜要注意什么？",
    ],
    RouteBranch.FULL_RECOMMEND: [
        "今天该吃什么？",
        "加班到很晚了，晚饭吃点什么比较好",
        "帮我推荐一天的饮食安排",
        "今晚想吃点什么合适",
        "给我安排三餐吃什么",
    ],
}

ENGLISH_CASES: dict[RouteBranch, list[str]] = {
    RouteBranch.LOG_WRITE: [
        "Please log that I ate mapo tofu for lunch",
        "I just ate oyster-sauce gai lan, save it to my diet log",
    ],
    RouteBranch.LOG_REVIEW: [
        "What did I eat yesterday?",
        "Review my diet log from last week",
    ],
    RouteBranch.FACT_QUERY: [
        "What is the nature of jujube?",
        "Does oyster sauce contain any allergens?",
    ],
    RouteBranch.CANDIDATE_EVAL: [
        "Can I eat this right now?",
        "Braised chicken or rice noodles, which should I pick?",
    ],
    RouteBranch.SINGLE_DOMAIN: [
        "From a TCM perspective, what should yang-deficiency constitution avoid?",
        "Iron deficiency anemia, what to eat?",
    ],
    RouteBranch.FULL_RECOMMEND: [
        "What should I eat today?",
        "Working late, what should I eat for dinner?",
    ],
}


@pytest.mark.parametrize(
    "branch,query",
    [(b, q) for b, qs in CASES.items() for q in qs],
    ids=lambda x: x if isinstance(x, str) else x.value,
)
def test_six_branches_five_each(branch: RouteBranch, query: str) -> None:
    decision = classify_route(query)
    assert decision.branch is branch, (
        f"query={query!r} expected {branch.value}, got {decision.branch.value} ({decision.reason})"
    )


class TestBoundaryLogReviewVsFactQuery:
    """D25：记录回顾查 diet_log；事实查询查 knowledge_chunks——不要合并。"""

    def test_own_history_is_log_review_not_fact(self) -> None:
        d = classify_route("我昨天晚上吃了什么，我都忘了")
        assert d.branch is RouteBranch.LOG_REVIEW

    def test_knowledge_property_is_fact_not_log_review(self) -> None:
        d = classify_route("红枣是什么性味？")
        assert d.branch is RouteBranch.FACT_QUERY

    def test_allergen_in_condiment_is_fact_not_log_review(self) -> None:
        d = classify_route("蚝油里面有什么过敏原？")
        assert d.branch is RouteBranch.FACT_QUERY


class TestBoundaryCandidateVsFullRecommend:
    """D25：候选评估评估给定选项；完整推荐从零生成——不要合并。"""

    def test_two_options_is_candidate_not_full(self) -> None:
        d = classify_route("楼下有黄焖鸡、米线，我选哪一个吃")
        assert d.branch is RouteBranch.CANDIDATE_EVAL

    def test_open_ended_dinner_is_full_not_candidate(self) -> None:
        d = classify_route("加班到很晚了，晚饭吃点什么比较好")
        assert d.branch is RouteBranch.FULL_RECOMMEND

    def test_weather_constrained_eating_is_full_recommend_not_unmatched(self) -> None:
        """'今天的天气适合吃什么' used to miss the explicit full_recommend regex
        and the turn LLM dumped it into other because other mentioned weather."""
        for query in (
            "今天的天气适合吃什么",
            "这种天气适合吃什么",
            "What should I eat in this weather?",
        ):
            d = classify_route(query)
            assert d.branch is RouteBranch.FULL_RECOMMEND, query
            assert d.rule_matched is True, query

    def test_pure_weather_question_is_not_a_rule_hit(self) -> None:
        """'今天天气怎么样' is D33's canonical other case — rules must not
        swallow it as full_recommend; LLM fallback still decides other."""
        d = classify_route("今天天气怎么样")
        assert d.rule_matched is False
        assert d.reason == "unmatched"

    def test_already_ate_tonifying_food_is_candidate(self) -> None:
        d = classify_route("我今天已经吃了某个大补的东西，还能吃什么、不能吃什么？")
        assert d.branch is RouteBranch.CANDIDATE_EVAL


def test_empty_defaults_to_full_recommend() -> None:
    assert classify_route("").branch is RouteBranch.FULL_RECOMMEND
    assert classify_route("   ").branch is RouteBranch.FULL_RECOMMEND


class TestDomainHint:
    """fact_query/single_domain 分支内部"该查哪一侧知识库"的提示——
    api/main.py 派发 retrieve_tcm 还是 retrieve_nutrition 要靠这个字段，
    不是分支本身就能回答的问题。"""

    @pytest.mark.parametrize(
        "query,expected_domain",
        [
            ("红枣是什么性味？", "tcm"),
            ("生姜归经是什么？", "tcm"),
            ("蚝油里面有什么过敏原？", "nutrition"),
            ("什么是药食同源？", "tcm"),
            ("牛奶含不含乳糖？", "nutrition"),
        ],
    )
    def test_fact_query_domain_hint(self, query: str, expected_domain: str) -> None:
        d = classify_route(query)
        assert d.branch is RouteBranch.FACT_QUERY
        assert d.domain_hint == expected_domain

    def test_single_domain_tcm_hint(self) -> None:
        d = classify_route("阳虚质该吃什么")
        assert d.branch is RouteBranch.SINGLE_DOMAIN
        assert d.domain_hint == "tcm"

    def test_single_domain_nutrition_hint(self) -> None:
        d = classify_route("缺铁怎么补")
        assert d.branch is RouteBranch.SINGLE_DOMAIN
        assert d.domain_hint == "nutrition"

    def test_other_branches_have_no_domain_hint(self) -> None:
        # 双派发/写入/回顾分支不需要这个字段——不是"忘了填"，是根本不适用
        assert classify_route("帮我记录一下，中午吃了麻婆豆腐").domain_hint is None
        assert classify_route("今天吃什么比较好").domain_hint is None

    def test_english_fact_query_domain_hint(self) -> None:
        d = classify_route("What is the nature of jujube?")
        assert d.branch is RouteBranch.FACT_QUERY
        assert d.domain_hint == "tcm"
        d = classify_route("Does oyster sauce contain any allergens?")
        assert d.branch is RouteBranch.FACT_QUERY
        assert d.domain_hint == "nutrition"


@pytest.mark.parametrize(
    "branch,query",
    [(b, q) for b, qs in ENGLISH_CASES.items() for q in qs],
    ids=lambda x: x if isinstance(x, str) else x.value,
)
def test_english_rule_patterns(branch: RouteBranch, query: str) -> None:
    decision = classify_route(query)
    assert decision.branch is branch, (
        f"query={query!r} expected {branch.value}, got {decision.branch.value} ({decision.reason})"
    )
    assert decision.rule_matched


def test_unmatched_is_not_a_real_full_recommend_hit() -> None:
    d = classify_route("I'm tired and a bit bloated after a long flight")
    assert d.branch is RouteBranch.FULL_RECOMMEND
    assert d.rule_matched is False
    assert d.reason == "unmatched"


class TestLlmFallback:
    """Rules miss → LLM classifies via English-key JSON; LLM fail → full_recommend."""

    def test_rule_hit_does_not_call_llm(self) -> None:
        called = {"n": 0}

        async def complete(messages, **kwargs):
            called["n"] += 1
            return _llm('{"branch":"log_write","domain_hint":null}')

        d = _run(classify_route_async("今天该吃什么？", complete=complete))
        assert d.branch is RouteBranch.FULL_RECOMMEND
        assert d.rule_matched is True
        assert called["n"] == 0

    def test_empty_skips_llm(self) -> None:
        async def complete(messages, **kwargs):
            raise AssertionError("empty query must not call the route LLM")

        d = _run(classify_route_async("   ", complete=complete))
        assert d.branch is RouteBranch.FULL_RECOMMEND
        assert d.reason == "empty_query_default"

    def test_unmatched_uses_english_json_branch(self) -> None:
        async def complete(messages, **kwargs):
            assert "英文键名" in messages[0]["content"] or "English keys" in messages[0]["content"]
            return _llm('{"branch":"fact_query","domain_hint":"tcm"}')

        d = _run(
            classify_route_async(
                "I'm tired and a bit bloated after a long flight", complete=complete
            )
        )
        assert d.branch is RouteBranch.FACT_QUERY
        assert d.domain_hint == "tcm"
        assert d.rule_matched is False
        assert d.reason.startswith("llm_fallback:")

    def test_markdown_fenced_json_is_parsed(self) -> None:
        async def complete(messages, **kwargs):
            return _llm('```json\n{"branch":"candidate_eval","domain_hint":null}\n```')

        d = _run(classify_route_async("hmm not sure about those two dishes", complete=complete))
        assert d.branch is RouteBranch.CANDIDATE_EVAL

    def test_chinese_keys_are_ignored_english_keys_required(self) -> None:
        assert _parse_route_llm_json('{"分支":"fact_query","domain_hint":"tcm"}') is None
        parsed = _parse_route_llm_json('{"branch":"single_domain","domain_hint":"nutrition"}')
        assert parsed == (RouteBranch.SINGLE_DOMAIN, "nutrition")

    def test_llm_failure_defaults_to_full_recommend(self) -> None:
        async def complete(messages, **kwargs):
            raise RuntimeError("provider down")

        d = _run(classify_route_async("totally unmatched utterance xyz", complete=complete))
        assert d.branch is RouteBranch.FULL_RECOMMEND
        assert d.reason == "llm_fallback_failed_default_full_recommend"

    def test_garbage_output_defaults_to_full_recommend(self) -> None:
        async def complete(messages, **kwargs):
            return _llm("sorry I cannot help with that")

        d = _run(classify_route_async("totally unmatched utterance xyz", complete=complete))
        assert d.branch is RouteBranch.FULL_RECOMMEND
        assert d.reason == "llm_fallback_failed_default_full_recommend"
