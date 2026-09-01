"""
测试目标：三级查找优先级顺序（全局表>已晋升个人别名>LLM兜底）
对应实现：backend/memory/dish_decomposition.py
覆盖要求：常规

数据库相关的第二级(user_dish_aliases)查询失败即静默降级到第三级，同
backend/agents/user_context.py 的既有原则测(不连真实数据库)；真实数据库端到
端验证见对话记录。
"""
from __future__ import annotations

import asyncio

from backend.llm.adapter import LLMResult, ModelTier
from backend.memory.dish_decomposition import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    SOURCE_GLOBAL_TABLE,
    SOURCE_LLM_FALLBACK,
    DishMatch,
    MealDecomposition,
    decompose_meal,
    llm_fallback_decompose,
    match_global_table,
    normalize_phrase,
)


def _run(coro):
    return asyncio.run(coro)


def _result(text: str) -> LLMResult:
    return LLMResult(text=text, model="m", tier=ModelTier.DEV, provider="fake", tool_calls=None)


class _ScriptedComplete:
    def __init__(self, script):
        self._script = list(script)
        self.call_count = 0
        self.last_messages = None

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.call_count += 1
        self.last_messages = messages
        return self._script.pop(0)


def test_normalize_phrase_strips_whitespace_and_punctuation():
    assert normalize_phrase("西红柿炒蛋, 加饭！") == "西红柿炒蛋加饭"
    assert normalize_phrase("  番茄炒蛋  ") == "番茄炒蛋"


def test_match_global_table_finds_known_dish_as_substring():
    matches, remaining = match_global_table("晚上吃了番茄炒蛋加一小碗白米饭")
    dishes = {m.dish for m in matches}
    assert dishes == {"番茄炒蛋", "白米饭"}
    assert all(m.confidence == CONFIDENCE_HIGH for m in matches)
    assert all(m.source_tier == SOURCE_GLOBAL_TABLE for m in matches)
    assert "番茄炒蛋" not in remaining
    assert "白米饭" not in remaining


def test_standalone_staples_match_without_llm():
    """米饭/粗粮饭/南瓜这类能单独吃的食材应直接命中全局表。"""
    for name in ("白米饭", "米饭", "粗粮饭", "南瓜", "红薯", "玉米"):
        matches, remaining = match_global_table(f"中午吃了{name}")
        assert name in {m.dish for m in matches}, name
        assert name not in remaining


def test_bare_fan_is_not_in_the_table():
    """单字「饭」不能入表：晚饭/午饭/吃饭都会被当成一碗米饭。"""
    matches, remaining = match_global_table("晚饭吃了番茄炒蛋")
    assert {m.dish for m in matches} == {"番茄炒蛋"}
    assert "饭" not in {m.dish for m in matches}
    assert "晚" in remaining


def test_longer_rice_dish_is_not_split_into_plain_rice():
    matches, remaining = match_global_table("中午吃了黄焖鸡米饭")
    assert {m.dish for m in matches} == {"黄焖鸡米饭"}
    assert "米饭" not in remaining
    assert "白米饭" not in {m.dish for m in matches}


def test_match_global_table_no_match_returns_full_remaining():
    matches, remaining = match_global_table("吃了一些奇怪的东西")
    assert matches == ()
    assert remaining == "吃了一些奇怪的东西"


def test_match_global_table_prefers_longer_dish_name_first():
    """"蚝油炒芥蓝"和单独的"芥蓝"都在表里；确保长名字不会被短子串抢先破坏。"""
    matches, remaining = match_global_table("今天中午吃了蚝油炒芥蓝")
    dishes = {m.dish for m in matches}
    assert dishes == {"蚝油炒芥蓝"}
    assert "芥蓝" not in remaining  # consumed as part of the longer match


def test_match_global_table_multiple_dishes_in_one_sentence():
    matches, remaining = match_global_table("红烧肉和宫保鸡丁都点了")
    dishes = {m.dish for m in matches}
    assert dishes == {"红烧肉", "宫保鸡丁"}


def test_dish_match_carries_ingredients_and_tcm_nature():
    matches, _ = match_global_table("番茄炒蛋")
    m = matches[0]
    assert "鸡蛋" in m.ingredients
    assert m.tcm_nature == "平"


def test_dish_match_carries_known_allergens():
    matches, _ = match_global_table("宫保鸡丁")
    m = matches[0]
    assert "花生" in m.allergens


def test_meal_decomposition_dedupes_ingredients_across_matches():
    m1 = DishMatch(dish="A", ingredients=("鸡蛋", "盐"), tcm_nature="温", allergens=("蛋",), confidence=CONFIDENCE_HIGH, source_tier=SOURCE_GLOBAL_TABLE)
    m2 = DishMatch(dish="B", ingredients=("盐", "糖"), tcm_nature="温", allergens=(), confidence=CONFIDENCE_HIGH, source_tier=SOURCE_GLOBAL_TABLE)
    decomp = MealDecomposition(matches=(m1, m2))
    assert decomp.all_ingredients() == ("鸡蛋", "盐", "糖")
    assert decomp.all_food_properties() == ("温",)
    assert decomp.all_allergens() == ("蛋",)


def test_llm_fallback_decompose_parses_structured_json():
    complete = _ScriptedComplete(
        [_result('{"dishes":[{"dish":"白米饭","ingredients":["大米"],"tcm_nature":"平","allergens":[]}]}')]
    )
    matches = _run(llm_fallback_decompose("一小碗白米饭", complete=complete))
    assert len(matches) == 1
    assert matches[0].dish == "白米饭"
    assert matches[0].confidence == CONFIDENCE_LOW
    assert matches[0].source_tier == SOURCE_LLM_FALLBACK


def test_llm_fallback_decompose_handles_empty_dishes_list():
    complete = _ScriptedComplete([_result('{"dishes":[]}')])
    matches = _run(llm_fallback_decompose("嗯", complete=complete))
    assert matches == ()


def test_llm_fallback_decompose_strips_markdown_code_fence():
    """真实发现的问题：Haiku 即便被明确要求"只输出严格 JSON"，仍然会把 JSON
    包在 ```json ... ``` 代码块里，直接 json.loads 会报错(观察到的真实响应
    见对话记录)。"""
    complete = _ScriptedComplete(
        [_result('```json\n{"dishes":[{"dish":"燕麦牛奶","ingredients":["燕麦","牛奶"],"tcm_nature":"平","allergens":["牛奶"]}]}\n```')]
    )
    matches = _run(llm_fallback_decompose("一杯燕麦牛奶", complete=complete))
    assert len(matches) == 1
    assert matches[0].dish == "燕麦牛奶"


def test_llm_fallback_decompose_tolerates_non_json_response():
    complete = _ScriptedComplete([_result("not json at all")])
    matches = _run(llm_fallback_decompose("随便说点什么", complete=complete))
    assert matches == ()


def test_decompose_meal_pure_global_table_hit_never_calls_llm():
    complete = _ScriptedComplete([])  # 不应该被调用
    result = _run(decompose_meal("番茄炒蛋", complete=complete))
    assert len(result.matches) == 1
    assert result.matches[0].dish == "番茄炒蛋"
    assert complete.call_count == 0


def test_decompose_meal_falls_back_to_llm_for_unknown_dish():
    complete = _ScriptedComplete(
        [_result('{"dishes":[{"dish":"龙须酥","ingredients":["面粉","糖"],"tcm_nature":"平","allergens":["含麸质谷物"]}]}')]
    )
    # dsn 指向一个连不上的地址 -> 第二级(user_dish_aliases)静默降级，落到 LLM。
    result = _run(
        decompose_meal(
            "龙须酥", dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist", complete=complete
        )
    )
    assert len(result.matches) == 1
    assert result.matches[0].source_tier == SOURCE_LLM_FALLBACK
    assert complete.call_count == 1


def test_decompose_meal_calls_llm_even_for_narrative_filler_but_result_stays_correct():
    """"帮我记录一下，中午吃了麻婆豆腐"里，去掉"麻婆豆腐"之后剩下的是纯叙述性
    填充词——2026-08-31 之前这里靠一份连接词关键词表("加"/"和"/"还有"...)
    短路掉这次 LLM 调用；那份关键词表在真实数据里发现覆盖不全(用户用"+"分隔
    多个菜品时会漏记，见 dish_decomposition.py 模块文档)，改成"只要有残留就
    查"，主动接受"纯填充词也会多打一次 LLM"这个代价。这次 LLM 调用应该按
    `_LLM_SYSTEM_PROMPT` 的约定正确返回空 dishes，最终结果依然只有全局表
    命中的那一个菜，不会凭空多出东西。"""
    complete = _ScriptedComplete([_result('{"dishes":[]}')])
    result = _run(decompose_meal("帮我记录一下，中午吃了麻婆豆腐", complete=complete))
    assert len(result.matches) == 1
    assert result.matches[0].dish == "麻婆豆腐"
    assert complete.call_count == 1


def test_decompose_meal_calls_llm_for_plus_separated_dishes_not_in_continuation_keywords():
    """真实发现的问题("鸡爪+牛肉南瓜炖年糕+半个小西瓜"，见 2026-08-30 用户
    真实数据)：用户用"+"分隔多个菜品，"+"不在旧的连接词关键词表里，命中
    全局表的"西瓜"之后就直接短路，"鸡爪"整个丢了。这条测试锁定修复后的
    行为——用全局表里没有的「龙须酥」代表那个被漏掉的菜。"""
    complete = _ScriptedComplete(
        [_result('{"dishes":[{"dish":"龙须酥","ingredients":["面粉","糖"],"tcm_nature":"平","allergens":["含麸质谷物"]}]}')]
    )
    result = _run(decompose_meal("番茄炒蛋+龙须酥", complete=complete))
    dishes = {m.dish for m in result.matches}
    assert dishes == {"番茄炒蛋", "龙须酥"}
    assert complete.call_count == 1


def test_decompose_meal_still_calls_llm_when_continuation_marker_present():
    """旧连接词表里的"加"这类词依然应该继续走 LLM 兜底(现在是"任何非空残留
    都查"的一个特例，不再是靠关键词表才触发)。用全局表里没有的「龙须酥」，
    避免白米饭入表后这条变成纯查表命中。"""
    complete = _ScriptedComplete(
        [_result('{"dishes":[{"dish":"龙须酥","ingredients":["面粉","糖"],"tcm_nature":"平","allergens":["含麸质谷物"]}]}')]
    )
    result = _run(decompose_meal("番茄炒蛋加一块龙须酥", complete=complete))
    dishes = {m.dish for m in result.matches}
    assert dishes == {"番茄炒蛋", "龙须酥"}
    assert complete.call_count == 1


def test_decompose_meal_combines_global_table_and_llm_fallback():
    complete = _ScriptedComplete(
        [_result('{"dishes":[{"dish":"龙须酥","ingredients":["面粉","糖"],"tcm_nature":"平","allergens":["含麸质谷物"]}]}')]
    )
    result = _run(
        decompose_meal(
            "番茄炒蛋加一块龙须酥",
            dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist",
            complete=complete,
        )
    )
    dishes = {m.dish for m in result.matches}
    assert dishes == {"番茄炒蛋", "龙须酥"}
