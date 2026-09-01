"""
测试目标：核查 pass 第 1/4/6/8 项硬安全检查只拒绝不改写（D15）；无 source_id
的建议被移除而非补全；第 2/3/5 项软判定允许 annotate 改写(2026-08-31)，
改写文本在初次硬检查通过后直接采用；
Skill 拼入核查调用而非中枢常驻 prompt；候选评估走 D25 变体规则。
对应实现：backend/agents/verification.py
覆盖要求：集成测试，mock LLM，不打真实网络
"""
from __future__ import annotations

import asyncio
import json

from backend.agents.verification import (
    SuggestionItem,
    apply_deterministic_source_check,
    build_verification_system_prompt,
    repair_insufficient_evidence,
    verify,
)


def _run(coro):
    return asyncio.run(coro)


# ----- 完成判据：无 source_id → 移除，不是补全 -----


def test_missing_source_id_is_removed_not_completed() -> None:
    items = [
        SuggestionItem(text="建议多吃红枣补血。", item_id="bad"),
        SuggestionItem(
            text="阳虚质忌生冷 [source: tcm_000123]。",
            item_id="good",
        ),
    ]
    accepted, rejected = apply_deterministic_source_check(
        items, available_source_ids=["tcm_000123", "tcm_000456"]
    )
    assert [a.item_id for a in accepted] == ["good"]
    assert len(rejected) == 1
    assert rejected[0].item.item_id == "bad"
    assert rejected[0].action == "remove"
    assert "不补全" in rejected[0].reason
    # 被拒条目不能被改写成带伪造 source_id
    assert "[source:" not in rejected[0].item.text
    assert rejected[0].item.text == "建议多吃红枣补血。"


def test_hallucinated_source_id_is_removed_not_rewritten() -> None:
    items = [
        SuggestionItem(
            text="随便写一句 [source: tcm_FAKE]。",
            item_id="hallucinated",
        )
    ]
    accepted, rejected = apply_deterministic_source_check(
        items, available_source_ids=["tcm_000123"]
    )
    assert accepted == []
    assert rejected[0].item.item_id == "hallucinated"
    assert "幻觉" in rejected[0].reason or "不改写" in rejected[0].reason
    assert rejected[0].item.text.endswith("[source: tcm_FAKE]。")


def test_verify_end_to_end_removes_uncited_and_keeps_cited() -> None:
    items = [
        SuggestionItem(text="无依据的推荐：每天吃冰淇淋。", item_id="u1"),
        SuggestionItem(
            text="气虚宜山药 [source: tcm_000001]。",
            item_id="u2",
        ),
    ]

    async def complete(messages, **kwargs):
        return type("R", (), {"text": '{"reject":[],"retry_reconciliation":false}'})()

    result = _run(
        verify(
            items,
            available_source_ids=["tcm_000001"],
            complete=complete,
        )
    )
    assert [a.item_id for a in result.accepted] == ["u2"]
    assert "u1" in {r.item.item_id for r in result.rejected}
    for a in result.accepted:
        assert a.item_id != "u1"


def test_llm_cannot_resurrect_removed_item_by_rewriting() -> None:
    """LLM 若返回「改写后带 source_id 的文本」，也不采纳——已拒条目不回炉。"""
    items = [SuggestionItem(text="请补全：多吃雪糕。", item_id="bare")]

    async def evil_complete(messages, **kwargs):
        return type(
            "R",
            (),
            {
                "text": json.dumps(
                    {
                        "reject": [],
                        "retry_reconciliation": False,
                        "rewritten": [
                            {
                                "item_id": "bare",
                                "text": "多吃雪糕 [source: tcm_000001]。",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            },
        )()

    result = _run(
        verify(
            items,
            available_source_ids=["tcm_000001"],
            complete=evil_complete,
        )
    )
    assert result.accepted == []
    assert any(r.item.item_id == "bare" for r in result.rejected)
    assert all("[source:" not in r.item.text for r in result.rejected)


# ----- 2026-08-31：软判定 annotate（改写后直接采用）-----


def test_llm_annotate_rewrites_text_directly_after_initial_checks() -> None:
    """An annotate response replaces the initially checked text directly."""
    items = [SuggestionItem(text="喝绿茶清热 [source: n1]。", item_id="x1")]

    async def complete(messages, **kwargs):
        return type(
            "R",
            (),
            {
                "text": json.dumps(
                    {
                        "items": [
                            {
                                "item_id": "x1",
                                "action": "annotate",
                                "text": "适量饮水有助于补充水分 [source: n1]（此处引用原本讨论的是饮水，非绿茶本身的功效）。",
                                "check_number": 2,
                                "reason": "引用支持的是饮水，不是绿茶清热",
                            }
                        ],
                        "retry_reconciliation": False,
                    },
                    ensure_ascii=False,
                )
            },
        )()

    result = _run(verify(items, available_source_ids=["n1"], complete=complete))
    assert [a.item_id for a in result.accepted] == ["x1"]
    assert result.accepted[0].text.startswith("适量饮水")
    assert result.rejected == []


def test_llm_annotate_text_is_sent_without_a_second_source_check() -> None:
    """The current policy sends annotate text without a second source check."""
    items = [SuggestionItem(text="喝绿茶清热 [source: n1]。", item_id="x1")]

    async def complete(messages, **kwargs):
        return type(
            "R",
            (),
            {
                "text": json.dumps(
                    {
                        "items": [
                            {
                                "item_id": "x1",
                                "action": "annotate",
                                "text": "喝绿茶清热 [source: n_fake]。",
                                "check_number": 2,
                                "reason": "改写但编了个新 id",
                            }
                        ],
                        "retry_reconciliation": False,
                    },
                    ensure_ascii=False,
                )
            },
        )()

    result = _run(verify(items, available_source_ids=["n1"], complete=complete))
    assert [a.item_id for a in result.accepted] == ["x1"]
    assert result.accepted[0].text == "喝绿茶清热 [source: n_fake]。"
    assert result.accepted[0].source_ids == []
    assert result.rejected == []


def test_llm_annotate_text_is_sent_without_a_second_hard_check() -> None:
    """The initial text passes; a hard-check hit added by annotate is sent directly."""
    items = [SuggestionItem(text="推荐清蒸鱼 [source: n1]。", item_id="x1")]

    async def complete(messages, **kwargs):
        return type(
            "R",
            (),
            {
                "text": json.dumps(
                    {
                        "items": [
                            {
                                "item_id": "x1",
                                "action": "annotate",
                                "text": "推荐加一勺蚝油提鲜，风味更佳 [source: n1]。",
                                "check_number": 2,
                                "reason": "润色措辞",
                            }
                        ],
                        "retry_reconciliation": False,
                    },
                    ensure_ascii=False,
                )
            },
        )()

    result = _run(
        verify(
            items, available_source_ids=["n1"], complete=complete, user_allergens=["甲壳类"]
        )
    )
    assert [a.item_id for a in result.accepted] == ["x1"]
    assert result.accepted[0].text.startswith("推荐加一勺蚝油")
    assert result.rejected == []


def test_llm_reject_action_removes_item_with_llm_reason() -> None:
    """action=reject（新 schema 下等价于旧版 reject）依然整条移除，理由/
    check_number 采用 LLM 给出的值。"""
    items = [SuggestionItem(text="没什么依据的常识性建议 [source: n1]。", item_id="x1")]

    async def complete(messages, **kwargs):
        return type(
            "R",
            (),
            {
                "text": json.dumps(
                    {
                        "items": [
                            {
                                "item_id": "x1",
                                "action": "reject",
                                "check_number": 2,
                                "reason": "核心结论本身缺乏依据支撑，无法通过改写挽救",
                            }
                        ],
                        "retry_reconciliation": False,
                    },
                    ensure_ascii=False,
                )
            },
        )()

    result = _run(verify(items, available_source_ids=["n1"], complete=complete))
    assert result.accepted == []
    assert result.rejected[0].item.item_id == "x1"
    assert result.rejected[0].check_number == 2
    assert "无法通过改写挽救" in result.rejected[0].reason


def test_llm_item_missing_from_verdicts_defaults_to_accept() -> None:
    """解析出的 verdicts 里没提到的 item_id 按 accept 处理（宽松容错）。"""
    items = [SuggestionItem(text="气虚宜山药 [source: n1]。", item_id="x1")]

    async def complete(messages, **kwargs):
        return type("R", (), {"text": '{"items":[],"retry_reconciliation":false}'})()

    result = _run(verify(items, available_source_ids=["n1"], complete=complete))
    assert [a.item_id for a in result.accepted] == ["x1"]


def test_evidence_repair_preserves_valid_citation_without_tools() -> None:
    calls = []

    async def complete(messages, **kwargs):
        calls.append((messages, kwargs))
        return type(
            "R",
            (),
            {"text": "保留有依据的建议 [source: n1]，删去无法支持的具体数字。"},
        )()

    repaired = _run(
        repair_insufficient_evidence(
            "原始建议：保留有依据的建议 [source: n1]，以及不确定数字 [source: fake].",
            ["引用只支持部分结论"],
            available_source_ids=["n1"],
            complete=complete,
        )
    )
    assert repaired is not None
    assert repaired.source_ids == ["n1"]
    assert "保留有依据的建议" in repaired.text
    assert "fake" not in repaired.text
    assert "tools" not in calls[0][1]
    assert "引用只支持部分结论" in calls[0][0][1]["content"]


def test_evidence_repair_labels_uncited_general_knowledge() -> None:
    async def complete(messages, **kwargs):
        return type("R", (), {"text": "鸡蛋黄的胆固醇含量需要以可靠资料为准。"})()

    repaired = _run(
        repair_insufficient_evidence(
            "鸡蛋黄含有胆固醇。",
            ["本地知识库没有直接依据"],
            available_source_ids=[],
            complete=complete,
        )
    )
    assert repaired is not None
    assert "模型通用知识" in repaired.text
    assert repaired.source_ids == []


def test_evidence_repair_fallback_retains_usable_text_and_strips_placeholder() -> None:
    async def complete(messages, **kwargs):
        raise RuntimeError("repair unavailable")

    repaired = _run(
        repair_insufficient_evidence(
            "可以适量食用 [source: chunk_id]。",
            ["缺少真实依据"],
            complete=complete,
        )
    )
    assert repaired is not None
    assert "可以适量食用" in repaired.text
    assert "[source:" not in repaired.text
    assert "模型通用知识" in repaired.text


# ----- Skill 拼入 -----


def test_verification_prompt_loads_skill() -> None:
    prompt = build_verification_system_prompt(branch="full_recommend")
    assert "source_id" in prompt
    assert "候选评估" in prompt
    assert "不改写" in prompt or "只拒绝" in prompt


def test_verify_marks_skill_in_prompt() -> None:
    result = _run(
        verify(
            [SuggestionItem(text="有依据 [source: a1]。", item_id="x")],
            available_source_ids=["a1"],
            run_llm_soft_checks=False,
        )
    )
    assert result.skill_in_prompt is True
    assert "source_id" in result.system_prompt


# ----- D25 候选评估 -----


def test_candidate_eval_allows_conclusion_without_source_if_reason_has_one() -> None:
    items = [
        SuggestionItem(text="结论：现在不建议选黄焖鸡。", item_id="conclusion"),
        SuggestionItem(
            text="理由：油腻助湿 [source: tcm_000010]。",
            item_id="reason",
        ),
    ]
    accepted, rejected = apply_deterministic_source_check(
        items,
        available_source_ids=["tcm_000010"],
        branch="candidate_eval",
    )
    assert {a.item_id for a in accepted} == {"conclusion", "reason"}
    assert rejected == []


def test_candidate_eval_rejects_when_no_supporting_source() -> None:
    items = [
        SuggestionItem(text="结论：选米线。", item_id="conclusion"),
        SuggestionItem(text="理由：比较清淡。", item_id="reason"),
    ]
    accepted, rejected = apply_deterministic_source_check(
        items,
        available_source_ids=["tcm_000010"],
        branch="candidate_eval",
    )
    assert accepted == []
    assert len(rejected) == 2
    assert all("支持理由" in r.reason for r in rejected)


# ----- ED 确定性 -----


def test_ed_numeric_calorie_hard_blocked() -> None:
    result = _run(
        verify(
            [
                SuggestionItem(
                    text="每天只吃 1200 大卡 [source: n1]。",
                    item_id="ed",
                )
            ],
            available_source_ids=["n1"],
            run_llm_soft_checks=False,
        )
    )
    assert result.accepted == []
    assert result.rejected[0].check_number == 6
    assert result.rejected[0].action == "hard_block"


def test_ed_check_uses_guardrails_module_not_narrow_inline_regex() -> None:
    """之前内联的窄正则只挡"数字+kcal/大卡"这一种形态；现在复用
    backend/guardrails/ed_protection.py（THREAT_MODEL.md E3 打磨过），应该也能
    挡住 BMI/千分位数字/"两位数的千卡"这类换皮表述。"""
    for text in ["BMI 18.5 [source: n1]。", "1,200 Cal per day [source: n1]。", "建议大约两位数的千卡 [source: n1]。"]:
        result = _run(
            verify(
                [SuggestionItem(text=text, item_id="ed")],
                available_source_ids=["n1"],
                run_llm_soft_checks=False,
            )
        )
        assert result.accepted == [], f"应被 ED 防护拦下: {text!r}"
        assert result.rejected[0].check_number == 6


# ----- 过敏原确定性 -----


def test_allergen_hit_is_hard_blocked_not_regenerated() -> None:
    result = _run(
        verify(
            [SuggestionItem(text="推荐加一勺蚝油提鲜 [source: n1]。", item_id="a1")],
            available_source_ids=["n1"],
            run_llm_soft_checks=False,
            user_allergens=["甲壳类"],
        )
    )
    assert result.accepted == []
    assert result.rejected[0].check_number == 4
    assert result.rejected[0].action == "hard_block"
    assert result.rejected[0].allergen_names == ("甲壳类",)
    assert "不重生成" in result.rejected[0].reason


def test_skip_allergen_check_passes_hit_after_reconciliation_retry() -> None:
    """After an allergen reconciliation retry, check 4 is skipped; ED/source still run."""
    result = _run(
        verify(
            [SuggestionItem(text="推荐加一勺蚝油提鲜 [source: n1]。", item_id="a1")],
            available_source_ids=["n1"],
            run_llm_soft_checks=False,
            user_allergens=["甲壳类"],
            skip_allergen_check=True,
        )
    )
    assert [a.item_id for a in result.accepted] == ["a1"]
    assert result.rejected == []


def test_allergen_check_skipped_when_no_allergens_passed() -> None:
    """空 user_allergens 不代表"用户没有过敏原"，只代表调用方没传——这种情况下
    不应该悄悄挡住任何东西(那会是另一种错误：把"不知道"当成"命中")。"""
    result = _run(
        verify(
            [SuggestionItem(text="推荐加一勺蚝油提鲜 [source: n1]。", item_id="a1")],
            available_source_ids=["n1"],
            run_llm_soft_checks=False,
        )
    )
    assert [a.item_id for a in result.accepted] == ["a1"]


def test_allergen_not_matching_user_list_passes() -> None:
    result = _run(
        verify(
            [SuggestionItem(text="推荐加一勺蚝油提鲜 [source: n1]。", item_id="a1")],
            available_source_ids=["n1"],
            run_llm_soft_checks=False,
            user_allergens=["花生"],
        )
    )
    assert [a.item_id for a in result.accepted] == ["a1"]


# ----- 诊断性表述确定性（ARCHITECTURE §5.4，编号 8） -----


def test_diagnostic_statement_is_blocked() -> None:
    result = _run(
        verify(
            [SuggestionItem(text="根据你的描述，你患有胃炎 [source: n1]。", item_id="d1")],
            available_source_ids=["n1"],
            run_llm_soft_checks=False,
        )
    )
    assert result.accepted == []
    assert result.rejected[0].check_number == 8
    assert result.rejected[0].action == "hard_block"
