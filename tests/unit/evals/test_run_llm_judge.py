"""
测试目标：`evals/run_llm_judge.py` 里不碰网络的纯函数——JSON 解析容错、分数
裁剪、ground truth 拼接。真实打分(真的调 LLM)不在单测范围内，那是
`evals/run_llm_judge.py` 本身跑起来才验证的事。
对应实现：evals/run_llm_judge.py
覆盖要求：常规
"""
from __future__ import annotations

from evals.run_llm_judge import (
    _clamp_score,
    _ground_truth_block,
    _normalize_scores,
    _strip_json_fences,
    _total,
)


def test_clamp_score_clamps_out_of_range_values():
    assert _clamp_score(5, 0, 2) == 2
    assert _clamp_score(-1, 0, 2) == 0
    assert _clamp_score(1, 0, 2) == 1


def test_clamp_score_defaults_to_low_on_non_numeric():
    assert _clamp_score("not a number", 0, 2) == 0
    assert _clamp_score(None, 0, 1) == 0


def test_normalize_scores_clamps_each_metric_independently():
    raw = {
        "relation_correct": 3,  # out of range, max is 1
        "resolution_correct": -1,
        "safety": 1,
        "synthesis": 2,
        "evidence_honesty": "bad",
        "rationale": "x" * 500,
    }
    out = _normalize_scores(raw)
    assert out["relation_correct"] == 1
    assert out["resolution_correct"] == 0
    assert out["safety"] == 1
    assert out["synthesis"] == 2
    assert out["evidence_honesty"] == 0
    assert len(out["rationale"]) <= 300


def test_total_sums_all_five_metrics():
    scores = {
        "relation_correct": 1,
        "resolution_correct": 2,
        "safety": 1,
        "synthesis": 2,
        "evidence_honesty": 2,
        "rationale": "ok",
    }
    assert _total(scores) == 8


def test_strip_json_fences_removes_markdown_code_block():
    text = "```json\n{\"a\": 1}\n```"
    assert _strip_json_fences(text) == '{"a": 1}'


def test_strip_json_fences_leaves_plain_json_untouched():
    text = '{"a": 1}'
    assert _strip_json_fences(text) == '{"a": 1}'


def test_ground_truth_block_uses_conflict_rule_when_rule_id_present():
    case_row = {"expect": {"relation": "conflict", "must_not": ["足够补铁"]}, "rule_id": "B01"}
    rules = {
        "B01": {
            "tcm_position": "红枣补气血",
            "nutrition_position": "非血红素铁吸收率低",
            "resolution": "概念不对等",
            "resolution_rationale": "同名不同指",
            "confidence": "high",
            "evidence_level": "双边实证",
        }
    }
    block = _ground_truth_block(case_row, rules)
    assert "概念不对等" in block
    assert "同名不同指" in block
    assert "足够补铁" in block


def test_ground_truth_block_falls_back_to_expect_when_no_rule_id():
    case_row = {
        "expect": {"handling": "decline_or_limit", "resolution_keywords": ["长期", "不建议"], "must_not": []},
        "notes": "表外：类似 W08",
    }
    block = _ground_truth_block(case_row, {})
    assert "decline_or_limit" in block
    assert "表外：类似 W08" in block
