#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCMQ 简版计分逻辑：转化分公式 + 体质夹杂判定（含平和质单独一套规则）。

设计依据：docs/ARCHITECTURE.md §11.3
决策依据：docs/DECISIONS.md D28
题库：backend/skills/ccmq_questionnaire.md（每类体质 5 条，共 45 条，改写简化版——
不是官方 60 条量表的逐字复刻，理由见该文件顶部说明）

⚠️ 判定规则不是九类体质统一一套：8 种偏颇体质用 ≥40/30-40/<30 三档；平和质单独一套
（≥60 且其余 8 种均 <30 → "是"；≥60 且其余 8 种均 <40 → "基本是"；否则 "否"）——这是
2026-08-26 对照官方《中医体质分类与判定》标准（中华中医药学会标准）核实后，对
ARCHITECTURE.md §11.3 原先"九类统一"的简化表述做的修订，本文件实现的是修订后的版本。

本模块只负责"给定每题分值 → 算出转化分 → 判定体质"这一段纯函数逻辑，不负责对话式
收集题目、不负责把结果写进 user_profile——那是 backend/onboarding/flow.py（仍是占位，
见该文件模块文档）和 /api/onboarding/* 端点（仍未实现）的职责，不在本次改动范围。
"""
from __future__ import annotations

from dataclasses import dataclass

# 九类体质，顺序对齐 ARCHITECTURE.md §11.3 行文顺序。ping_he 是唯一走特殊判定规则的类型。
PING_HE = "ping_he"
CONSTITUTIONS: tuple[str, ...] = (
    PING_HE,
    "qi_xu",
    "yang_xu",
    "yin_xu",
    "tan_shi",
    "shi_re",
    "xue_yu",
    "qi_yu",
    "te_bing",
)

_PATHOLOGICAL = tuple(c for c in CONSTITUTIONS if c != PING_HE)

# 每类体质的题目数（backend/skills/ccmq_questionnaire.md），用于校验答案数组长度。
ITEMS_PER_CONSTITUTION = 5

_MIN_ITEM_SCORE = 1
_MAX_ITEM_SCORE = 5

# 判定档位常量，避免各处散落魔法字符串。
VERDICT_YES = "是"
VERDICT_BASICALLY_YES = "基本是"  # 仅平和质会出现这一档
VERDICT_LEANING_YES = "倾向是"  # 仅 8 种偏颇体质会出现这一档
VERDICT_NO = "否"

# 主/次体质候选门槛：命中这些档位的体质会进入 constitution / constitution_secondary。
_PRIMARY_CANDIDATE_VERDICTS = (VERDICT_YES, VERDICT_BASICALLY_YES)
_SECONDARY_CANDIDATE_VERDICTS = (VERDICT_YES, VERDICT_BASICALLY_YES, VERDICT_LEANING_YES)


def raw_to_transformed(raw_score: int, item_count: int = ITEMS_PER_CONSTITUTION) -> float:
    """官方标准转化分公式：[(原始分-条目数)/(条目数×4)]×100，映射到 0-100。"""
    if item_count <= 0:
        raise ValueError(f"item_count must be positive, got {item_count!r}")
    return (raw_score - item_count) / (item_count * 4) * 100


def _pathological_verdict(score: float) -> str:
    if score >= 40:
        return VERDICT_YES
    if score >= 30:
        return VERDICT_LEANING_YES
    return VERDICT_NO


def _ping_he_verdict(score: float, other_scores: dict[str, float]) -> str:
    if score >= 60 and all(v < 30 for v in other_scores.values()):
        return VERDICT_YES
    if score >= 60 and all(v < 40 for v in other_scores.values()):
        return VERDICT_BASICALLY_YES
    return VERDICT_NO


@dataclass(frozen=True)
class ConstitutionScore:
    constitution: str
    raw_score: int
    transformed_score: float
    verdict: str


@dataclass(frozen=True)
class CcmqResult:
    scores: dict[str, ConstitutionScore]
    primary: str | None  # 转化分最高的候选体质；全部落在"否"时为 None
    secondary: tuple[str, ...]  # 其余候选，按转化分从高到低排列


def _validate_answers(answers: dict[str, list[int]]) -> None:
    missing = [c for c in CONSTITUTIONS if c not in answers]
    if missing:
        raise ValueError(f"missing answers for constitutions: {missing}")
    unknown = [c for c in answers if c not in CONSTITUTIONS]
    if unknown:
        raise ValueError(f"unknown constitution ids: {unknown}")
    for constitution, items in answers.items():
        if len(items) != ITEMS_PER_CONSTITUTION:
            raise ValueError(
                f"{constitution!r}: expected {ITEMS_PER_CONSTITUTION} item scores, got {len(items)}"
            )
        for v in items:
            if not (_MIN_ITEM_SCORE <= v <= _MAX_ITEM_SCORE):
                raise ValueError(
                    f"{constitution!r}: item score {v!r} out of range "
                    f"[{_MIN_ITEM_SCORE}, {_MAX_ITEM_SCORE}]"
                )


def score_ccmq(answers: dict[str, list[int]]) -> CcmqResult:
    """给定九类体质各自的 5 题原始分值，算出转化分并判定主/次候选体质。

    `answers`：`{constitution_id: [item1..item5]}`，每个分值 1-5（5 级李克特量表，
    "不确定"记 3 分，见 ccmq_questionnaire.md）。九类体质必须全部给出答案——分批对话
    收集完整题目后再一次性调用本函数，不支持增量部分计分（体质夹杂判定需要看到全部
    9 个转化分，平和质规则尤其依赖"其余 8 种均 <30/40"这个横向比较）。
    """
    _validate_answers(answers)

    transformed: dict[str, float] = {
        c: raw_to_transformed(sum(answers[c])) for c in CONSTITUTIONS
    }

    scores: dict[str, ConstitutionScore] = {}
    for c in _PATHOLOGICAL:
        verdict = _pathological_verdict(transformed[c])
        scores[c] = ConstitutionScore(c, sum(answers[c]), transformed[c], verdict)

    other_scores = {c: transformed[c] for c in _PATHOLOGICAL}
    ping_he_verdict = _ping_he_verdict(transformed[PING_HE], other_scores)
    scores[PING_HE] = ConstitutionScore(
        PING_HE, sum(answers[PING_HE]), transformed[PING_HE], ping_he_verdict
    )

    candidates = [c for c in CONSTITUTIONS if scores[c].verdict in _SECONDARY_CANDIDATE_VERDICTS]
    candidates.sort(key=lambda c: scores[c].transformed_score, reverse=True)

    primary: str | None = None
    secondary: tuple[str, ...] = ()
    if candidates:
        top = candidates[0]
        if scores[top].verdict in _PRIMARY_CANDIDATE_VERDICTS:
            primary = top
            secondary = tuple(candidates[1:])
        else:
            # 最高分的候选也只是"倾向是"（没有任何体质摸到"是"/"基本是"门槛）：
            # 按 ARCHITECTURE §11.3，"倾向是"只进 constitution_secondary，不能顶替
            # primary——这种情况下 primary 留空，全部候选都进 secondary。
            secondary = tuple(candidates)

    return CcmqResult(scores=scores, primary=primary, secondary=secondary)
