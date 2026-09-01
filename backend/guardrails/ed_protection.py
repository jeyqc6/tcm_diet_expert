#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED 防护四条：确定性规则，不调 LLM。

设计依据：docs/PRD.md §10 / §16；docs/ARCHITECTURE.md §5.4；DECISIONS D16
威胁模型：docs/THREAT_MODEL.md E3
话术来源：docs/prompts/disclaimers.md §9 / §10（命中后走模板，不临场生成）
roadmap：阶段 5；完成判据是数值化表述 100% 硬拦截

四条（PRD §10，优先级从上到下——命中更高优先级时用那条的模板）：
  4. 用户自述极低摄入 / 体重焦虑 → 停止一切限制性建议，转介专业支持
  3. 用户索要热量或体重目标 → 说明不提供数值目标及原因，给定性替代
  1. 输出里的数值化体重/热量表述 → 硬拦截
  2. 极端限制性表述 → 拦截，不生成断食/戒断主食类方案

本文件只做检测 + 返回审阅过的模板。不改写模型原文、不调用 LLM 重生成——
「拦截后重生成为温和版本」是调用方用本模块的模板替换，不是这里再打一次模型
（THREAT_MODEL E3：命中后走模板，不重生成自由文本）。

不在这里接线到 api/main.py / verification.py：核查 pass 里仍有一段更窄的
数值正则；阶段 5 其余 guardrails 未动。调用方以后改 import 即可。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.i18n import normalize_locale

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class EdRule(str, Enum):
    NUMERIC_METRIC = "numeric_metric"  # PRD row 1
    EXTREME_RESTRICTION = "extreme_restriction"  # PRD row 2
    REQUEST_NUMERIC_TARGET = "request_numeric_target"  # PRD row 3
    DISTRESS_SELF_REPORT = "distress_self_report"  # PRD row 4


class EdAction(str, Enum):
    HARD_BLOCK = "hard_block"
    CANNED_REFUSAL = "canned_refusal"
    STOP_RESTRICTIVE = "stop_restrictive"


# Higher number = higher priority. Distress outranks everything else.
_PRIORITY: dict[EdRule, int] = {
    EdRule.DISTRESS_SELF_REPORT: 4,
    EdRule.REQUEST_NUMERIC_TARGET: 3,
    EdRule.NUMERIC_METRIC: 2,
    EdRule.EXTREME_RESTRICTION: 1,
}


@dataclass(frozen=True)
class EdHit:
    rule: EdRule
    action: EdAction
    matched: str
    reason: str


@dataclass(frozen=True)
class EdCheckResult:
    hits: tuple[EdHit, ...] = ()

    @property
    def blocked(self) -> bool:
        return bool(self.hits)

    @property
    def primary(self) -> EdHit | None:
        if not self.hits:
            return None
        return max(self.hits, key=lambda h: _PRIORITY[h.rule])

    @property
    def canned_response(self) -> str | None:
        hit = self.primary
        if hit is None:
            return None
        return _TEMPLATES[hit.rule]

    def canned_response_for_locale(self, locale: str = "zh") -> str | None:
        hit = self.primary
        if hit is None:
            return None
        if normalize_locale(locale) == "en":
            return _TEMPLATES_EN[hit.rule]
        return _TEMPLATES[hit.rule]


# ---------------------------------------------------------------------------
# Reviewed templates (disclaimers.md §9 / §10). Do not invent wording here.
# ---------------------------------------------------------------------------

# Rule 1 + 3: no numeric targets; give a qualitative alternative (D16).
CANNED_NUMERIC_TARGET = (
    "本工具不提供热量或体重的数值目标——这类数字容易变成压力来源，"
    "而且脱离个体情况的数字并不可靠。\n\n"
    "我可以从另一个角度帮你：近期若饮食偏肥甘厚味，建议这周多用蒸煮、"
    "增加赤小豆薏米这类利湿食材，而不是盯一个热量额度或目标体重。"
)

# Rule 2: refuse to assemble an extreme-restriction plan.
CANNED_EXTREME_RESTRICTION = (
    "本工具不提供极端限制性的饮食安排（断食、戒断主食、只吃某一种食物这类）。\n\n"
    "可以从另一个角度：若最近吃得偏油腻厚味，这周多用蒸煮、搭配一些利湿的食材，"
    "而不是把某一类食物从桌子上拿掉。"
)

# Rule 4: stop ALL restrictive advice. Highest priority. No meal plan, no direction.
CANNED_DISTRESS = (
    "听起来你最近在饮食上给自己的压力不小。\n"
    "我先不给限制性的建议了——这种时候更重要的是有人一起看看情况。\n"
    "如果你愿意，可以和信任的人或专业人士聊聊。"
)

CANNED_NUMERIC_TARGET_EN = (
    "This tool does not provide numeric calorie or weight targets — those numbers "
    "easily become a source of pressure, and a figure taken out of personal context "
    "is not reliable.\n\n"
    "I can help from another angle: if recent meals have been rich and greasy, "
    "lean toward steaming and boiling this week and add dampness-draining foods "
    "such as adzuki bean and coix seed, rather than chasing a calorie quota or a target weight."
)

CANNED_EXTREME_RESTRICTION_EN = (
    "This tool does not assemble extreme-restriction eating plans "
    "(fasting, cutting out staples, eating only one food, and similar).\n\n"
    "Another angle: if recent meals have been oily and heavy, use more steaming "
    "and boiling this week and pair some dampness-draining ingredients, "
    "rather than taking a whole food group off the table."
)

CANNED_DISTRESS_EN = (
    "It sounds like food has been putting a lot of pressure on you lately.\n"
    "I will not give restrictive advice right now — what matters more is having "
    "someone look at the situation with you.\n"
    "If you are willing, consider talking with someone you trust or a professional."
)

_TEMPLATES: dict[EdRule, str] = {
    EdRule.NUMERIC_METRIC: CANNED_NUMERIC_TARGET,
    EdRule.REQUEST_NUMERIC_TARGET: CANNED_NUMERIC_TARGET,
    EdRule.EXTREME_RESTRICTION: CANNED_EXTREME_RESTRICTION,
    EdRule.DISTRESS_SELF_REPORT: CANNED_DISTRESS,
}

_TEMPLATES_EN: dict[EdRule, str] = {
    EdRule.NUMERIC_METRIC: CANNED_NUMERIC_TARGET_EN,
    EdRule.REQUEST_NUMERIC_TARGET: CANNED_NUMERIC_TARGET_EN,
    EdRule.EXTREME_RESTRICTION: CANNED_EXTREME_RESTRICTION_EN,
    EdRule.DISTRESS_SELF_REPORT: CANNED_DISTRESS_EN,
}

_RULE_ACTIONS: dict[EdRule, EdAction] = {
    EdRule.NUMERIC_METRIC: EdAction.HARD_BLOCK,
    EdRule.EXTREME_RESTRICTION: EdAction.CANNED_REFUSAL,
    EdRule.REQUEST_NUMERIC_TARGET: EdAction.CANNED_REFUSAL,
    EdRule.DISTRESS_SELF_REPORT: EdAction.STOP_RESTRICTIVE,
}

_RULE_REASONS: dict[EdRule, str] = {
    EdRule.NUMERIC_METRIC: "numeric calorie/weight/BMI/body-fat expression (PRD §10 ED-1)",
    EdRule.EXTREME_RESTRICTION: "extreme restriction language (PRD §10 ED-2)",
    EdRule.REQUEST_NUMERIC_TARGET: "user asked for a calorie or weight target (PRD §10 ED-3)",
    EdRule.DISTRESS_SELF_REPORT: "self-reported very low intake or weight anxiety (PRD §10 ED-4)",
}


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

def _p(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# Arabic or Chinese numerals, including 1,200 thousands separators.
_NUM = r"(?:\d{1,5}(?:,\d{3})*(?:\.\d+)?|[一二三四五六七八九十百千万两]+)"

_CAL_UNIT = r"(?:kcal|cals?\b|calories?\b|大卡|千卡|卡路里|(?<![大千])卡(?!路))"
_WT_UNIT = r"(?:kg|kilos?\b|kilograms?\b|公斤|斤|磅|lbs?\b|pounds?\b)"


def _collect(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in pat.finditer(text):
            span = m.group(0).strip()
            if span and span not in seen:
                seen.add(span)
                found.append(span)
    return found


# ----- Rule 1: numeric metrics in model OUTPUT -----
# Covers the shapes verification.py already caught, plus THREAT_MODEL E3
# bypasses: 1,200 Cal / BMI 18.5 / 体脂 12% / 热量缺口 500 / 大约两位数的千卡.

_NUMERIC_PATTERNS = [
    # number + calorie unit (1200 大卡 / 1500 kcal / 1,200 Cal / 500卡)
    _p(rf"{_NUM}\s*{_CAL_UNIT}"),
    # 每天/每日 + 只吃/摄入 + number + calorie
    _p(rf"(?:每天|每日|一天|per\s*day|daily)\s*(?:只吃|摄入|吃|eat)?\s*{_NUM}\s*{_CAL_UNIT}"),
    # calorie deficit / 热量缺口
    _p(rf"(?:热量缺口|热量额度|卡路里缺口)\s*(?:为|是|of)?\s*{_NUM}"),
    _p(rf"calorie\s*deficit\s*(?:of\s*)?{_NUM}"),
    _p(rf"(?:tdee|bmr)\s*[:=]?\s*{_NUM}"),
    # circumlocution that drops the digit (THREAT_MODEL: 大约两位数的千卡)
    _p(r"(?:两位数|三位数|大约|大概).{0,8}(?:千卡|大卡|kcal|卡路里|热量额度)"),
    _p(r"(?:kcal|calorie|千卡|大卡).{0,6}(?:额度|目标|budget|target)"),
    # BMI
    _p(rf"\bbmi\s*[:=]?\s*{_NUM}"),
    _p(rf"体质指数\s*[:=]?\s*{_NUM}"),
    _p(rf"BMI\s*目标\s*{_NUM}?"),
    # body fat %
    _p(rf"体脂\s*(?:率\s*)?{_NUM}\s*%?"),
    _p(rf"body\s*fat\s*(?:percentage\s*)?{_NUM}\s*%?"),
    # weight goal / loss / rate — require a goal/loss verb so "200g 鸡胸" does not fire.
    # ⚠️ 真实跑通时发现的坑：`{_WT_UNIT}` 后面原来带着 `?`(可选)，本意是想多接住
    # "瘦到90"这种没写单位的说法，但代价是"降到"/"控制在"这类动词只要后面跟
    # 一个数字就命中——不管那个数字是体重、气温、价格还是别的什么("气温又
    # 下降到17.7°C"、"价格控制在200元"都会被误判成体重目标)。这两处都收紧成
    # 必须真的带一个体重单位(kg/斤/公斤/磅/lbs)才算命中——现有全部测试用例
    # (穷举 THREAT_MODEL E3 的 MUST_BLOCK 列表)本来就每条都带着显式单位，收紧
    # 不会漏检任何一条已知场景，只是不再对"没写单位、也不是体重"的数字瞎猜。
    _p(
        rf"(?:减到|降到|减至|瘦到|控制在|减掉|减重|目标体重|体重(?:降|减|控制)?)"
        rf"\s*{_NUM}\s*{_WT_UNIT}"
    ),
    _p(rf"(?:lose|lost|drop(?:\s*to)?|down\s*to|weigh(?:t)?\s*(?:goal|target)?)\s*{_NUM}\s*{_WT_UNIT}"),
    _p(rf"(?:每周|每天|每日|per\s*week|a\s*week)\s*(?:减|瘦|lose)\s*{_NUM}\s*{_WT_UNIT}"),
    _p(rf"{_NUM}\s*{_WT_UNIT}\s*(?:的)?(?:目标|体重目标|goal)"),
    # 体重 + number + unit in generated advice ("建议体重 50kg")
    _p(rf"体重\s*{_NUM}\s*{_WT_UNIT}"),
]

# ----- Rule 2: extreme restriction (staple / all-food, not allergen avoidance) -----

_STAPLE = (
    r"(?:主食|碳水|碳水化合物|谷物|米饭|米面|粮食|淀粉|grains?|carbs?|staples?|solid food)"
)

_EXTREME_PATTERNS = [
    _p(rf"戒断\s*{_STAPLE}"),
    _p(r"(?:断食|绝食|辟谷)(?:\s*(?:[一二三四五六七八九十两]+\s*天|\d+\s*(?:天|days?)))?"),
    _p(r"(?:water|juice)\s*fast(?:ing)?(?:\s*for\s*\d+\s*days?)?"),
    _p(r"\bfasting\s*for\s*\d+\s*days?\b"),
    _p(rf"完全不碰\s*{_STAPLE}"),
    _p(rf"(?:cut out|give up|eliminate)\s+all\s+{_STAPLE}"),
    _p(r"只吃\s*(?:水煮|菜叶|菜叶子|生菜|清水|白菜|水果|苹果)"),
    _p(r"eat nothing but\s+\w+"),
    _p(r"only eat\s+(?:lettuce|leaves|water|fruit|cabbage|boiled)"),
    _p(r"(?:零碳水|去碳水|无碳水|zero[\s-]?carb|no[\s-]?carb)"),
    _p(r"(?:一天只吃一顿|每日只吃一顿|one meal a day|\bomad\b)"),
    _p(r"(?:一周|七天|整周).{0,12}只吃"),
    _p(r"(?:帮我|给我|安排).{0,16}(?:只吃|断食|绝食|辟谷|戒断主食)"),
]

# ----- Rule 3: user asks for numeric calorie / weight targets -----

_REQUEST_PATTERNS = [
    _p(r"(?:给我|帮我|我要|我想要).{0,16}(?:热量|卡路里|kcal|缺口|热量额度|目标体重)"),
    _p(r"(?:每天|每日|一天).{0,8}(?:多少|几)\s*(?:卡|大卡|千卡|kcal|calories?)"),
    _p(r"(?:目标体重|体重目标|热量目标|热量额度)"),
    _p(r"(?:不要定性|我就要|不要模糊).{0,12}(?:bmi|热量|数字|kcal|体重)"),
    _p(r"(?:bmi|热量).{0,6}(?:和|与|跟).{0,6}(?:每日热量|热量|bmi|体重)"),
    _p(r"how many calories\b"),
    _p(r"(?:calorie|kcal|tdee|bmr)\s*(?:target|goal|budget|should i)"),
    _p(r"(?:weight|bmi)\s*(?:target|goal)\b"),
    _p(r"set (?:my )?weight (?:to|goal)"),
    # supplying a calorie figure as the plan they want the system to follow
    _p(rf"{_NUM}\s*{_CAL_UNIT}"),
    _p(rf"(?:热量缺口|calorie\s*deficit)\s*(?:of\s*)?{_NUM}"),
    _p(rf"(?:减到|降到|减至|瘦到|控制在)\s*{_NUM}\s*{_WT_UNIT}"),
    _p(rf"\bbmi\s*[:=]?\s*{_NUM}"),
]

# ----- Rule 4: distress / very low intake self-report -----

_DISTRESS_PATTERNS = [
    _p(r"(?:还是|仍然|依旧)觉得胖"),
    _p(r"(?:再严一点|再严格|再狠一点|再少一点)"),
    _p(r"(?:我已经|已经).{0,12}(?:一天只吃一顿|只吃一顿|不怎么吃|不敢吃)"),
    _p(r"(?:吃了就(?:想)?吐|吃完就(?:想)?吐|催吐|导泻)"),
    _p(r"(?:吃了就内疚|一吃就内疚|不敢吃|恐惧进食)"),
    _p(r"(?:已经很瘦).{0,8}还(?:想|要)再?减"),
    _p(r"骨头(?:都)?(?:突|露)"),
    _p(r"体重焦虑"),
    _p(r"still feel(?:ing)? fat"),
    _p(r"(?:even )?(?:more )?(?:strict|stricter|harsher)"),
    _p(r"(?:make it|go)\s+(?:stricter|harsher|tighter)"),
    _p(r"only eat(?:ing)? (?:once|one meal).{0,24}(?:still|feel fat)"),
    _p(r"(?:guilty|guilt) after eat"),
    _p(r"\b(?:purging|purge|laxatives?)\b"),
    _p(r"afraid to eat"),
    _p(r"(?:too|so) (?:skinny|thin).{0,24}(?:lose|cut)"),
]


def _hits_for(rule: EdRule, text: str, patterns: list[re.Pattern[str]]) -> list[EdHit]:
    return [
        EdHit(
            rule=rule,
            action=_RULE_ACTIONS[rule],
            matched=span,
            reason=_RULE_REASONS[rule],
        )
        for span in _collect(patterns, text)
    ]


def _result_from(hits: list[EdHit]) -> EdCheckResult:
    return EdCheckResult(hits=tuple(hits))


def scan_model_output(text: str) -> EdCheckResult:
    """Rules 1 + 2 on generated advice. Food-gram amounts (200g 鸡胸) do not fire."""
    text = text or ""
    if not text.strip():
        return EdCheckResult()
    hits = _hits_for(EdRule.NUMERIC_METRIC, text, _NUMERIC_PATTERNS)
    hits.extend(_hits_for(EdRule.EXTREME_RESTRICTION, text, _EXTREME_PATTERNS))
    return _result_from(hits)


def scan_user_input(text: str) -> EdCheckResult:
    """Rules 2 + 3 + 4 on the user utterance.

    A calorie figure in the user message is treated as asking for a numeric
    target (rule 3), not as leaked model output (rule 1). Current-weight
    self-description like「我体重 80 公斤」does not match rule 3.
    """
    text = text or ""
    if not text.strip():
        return EdCheckResult()
    hits = _hits_for(EdRule.DISTRESS_SELF_REPORT, text, _DISTRESS_PATTERNS)
    hits.extend(_hits_for(EdRule.REQUEST_NUMERIC_TARGET, text, _REQUEST_PATTERNS))
    hits.extend(_hits_for(EdRule.EXTREME_RESTRICTION, text, _EXTREME_PATTERNS))
    return _result_from(hits)


def evaluate_ed_protection(
    *,
    user_message: str = "",
    model_output: str | None = None,
) -> EdCheckResult:
    """Combine input + output scans. Primary rule decides the canned template.

    Callers that only have the user turn (before generation) can omit
    model_output — rule 4/3/2 on input are enough to skip the pipeline.
    """
    hits: list[EdHit] = list(scan_user_input(user_message).hits)
    if model_output:
        hits.extend(scan_model_output(model_output).hits)
    return _result_from(hits)


def contains_numeric_metric(text: str) -> bool:
    """Convenience for callers that only need the 100% numeric hard-block bit."""
    return any(h.rule is EdRule.NUMERIC_METRIC for h in scan_model_output(text).hits)
