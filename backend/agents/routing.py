#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中枢 agent：七条分支路由判断（记录 / 记录回顾 / 事实查询 / 候选评估 / 单领域 /
完整推荐 / 其他）+ 一句话多意图切分。

设计依据：docs/ARCHITECTURE.md §5.1；决策依据：docs/DECISIONS.md D12 / D25
roadmap：阶段 4.2 任务 6

机制选择（落实 DECISIONS「待决问题」表里「路由判断由谁完成」）：
  先用**确定性规则**（中英双语正则）分类，命中即返回，不调 LLM。
  规则按「更具体优先」级联。未命中时不再直接兜底完整推荐，而是走一次
  轻量 LLM 分类（`classify_route_async`）；LLM 失败/解析失败才兜底
  full_recommend。`classify_route(query) -> RouteDecision` 仍是纯规则、
  同步、可复现（M13）；异步包装才碰 LLM。

易混淆边界（D25 点名不要合并）：
  - 记录回顾 vs 事实查询：查 diet_log vs 查 knowledge_chunks
  - 候选评估 vs 完整推荐：评估用户给定候选 vs 从零生成方案

2026-08-28：从 `backend/agents/router.py` 拆出——那个文件原本同时装着"六条
分支路由判断"和"Agent Loop"两件互不相关的事，Agent Loop 部分搬去了
`backend/agents/agent_loop.py`。纯粹搬文件，不改变任何函数签名/行为。
`CompleteFn` 类型别名同一次搬到了 `backend/llm/adapter.py`（它描述的是
`complete()` 本身的签名，和路由判断无关，只是此前顺手就近定义在这里）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

from backend.llm import adapter as llm_adapter
from backend.llm.adapter import CompleteFn
from backend.observability.redact import redact_text
from backend.observability.tracing import observation, stage_log, update_current

logger = logging.getLogger("diet_expert.agents.router")


class RouteBranch(str, Enum):
    LOG_WRITE = "log_write"  # 记录
    LOG_REVIEW = "log_review"  # 记录回顾
    FACT_QUERY = "fact_query"  # 事实查询
    CANDIDATE_EVAL = "candidate_eval"  # 候选评估
    SINGLE_DOMAIN = "single_domain"  # 单领域
    FULL_RECOMMEND = "full_recommend"  # 完整推荐
    OTHER = "other"  # 不属于以上任何一种(2026-08-27 新增,D33/PRD §17)


@dataclass(frozen=True)
class RouteDecision:
    branch: RouteBranch
    reason: str
    """Which rule / pattern family fired — for M13 debugging, not shown to users."""
    domain_hint: str | None = None
    """"tcm" | "nutrition" | None — 只在 fact_query/single_domain 这两条"只查一侧
    知识库"的分支上有意义(§5.3:"只走单库检索retrieve_tcm或retrieve_nutrition")。
    classify_route 本身此前没有产出这个字段——six-branch 分类只回答"走哪条分支"，
    没回答"分支内部该查哪一侧"，这是 api/main.py 需要真正调用 retrieve_tcm 还是
    retrieve_nutrition 时必须知道的信息，不能靠 API 层自己再猜一遍。"""
    rule_matched: bool = True
    """False when no regex fired. classify_route still reports FULL_RECOMMEND so
    the sync contract stays a six-branch enum; classify_route_async then asks
    the LLM instead of treating unmatched as a real full_recommend hit."""


# ---------------------------------------------------------------------------
# Pattern families (order within classify_route matters more than list order)
# Each family is bilingual: Chinese patterns first, English with IGNORECASE.
# ---------------------------------------------------------------------------

def _en(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# 记录：显式落库意图（陈述「吃了什么」并要求记下）
# ⚠️ 不要用裸的「记录」——「饮食记录」是名词，属于回顾分支。
# ⚠️ Do not use bare "log"/"record" — "diet log" as a noun belongs to log_review.
_LOG_WRITE = [
    re.compile(r"(帮我)?记录一下"),
    re.compile(r"(帮我)?记(一下|下)(?!录)"),
    re.compile(r"请帮我记下"),
    re.compile(r"(写入|存入|保存).{0,8}(饮食|餐|记录)"),
    re.compile(r"^(今天|今早|中午|今晚|刚才|刚刚).{0,12}吃了"),
    re.compile(r"吃了.{0,20}(记一下|帮我记|请记住|记录一下)"),
    _en(r"(please )?(help me )?(log|record|save|write down)\b.{0,24}(i |that i )?(ate|eaten|had)\b"),
    _en(r"\b(log|record|save|write)\b.{0,16}(to |into )?(my )?(diet |food |meal )?(log|record)\b"),
    _en(r"\bi (just |already )?(ate|had)\b.{0,40}\b(log|record|save|write) (it|this|that)\b"),
    _en(r"please remember.{0,12}\bi (ate|had)\b"),
]

# 记录回顾：问的是「我自己」过去吃了什么（query_diet_log）
# ⚠️ 不要用裸的「饮食记录」——「写入饮食记录」是写库意图。
_LOG_REVIEW = [
    re.compile(r"我.{0,8}(昨天|前天|上周|这周|这几天|今天|今晚|中午|早上).{0,12}吃了什么"),
    re.compile(r"(昨天|前天|上周).{0,8}(晚上|中午|早上)?.{0,6}吃了什么"),
    re.compile(r"(回顾|查(一?下)?|看看).{0,8}(我的)?(饮食|吃过|吃了什么|餐次|记录)"),
    re.compile(r"我都忘了.{0,8}吃了什么"),
    _en(r"\b(what did i|what'd i)\b.{0,24}\beat\b"),
    _en(r"\b(review|check|look( at)?|show)\b.{0,16}(my )?(diet|food|meal).{0,10}(log|record|history)\b"),
    _en(r"\bi (forgot|don't remember|do not remember)\b.{0,20}(what i )?ate\b"),
    _en(r"\b(yesterday|last week|this week|the other day)\b.{0,20}what.{0,10}(did i )?eat\b"),
]

# 候选评估：用户给出具体候选，要判断/取舍（不是从零生成）
_CANDIDATE_EVAL = [
    re.compile(r"(这个|这些|那道|那碗).{0,10}(能|可以|可不可以|能不能).{0,4}吃"),
    re.compile(r"(能|可以|可不可以|能不能)吃.{0,6}(这个|这些|吗)"),
    re.compile(r"(选哪个|选哪一个|哪个更好|哪个合适|二选一)"),
    re.compile(r".{1,20}(和|还是|或者).{1,20}(选哪个|选哪|哪个)"),
    re.compile(r"(已经吃了|刚吃了|吃过).{0,20}(还能|还能不能|还能吃|不能吃什么)"),
    re.compile(r"(楼下|菜单上|眼前).{0,20}(有|是).{0,30}(选|吃哪)"),
    _en(r"\b(can i|could i|may i|is it (ok|okay|safe) to)\b.{0,8}\beat (this|these|that)\b"),
    _en(r"\b(which (one|should i)|which is (better|safer)|pick which|choose (between|which))\b"),
    _en(r".{1,30}\b(or|vs\.?|versus)\b.{1,30}\b(which|choose|pick|eat)\b"),
    _en(r"\b(already ate|just ate|already had)\b.{0,40}\b(can i still|what else can|what (can|can't|cannot) i)\b"),
    _en(r"\b(on the menu|downstairs|in front of me)\b.{0,40}\b(which|choose|pick)\b"),
]

# 事实查询：静态知识库上的单一事实（性味/归经/过敏原/成分等）
_FACT_QUERY = [
    re.compile(r"(是什么|有什么|属于什么).{0,6}(性味|性|味|归经|寒热|温凉)"),
    re.compile(r"(性味|归经|寒热属性).{0,4}(是|是什么)"),
    re.compile(r"(含(有)?|有没有|属于).{0,8}(过敏原|甲壳类|麸质|乳糖)"),
    re.compile(r"(营养成分|热量|含铁量|维C|维生素).{0,6}(是多少|高吗|怎么样)"),
    re.compile(r"(什么是|解释一下|介绍一下).{0,12}(药食同源|食疗|体质)"),
    re.compile(r".{1,12}(里面|中)(有|含).{0,8}(什么|哪些).{0,6}(过敏原|成分)"),
    _en(r"\bwhat (is|are) (the )?(nature|flavor|property|meridian|channel)\b"),
    _en(r"\b(nature|flavor|property|meridian)\b.{0,12}\bof\b"),
    _en(r"\b(contain|contains|any|have|has)\b.{0,16}\b(allergens?|gluten|lactose|shellfish|crustaceans?)\b"),
    _en(r"\b(allergens?|gluten|lactose|shellfish|crustaceans?).{0,16}\b(in|inside|of)\b"),
    _en(r"\b(calories?|nutritional? (facts?|info|information)|iron content|vitamin)\b"
        r".{0,12}\b(how much|is it|high|low)\b"),
    _en(r"\b(how much|what is|how high)\b.{0,16}\b(calories?|iron|vitamin|protein)\b"),
    _en(r"\b(what is|explain|introduce)\b.{0,24}\b(medicinal food|food as medicine|constitution)\b"),
    _en(r"\bwhat.{0,16}(allergens?|ingredients?).{0,12}\b(in|does)\b"),
]

# 单领域：明确只落在中医侧或营养侧之一（不做双侧综合「今天吃什么」）
_SINGLE_DOMAIN_TCM = [
    re.compile(r"(气虚|阳虚|阴虚|痰湿|湿热|血瘀|气郁|特禀|平和).{0,20}(质)?.{0,20}(该吃|适合吃|忌|不宜|饮食)"),
    re.compile(r"(从中医|按体质|辨体施食|食养).{0,20}(角度|看|怎么)"),
    re.compile(r"(长夏|倒春寒|秋燥).{0,16}(吃什么|饮食|忌)"),
    _en(r"\b(qi[- ]deficiency|yang[- ]deficiency|yin[- ]deficiency|phlegm[- ]damp|damp[- ]heat|"
        r"blood stasis|qi stagnation|allergic constitution|balanced constitution)\b"
        r".{0,24}\b(eat|diet|avoid|should)\b"),
    _en(r"\b(from (a )?tcm|tcm perspective|according to (my )?constitution)\b"),
    _en(r"\b(long summer|late spring cold|autumn dryness)\b.{0,20}\b(eat|diet|avoid)\b"),
]
_SINGLE_DOMAIN_NUTRITION = [
    re.compile(r"(缺铁|补铁|贫血|补钙|蛋白质|纤维).{0,16}(怎么补|吃什么|注意)"),
    re.compile(r"(从营养|按营养学|膳食指南).{0,20}(角度|看|怎么)"),
    re.compile(r"(华法林|抗凝).{0,20}(菠菜|绿叶|维K|维生素K)"),
    re.compile(r"红枣.{0,8}(能不能|可以).{0,6}(补血|纠正).{0,6}(贫血)?"),
    _en(r"\b(iron[- ]deficien\w*|anemia|iron supplement|calcium|protein|fiber)\b"
        r".{0,20}\b(how to|what to eat|diet|watch|careful)\b"),
    _en(r"\b(from (a )?nutrition|nutritional perspective|dietary guidelines)\b"),
    _en(r"\b(warfarin|anticoagulant)\b.{0,24}\b(spinach|leafy|vitamin k)\b"),
    _en(r"\b(jujube|red date)s?\b.{0,16}\b(anemia|iron[- ]deficien)\b"),
]

# fact_query 分支内部的域提示——不改变"是不是 fact_query"这个判断本身(那仍然
# 只看 _FACT_QUERY)，只在已经确定是 fact_query 之后，用简单关键词猜一下这条
# 事实性提问该查 retrieve_tcm 还是 retrieve_nutrition。用关键词而不是复用
# _SINGLE_DOMAIN_TCM/_NUTRITION 那两组正则，是因为那两组是为"这问题该不该整个
# 归为单领域分支"设计的，触发条件比"这句话提到的是哪一侧的术语"更严格，直接拿来
# 复用会让不少真实 fact_query 问法猜不出域(比如"过敏原"三个字不会让任何一条
# SINGLE_DOMAIN 正则命中，但足够判断这是营养侧的事实性问题)。
_TCM_DOMAIN_HINTS = (
    "性味", "归经", "寒热", "温凉", "药食同源", "食疗", "体质",
    "meridian", "constitution", "medicinal food", "four natures",
    "flavor and nature", "food as medicine", "nature of", "flavor of",
)
_NUTRITION_DOMAIN_HINTS = (
    "过敏原", "甲壳类", "麸质", "乳糖", "营养成分", "热量", "含铁量", "维c", "维C", "维生素", "蛋白质",
    "allergen", "gluten", "lactose", "shellfish", "crustacean",
    "calorie", "iron content", "vitamin", "protein", "nutritional",
)


def _guess_domain_hint(text: str) -> str | None:
    lowered = text.lower()
    if any(h in lowered for h in _TCM_DOMAIN_HINTS):
        return "tcm"
    if any(h in lowered for h in _NUTRITION_DOMAIN_HINTS):
        return "nutrition"
    return None


# 完整推荐：开放式综合方案（兜底前的显式模式）
_FULL_RECOMMEND = [
    re.compile(r"(今天|今晚|晚饭|午餐|早餐).{0,16}(该|应该|想|要|适合).{0,4}吃(什么|点什么)"),
    re.compile(r"(天气|气候|时令|节气|气温).{0,12}(适合吃|该吃|吃什么)"),
    re.compile(r"(吃什么|吃点什么).{0,8}(比较好|合适|推荐)"),
    re.compile(r"(加班|很晚|宵夜).{0,16}(吃什么|晚饭)"),
    re.compile(r"(帮我|给我).{0,6}(推荐|安排).{0,8}(一?天|三餐|饮食|食谱)"),
    _en(r"\bwhat (should|can|do) i eat\b.{0,20}(today|tonight|for (dinner|lunch|breakfast))?\b"),
    _en(r"\b(weather|climate)\b.{0,32}\b(eat|eating|dinner|lunch|breakfast)\b"),
    _en(r"\b(eat|eating)\b.{0,24}\b(weather|climate)\b"),
    _en(r"\b(recommend|suggest)\b.{0,20}\b(meal|diet|what to eat|a day's|three meals)\b"),
    _en(r"\b(working late|staying up|late night)\b.{0,24}\b(eat|dinner|supper)\b"),
    _en(r"\b(plan|arrange)\b.{0,16}\b(three meals|my meals|today's (food|diet|meals))\b"),
]

# other：不属于以上任何一种(2026-08-27 新增,D33/PRD §17 那条开放问题的一部分
# 解决方案)——纯问候/致谢/告别，**整句锚定匹配**，不是其余分支惯用的子串
# 搜索。用子串搜索会把"谢谢你的建议，那我该吃什么"这种后半句才是真实请求的
# 输入误判成 other；锚定匹配只在"这条消息整体就是一句寒暄"时才命中，真实
# 请求会在 classify_route 级联更前面的分支被拦下，走不到这里。
_OTHER_GREETINGS = [
    re.compile(r"^(你好|您好|哈喽|嗨)[!！。.，,]*$"),
    re.compile(r"^(谢谢|谢了|多谢|感谢)[你您]?[!！。.，,]*$"),
    re.compile(r"^(拜拜|再见|回头见)[!！。.，,]*$"),
    _en(r"^(hi|hello|hey|howdy)[!.,]*$"),
    _en(r"^thanks?( you)?[!.,]*$"),
    _en(r"^(bye|goodbye|see you)[!.,]*$"),
]


def _first_match(patterns: list[re.Pattern[str]], text: str) -> re.Pattern[str] | None:
    for p in patterns:
        if p.search(text):
            return p
    return None


def classify_route(query: str) -> RouteDecision:
    """Classify a user utterance into one of the seven branches (rules only).

    Priority (more specific first — do not reorder casually):
      1. log_review          ← before log_write so 「查…饮食记录」≠ 写入
      2. log_write
      3. candidate_eval      ← must beat full_recommend ("选哪个" ≠ 从零生成)
      4. fact_query          ← log_review already beat "我昨天吃了什么"
      5. single_domain
      6. full_recommend (explicit patterns only)
      7. other               ← 纯寒暄(整句锚定匹配)，排在 full_recommend 之后：
         真实请求即便夹了一句问候也会先被更前面的分支拦下，走到这里的只剩
         "整条消息就是一句寒暄"这一种情况(2026-08-27 新增,D33)
      unmatched → FULL_RECOMMEND with rule_matched=False (LLM fallback is
      classify_route_async's job, not this function's)
    """
    text = (query or "").strip()
    if not text:
        return RouteDecision(
            RouteBranch.FULL_RECOMMEND, reason="empty_query_default", rule_matched=False
        )

    if m := _first_match(_LOG_REVIEW, text):
        return RouteDecision(RouteBranch.LOG_REVIEW, reason=f"log_review:{m.pattern}")

    if m := _first_match(_LOG_WRITE, text):
        return RouteDecision(RouteBranch.LOG_WRITE, reason=f"log_write:{m.pattern}")

    if m := _first_match(_CANDIDATE_EVAL, text):
        return RouteDecision(RouteBranch.CANDIDATE_EVAL, reason=f"candidate_eval:{m.pattern}")

    if m := _first_match(_FACT_QUERY, text):
        return RouteDecision(
            RouteBranch.FACT_QUERY,
            reason=f"fact_query:{m.pattern}",
            domain_hint=_guess_domain_hint(text),
        )

    if m := _first_match(_SINGLE_DOMAIN_TCM, text):
        return RouteDecision(
            RouteBranch.SINGLE_DOMAIN, reason=f"single_domain_tcm:{m.pattern}", domain_hint="tcm"
        )

    if m := _first_match(_SINGLE_DOMAIN_NUTRITION, text):
        return RouteDecision(
            RouteBranch.SINGLE_DOMAIN,
            reason=f"single_domain_nutrition:{m.pattern}",
            domain_hint="nutrition",
        )

    if m := _first_match(_FULL_RECOMMEND, text):
        return RouteDecision(RouteBranch.FULL_RECOMMEND, reason=f"full_recommend:{m.pattern}")

    if m := _first_match(_OTHER_GREETINGS, text):
        return RouteDecision(RouteBranch.OTHER, reason=f"other:{m.pattern}")

    return RouteDecision(
        RouteBranch.FULL_RECOMMEND, reason="unmatched", rule_matched=False
    )


# ---------------------------------------------------------------------------
# §5.1.1 / D32: 一句话包含多个意图时的多任务切分。确定性连接词切分,不调 LLM——
# 和 classify_route() 本身"确定性优先"的既有原则一致,也避免给每一条消息(哪怕
# 只有一个意图)都增加一次分类调用的延迟/成本。见 ARCHITECTURE.md §5.1.1、
# DECISIONS.md D32 的完整设计与取舍论证。
# ---------------------------------------------------------------------------

# 中英双语——这个项目也要给不看中文的用户用，六分支路由本身(_LOG_WRITE 等
# 列表)已经是双语的了，多任务切分不该只认中文连接词。
# 长的复合连接词排前面，且整个分组允许 `+` 连续重复——避免"顺便再问一下"这种
# 相邻连接词被拆成"顺便"+"再问一下"两个空洞的切分点(同 match_global_table()
# 长菜名优先扫的原则)。英文用 `\b` 词边界，避免匹配到别的单词里的子串
# (比如 "also" 不该命中 "salsa" 这类词的一部分)；中文不需要词边界，这个项目
# 别处的中文正则(比如 _LOG_WRITE)也都是直接子串匹配。
_INTENT_CONNECTORS_ZH = ("顺便问一下", "顺便问下", "再问一下", "再问下", "另外", "顺便", "还有")
_INTENT_CONNECTORS_EN = (
    "by the way", "one more question", "one more thing", "another thing",
    "and also", "oh and", "additionally", "also", "btw",
)
_INTENT_CONNECTOR_RE = re.compile(
    "(?:"
    + "|".join(sorted(_INTENT_CONNECTORS_ZH, key=len, reverse=True))
    + "|"
    + "|".join(
        rf"\b{re.escape(w)}\b" for w in sorted(_INTENT_CONNECTORS_EN, key=len, reverse=True)
    )
    + ")+",
    re.IGNORECASE,
)
_SEGMENT_STRIP_CHARS = " ，,。.!！?？、；;:："


def segment_intents(text: str) -> list[str]:
    """按意图连接词切分成候选片段。单意图消息(没有连接词，或连接词前面没有
    实质内容)原样返回一个元素的列表——调用方据此判断要不要走多任务路径，
    这个函数本身不做"是不是真的多任务"的判断(那需要结合 classify_route 的
    结果，见 D32 的激活条件)。"""
    parts = _INTENT_CONNECTOR_RE.split(text)
    if len(parts) == 1:
        stripped = text.strip()
        return [stripped] if stripped else []
    connectors = _INTENT_CONNECTOR_RE.findall(text)
    segments = [parts[0]]
    for i, connector in enumerate(connectors):
        following = parts[i + 1] if i + 1 < len(parts) else ""
        segments.append(connector + following)
    return [s.strip(_SEGMENT_STRIP_CHARS) for s in segments if s.strip(_SEGMENT_STRIP_CHARS)]


@dataclass(frozen=True)
class MultiTaskCandidate:
    text: str
    decision: RouteDecision


def _segmented_candidates(query: str) -> list[MultiTaskCandidate]:
    """按连接词切分后，给每个片段独立跑一次 `classify_route()`——原始结果，
    还没做"这算不算真的多任务"的判断，`classify_multi_task()`/`classify_turn()`
    在这份结果上各自加一层判断条件。"""
    return [MultiTaskCandidate(text=s, decision=classify_route(s)) for s in segment_intents(query)]


def classify_multi_task(query: str) -> list[MultiTaskCandidate] | None:
    """返回 None 表示按 §5.1 原有单分支路径处理(不满足 D32 的激活条件)；
    否则返回 ≥2 个 `MultiTaskCandidate`，按切分出现的先后顺序排列，供调用方
    (api/main.py)逐个分发。

    激活条件(D32,刻意保守)：切出 ≥2 个片段，且每个片段都必须
    `rule_matched=True`(哪怕只有一个片段落到"规则没命中"的模糊状态，也整体
    放弃切分)，且至少两个片段的 `branch` 不同(同分支的多个片段不需要这条
    新路径，各分支自己的处理逻辑本来就能处理一次输入里的多个同类项，比如
    "记录"分支的三级查找)。
    """
    candidates = _segmented_candidates(query)
    if len(candidates) < 2:
        return None
    if not all(c.decision.rule_matched for c in candidates):
        return None
    if len({c.decision.branch for c in candidates}) < 2:
        return None
    return candidates


# ---------------------------------------------------------------------------
# LLM fallback when no regex fired (not a substitute for the rule cascade)
# Prompt requires English-key JSON; parser only reads those English keys.
# Same contract as verification.py: Chinese rationale OK, machine keys English.
# ---------------------------------------------------------------------------

_VALID_BRANCHES = {b.value for b in RouteBranch}
_VALID_DOMAIN_HINTS = {"tcm", "nutrition"}

# Shared by classify_route_async and classify_turn so the two LLM fallbacks
# cannot drift. other is a last resort, not "anything that mentions weather".
_BRANCH_GUIDE = (
    "- log_write: user wants to save what they ate into the diet log\n"
    "- log_review: user asks what THEY ate in the past (their own log), not a knowledge-base fact\n"
    "- fact_query: a single factual question about one food's properties — TCM nature/"
    "flavor/meridian, nutrients, allergens, 'is X cold/hot'. Not a meal plan\n"
    "- candidate_eval: user names specific dish(es)/options and wants a judgement "
    "(can I eat this / which of these). Not generating a plan from scratch\n"
    "- single_domain: clearly ONLY TCM (constitution, what a qi-deficiency person "
    "should eat, seasonal diet from a TCM view with no nutrition angle) OR ONLY "
    "nutrition (iron, calcium, warfarin+vitamin K). Not a full day's meals\n"
    "- full_recommend: open-ended what-to-eat that needs both TCM and nutrition — "
    "today/tonight's meals, a day's plan, eating while working late, or weather/"
    "climate/season as a constraint on what to eat\n"
    "- other: ONLY when none of the six apply. Pure greetings/thanks/farewell; a "
    "question with no diet/food/TCM/nutrition intent at all (pure 'how's the "
    "weather', math, news); or a food-adjacent how-to we cannot retrieve (cooking "
    "steps, knife skills). Weather/season as a constraint on what to eat is "
    "full_recommend, not other. When unsure between other and a diet branch, pick "
    "the diet branch\n"
)

_ROUTE_LLM_SYSTEM = (
    "你是 diet_expert 的路由分类器。根据用户原话，把它分到下面七条分支之一。"
    "规则说明可用中文理解；机器可读输出必须是英文键名的 JSON，不要 Markdown，"
    "不要中文小节标题。"
    "\n\n"
    "Branches:\n"
    f"{_BRANCH_GUIDE}"
    "\n"
    "Output format (machine-parsed — English keys only). Return one JSON object:\n"
    '{"branch":"<one of the seven>","domain_hint":"tcm"|"nutrition"|null}\n'
    "domain_hint is required for fact_query and single_domain; otherwise null.\n"
    "Output ONLY that JSON object — no text, no explanation, before or after it.\n"
    "One call only. Do not plan multiple steps. Do not call tools."
)


def _strip_json_fences(text: str) -> str:
    """Isolate the JSON object from an LLM response.

    Two failure modes seen with real models, handled independently:
    1. The whole reply wrapped in a ```json ... ``` fence — strip it.
    2. A well-formed object followed by trailing prose despite the prompt
       saying "return one JSON object" / "one call only" — narrow to the
       first balanced {...} span so trailing text doesn't break json.loads.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def _parse_route_llm_json(text: str) -> tuple[RouteBranch, str | None] | None:
    """Extract English keys `branch` / `domain_hint` only. Chinese keys are ignored."""
    try:
        data = json.loads(_strip_json_fences(text or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw_branch = data.get("branch")
    if not isinstance(raw_branch, str) or raw_branch not in _VALID_BRANCHES:
        return None
    raw_hint = data.get("domain_hint")
    hint: str | None
    if raw_hint in _VALID_DOMAIN_HINTS:
        hint = raw_hint
    else:
        hint = None
    branch = RouteBranch(raw_branch)
    if branch not in (RouteBranch.FACT_QUERY, RouteBranch.SINGLE_DOMAIN):
        hint = None
    return branch, hint


async def _llm_classify_route(
    query: str, complete: CompleteFn | None
) -> RouteDecision | None:
    complete = complete or llm_adapter.complete
    try:
        result = await complete(
            [
                {"role": "system", "content": _ROUTE_LLM_SYSTEM},
                {"role": "user", "content": query},
            ]
        )
    except Exception:
        logger.warning("route LLM fallback failed", exc_info=True)
        return None
    parsed = _parse_route_llm_json(result.text or "")
    if parsed is None:
        logger.warning("route LLM fallback returned unparseable output: %r", result.text)
        return None
    branch, hint = parsed
    if branch in (RouteBranch.FACT_QUERY, RouteBranch.SINGLE_DOMAIN) and hint is None:
        hint = _guess_domain_hint(query)
    return RouteDecision(
        branch,
        reason=f"llm_fallback:{branch.value}",
        domain_hint=hint,
        rule_matched=False,
    )


async def _classify_route_async_inner(
    query: str, *, complete: CompleteFn | None = None
) -> RouteDecision:
    """Rule cascade first; LLM only when no regex matched."""
    decision = classify_route(query)
    if decision.rule_matched or decision.reason == "empty_query_default":
        return decision
    llm_decision = await _llm_classify_route(query, complete)
    if llm_decision is not None:
        return llm_decision
    return RouteDecision(
        RouteBranch.FULL_RECOMMEND,
        reason="llm_fallback_failed_default_full_recommend",
        rule_matched=False,
    )


async def classify_route_async(
    query: str, *, complete: CompleteFn | None = None
) -> RouteDecision:
    """Rule cascade first; LLM only when no regex matched.

    Empty input skips the LLM (nothing to classify) and stays full_recommend.
    LLM failure / invalid JSON / unknown branch → full_recommend, same as the
    old unmatched default, so the pipeline still has a branch to run.
    """
    t0 = time.perf_counter()
    with observation(
        "router",
        as_type="chain",
        input={"query": redact_text(query)},
    ):
        decision = await _classify_route_async_inner(query, complete=complete)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        update_current(
            output={
                "branch": decision.branch.value,
                "reason": decision.reason,
                "domain_hint": decision.domain_hint,
                "rule_matched": decision.rule_matched,
            },
            metadata={"latency_ms": round(latency_ms, 1)},
        )
        stage_log(
            logger,
            "router",
            latency_ms=round(latency_ms, 1),
            branch=decision.branch.value,
            reason=decision.reason,
            domain_hint=decision.domain_hint,
            rule_matched=decision.rule_matched,
        )
        return decision


# ---------------------------------------------------------------------------
# D32 补充(2026-08-27)：多任务的 LLM 兜底。`classify_multi_task()` 只处理
# "有显式连接词、且每个片段都规则命中"这一类；真实使用后发现这漏了两类情况：
# ①连接词存在但某个片段规则没命中(比如后半句用了正则没覆盖的说法)；
# ②压根没有连接词的隐式多意图(比如"麻婆豆腐好吃吗我中午吃了")。这两类都交给
# 下面这次 LLM 调用去判断，不再无条件退回单分支路径。见 DECISIONS.md D32
# 的补充说明——这是对"明确不做LLM语义切分"这条原有取舍的修订，不是从头设计。
# ---------------------------------------------------------------------------

_TURN_LLM_SYSTEM = (
    "你是 diet_expert 的路由分类器。判断用户这句话包含几个相互独立的请求,"
    "把每一个请求分别分到下面七条分支之一。**大多数消息只包含一个请求**——"
    "拿不准的时候优先当成一个请求处理,只有确实包含多个明显独立、分支也不同"
    "的请求时才拆开。"
    "规则说明可用中文理解；机器可读输出必须是英文键名的 JSON，不要 Markdown，"
    "不要中文小节标题。"
    "\n\n"
    "Branches:\n"
    f"{_BRANCH_GUIDE}"
    "\n"
    "Output format (machine-parsed — English keys only). Return one JSON object:\n"
    '{"tasks":[{"text":"<the original-language fragment for this request>",'
    '"branch":"<one of the seven>","domain_hint":"tcm"|"nutrition"|null}]}\n'
    "domain_hint is required only for fact_query/single_domain items; otherwise null.\n"
    "If there is only one request, return a list with exactly one item whose \"text\" "
    "is the whole original message.\n"
    "Output ONLY that JSON object — no text, no explanation, before or after it.\n"
    "One call only. Do not plan multiple steps. Do not call tools."
)


def _parse_turn_llm_json(text: str) -> list[tuple[str | None, RouteBranch, str | None]] | None:
    try:
        data = json.loads(_strip_json_fences(text or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return None
    parsed: list[tuple[str | None, RouteBranch, str | None]] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            return None
        raw_branch = item.get("branch")
        if not isinstance(raw_branch, str) or raw_branch not in _VALID_BRANCHES:
            return None
        raw_text = item.get("text")
        text_value = raw_text if isinstance(raw_text, str) and raw_text.strip() else None
        raw_hint = item.get("domain_hint")
        hint = raw_hint if raw_hint in _VALID_DOMAIN_HINTS else None
        branch = RouteBranch(raw_branch)
        if branch not in (RouteBranch.FACT_QUERY, RouteBranch.SINGLE_DOMAIN):
            hint = None
        parsed.append((text_value, branch, hint))
    return parsed


async def _llm_classify_turn(
    query: str, complete: CompleteFn | None
) -> list["MultiTaskCandidate"] | None:
    complete = complete or llm_adapter.complete
    try:
        result = await complete(
            [
                {"role": "system", "content": _TURN_LLM_SYSTEM},
                {"role": "user", "content": query},
            ]
        )
    except Exception:
        logger.warning("turn LLM classify failed", exc_info=True)
        return None
    parsed = _parse_turn_llm_json(result.text or "")
    if parsed is None:
        logger.warning("turn LLM classify returned unparseable output: %r", result.text)
        return None
    tasks: list[MultiTaskCandidate] = []
    for text_value, branch, hint in parsed:
        segment_text = text_value or query
        if branch in (RouteBranch.FACT_QUERY, RouteBranch.SINGLE_DOMAIN) and hint is None:
            hint = _guess_domain_hint(segment_text)
        tasks.append(
            MultiTaskCandidate(
                text=segment_text,
                decision=RouteDecision(
                    branch, reason=f"llm_turn:{branch.value}", domain_hint=hint, rule_matched=False
                ),
            )
        )
    return tasks


async def _classify_turn_inner(
    query: str, *, complete: CompleteFn | None = None
) -> tuple[MultiTaskCandidate, ...]:
    candidates = _segmented_candidates(query)
    if len(candidates) >= 2:
        all_matched = all(c.decision.rule_matched for c in candidates)
        if all_matched and len({c.decision.branch for c in candidates}) >= 2:
            return tuple(candidates)  # 确定性多任务，见 classify_multi_task()

    single = classify_route(query)

    if len(candidates) >= 2:
        # 有连接词、但没被判定为确定性多任务——原因要么是"至少一个片段规则
        # 没命中"(真的拿不准，值得核实)，要么是"所有片段其实是同一个分支"
        # (比如两句话都是"记录"，不需要核实，各分支自己的处理逻辑——比如
        # "记录"分支一次就能识别多个菜品——本来就能处理好，白打一次LLM没必要)。
        needs_llm_check = not all_matched
    else:
        # 完全没有连接词信号：和 classify_route_async 现有的单分支兜底触发
        # 条件一样，只有整句规则也没命中时才值得打一次 LLM——但这次判断的是
        # "这句话该拆成几个任务"，不只是"该分到哪一个分支"，覆盖的是压根
        # 没用连接词的隐式多意图(比如"麻婆豆腐好吃吗我中午吃了")。
        needs_llm_check = not single.rule_matched

    if not needs_llm_check:
        return (MultiTaskCandidate(text=query, decision=single),)

    llm_tasks = await _llm_classify_turn(query, complete)
    if llm_tasks is not None:
        return tuple(llm_tasks)

    if single.rule_matched:
        return (MultiTaskCandidate(text=query, decision=single),)
    return (
        MultiTaskCandidate(
            text=query,
            decision=RouteDecision(
                RouteBranch.FULL_RECOMMEND,
                reason="llm_fallback_failed_default_full_recommend",
                rule_matched=False,
            ),
        ),
    )


async def classify_turn(
    query: str, *, complete: CompleteFn | None = None
) -> tuple[MultiTaskCandidate, ...]:
    """一次性决定这一轮消息该拆成几个子任务、每个子任务走哪条分支——取代
    `api/main.py` 里"先调 `classify_multi_task()` 再调 `classify_route_async()`"
    两段独立调用，保证最坏情况下这条消息也只产生一次 LLM 调用(不会为同一条
    消息分别打一次"多任务检测"和一次"单分支兜底")。返回值恒为非空元组，
    长度为 1 时就是原来的单分支场景。

    优先级(确定性优先，一贯原则)：
    1. `classify_multi_task()`：连接词切分 + 每段都规则命中 + 分支不同 →
       零 LLM 调用。
    2. 整句 `classify_route()` 规则命中、且没有"连接词存在但没法确认"这个
       信号 → 当成单任务返回，零 LLM 调用。
    3. 都不满足(说明规则要么完全没命中，要么命中了但有理由怀疑漏了第二个
       意图)→ 一次 LLM 调用，模型自己判断这句话该拆成几个任务；调用失败/
       返回格式不对 → 退回步骤 2 的结果(如果规则命中过)，否则退回
       full_recommend 单任务，和 `classify_route_async` 现有的兜底一致。

    `classify_route_async()` 保留不变、继续单独存在——它是"只关心单分支"
    场景的既有契约，不因为这次新增多任务感知而改变行为。
    """
    t0 = time.perf_counter()
    # Span 名字沿用既有的 "router"(不叫 "router.turn")、`output` 里保留
    # branch/reason/domain_hint/rule_matched 这几个旧键——这个函数是
    # `classify_route_async()` 在 API 层的替代品，观测契约不应该因为新增了
    # 多任务感知就跟着变形状，多任务场景下这几个键退化成 "multi_task"/None，
    # 新增的 task_count/branches 两个键专门给多任务场景用。
    with observation("router", as_type="chain", input={"query": redact_text(query)}):
        tasks = await _classify_turn_inner(query, complete=complete)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        single = tasks[0].decision if len(tasks) == 1 else None
        branches = [t.decision.branch.value for t in tasks]
        update_current(
            output={
                "branch": single.branch.value if single else "multi_task",
                "reason": single.reason if single else None,
                "domain_hint": single.domain_hint if single else None,
                "rule_matched": single.rule_matched if single else None,
                "task_count": len(tasks),
                "branches": branches,
            },
            metadata={"latency_ms": round(latency_ms, 1)},
        )
        stage_log(
            logger,
            "router",
            latency_ms=round(latency_ms, 1),
            branch=single.branch.value if single else "multi_task",
            reason=single.reason if single else None,
            domain_hint=single.domain_hint if single else None,
            rule_matched=single.rule_matched if single else None,
            task_count=len(tasks),
            branches=branches,
        )
        return tasks
