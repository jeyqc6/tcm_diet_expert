#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出拦截：诊断性表述 / 过敏原硬阻断 / "多多益善"式表述。

设计依据：docs/PRD.md §10(Guardrails 表"输出拦截"四行 + §10.1 核查 pass 第 4 项
"过敏原交叉检查(含隐藏成分)")；docs/ARCHITECTURE.md §5.4(挂载点:核查 pass 步骤6)
决策依据：docs/ENGINEERING.md §7.3(过敏原命中是确定性优先原则点名的三类之一,
要求 100% 单测覆盖)
roadmap:阶段 5,过敏原部分⚠️ 要求 100% 覆盖

本文件不包含"无 source_id 的推荐条目 → 移除"这一项——那已经是
`backend/agents/verification.py` 的 `apply_deterministic_source_check` 在做的事
(阶段 4 任务 9 已完成)，这里不重新实现一遍，避免两处各管一半又互相不知道对方
存在。

过敏原比对逻辑与 ARCHITECTURE §4.2"菜品拆解"步骤 2(拆解出的 ingredients 与
user_profile.allergens 做集合比对)是**同一件事**，本应共用同一份代码——但
§4.2 那条写入链路依赖的 `backend/memory/dish_decomposition.py` 是阶段 7 的任务，
目前还不存在。`check_allergens()` 因此设计成不依赖任何 §4.2 才有的东西，
只吃"一段文本 + 用户过敏原列表"，两处未来接上时，dish_decomposition.py 拆出
ingredients 列表后传进来即可复用，不需要改这个函数的签名。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# 过敏原硬阻断
# ---------------------------------------------------------------------------

# 隐藏过敏原来源：调味品/加工品的名字本身不含过敏原类别的字面词，但含有该类别
# 成分——verification_checklist.md/recipe_and_shopping_list.md 两份 Skill 里
# 反复用"蚝油→甲壳类"这个例子，这里是它第一次真正被代码检查。⚠️ 这是一份种子
# 列表，不是穷举中式调味品的权威过敏原数据库(那需要真实食品成分数据支撑，不是
# 靠代码作者拍脑袋能扩充完的)——新增映射前确认来源，参照
# `knowledge/food/dish-decomposition.jsonl` README"用真实数据，不抄网站"的同一
# 条纪律。
HIDDEN_ALLERGEN_SOURCES: dict[str, str] = {
    "蚝油": "甲壳类",
    "虾皮": "甲壳类",
    "虾米": "甲壳类",
    "虾酱": "甲壳类",
    "虾油": "甲壳类",
    "鱼露": "鱼类",
    "黄油": "乳制品",
    "奶油": "乳制品",
    "芝士": "乳制品",
    "奶酪": "乳制品",
    "乳清蛋白": "乳制品",
    "面筋": "麸质",
    "面筋粉": "麸质",
    "麻酱": "芝麻",
    "芝麻油": "芝麻",
    "香油": "芝麻",
    "花生油": "花生",
    "花生酱": "花生",
    "酱油": "麸质",  # 传统酿造酱油多以小麦为原料之一，保守起见标注
    # ⚠️ 2026-09-01 补：英文条目。`user_profile.allergens` 存的永远是中文
    # 类别名(`critical_fact_scanner.py` 扫描时就归一化成"甲壳类"这类canonical
    # 值，不管用户当初用中文还是英文说的)——所以当回复语言是英文时(i18n.py
    # locale=en)，`check_allergens()` 的"类别本身字面出现在文本里"这条直接
    # 命中路径**永远不会命中**，因为英文文本里不会出现"甲壳类"这几个汉字。
    # 这份表的机制(词→类别的子串映射)对英文一样适用，所以英文类别本身的
    # 直接对应词(shrimp/shellfish 这类)也放进来，不只是"隐藏来源"这种间接
    # 情形——这是让英文回复也能被过敏原硬阻断保护住的唯一路径，不是可选项。
    "shellfish": "甲壳类",
    "crustacean": "甲壳类",
    "crustaceans": "甲壳类",
    "shrimp": "甲壳类",
    "shrimps": "甲壳类",
    "prawn": "甲壳类",
    "prawns": "甲壳类",
    "crab": "甲壳类",
    "lobster": "甲壳类",
    "oyster sauce": "甲壳类",
    "shrimp paste": "甲壳类",
    "fish": "鱼类",
    "fish sauce": "鱼类",
    "anchovy": "鱼类",
    "anchovies": "鱼类",
    "dairy": "乳制品",
    "milk": "乳制品",
    "butter": "乳制品",
    "cream": "乳制品",
    "cheese": "乳制品",
    "whey": "乳制品",
    "whey protein": "乳制品",
    "lactose": "乳制品",
    "gluten": "麸质",
    "wheat": "麸质",
    "wheat gluten": "麸质",
    "soy sauce": "麸质",  # 同上面中文"酱油"一条：传统酿造多以小麦为原料，保守标注
    "sesame": "芝麻",
    "sesame oil": "芝麻",
    "tahini": "芝麻",
    "peanut": "花生",
    "peanuts": "花生",
    "peanut oil": "花生",
    "peanut butter": "花生",
    "tree nut": "坚果",
    "tree nuts": "坚果",
    "almond": "坚果",
    "cashew": "坚果",
    "walnut": "坚果",
    "pistachio": "坚果",
    "hazelnut": "坚果",
    "soy": "大豆",
    "soybean": "大豆",
    "soybeans": "大豆",
    "tofu": "大豆",
    "egg": "蛋类",
    "eggs": "蛋类",
}


def hidden_sources_for_allergens(user_allergens: Iterable[str] | None) -> dict[str, list[str]]:
    """按用户过敏原类别，反查 `HIDDEN_ALLERGEN_SOURCES` 里对应的隐藏来源词。

    用途和 `check_allergens()` 相反的方向：`check_allergens()` 是"生成完之后
    扫文本里有没有踩雷"，这个函数是"生成之前提醒模型雷在哪"——两者共用同一份
    种子表，不是两份独立维护的数据(`backend/agents/_subagent_common.py` 的
    `build_allergen_avoidance_instruction()` 用这个函数拼生成阶段的提示词)。
    """
    allergens = {a.strip() for a in (user_allergens or []) if a and a.strip()}
    if not allergens:
        return {}
    result: dict[str, list[str]] = {}
    for term, category in HIDDEN_ALLERGEN_SOURCES.items():
        if category in allergens:
            result.setdefault(category, []).append(term)
    return result


@dataclass(frozen=True)
class AllergenFinding:
    matched_term: str  # 文本里实际出现的词（可能是隐藏来源词，也可能是类别本身）
    allergen: str  # 命中的用户过敏原类别


# 否定词：紧挨在命中词前面出现时，判定这次出现是"明确说明不含"，不算命中——
# 典型例子"蛋白质来源已换成不含甲壳类的选项"，"甲壳类"作为字面子串确实出现了，
# 但整句话说的正是我们希望模型说的"已经避开"，不能把模型主动声明的规避语句
# 也当成违规。⚠️ 这不是完整的否定辖域分析(真正的句法否定需要解析器)，只是
# "命中位置前 `_NEGATION_WINDOW` 个字符内有没有出现否定词"这样一个足够便宜、
# 足够管用的启发式——见模块文档"残留"部分。宁可因为窗口不够精确而漏掉一次
# 真该拦的("已经避开甲壳类了，但是我还是加了蚝油"这种混合表述里，"蚝油"
# 前面没有否定词，仍然会被正常命中)，也不做更复杂的规则，复杂化带来的维护
# 成本目前不值得。
#（不用单字"无"——"无花果"这类词本身带"无"字，会在窗口里造成误判，让真实
# 命中被错误地当成"否定过"。宁可漏挡一次"无XX"这种简短否定表述，也不要为了
# 接住它而制造出更危险的漏判。）
NEGATION_MARKERS = (
    "不含", "不包含", "没有", "不加", "不放", "未使用", "不用", "去除", "避开",
    "不使用", "排除", "不出现", "不要出现", "勿食", "勿用", "忌食",
    # 2026-09-01 补英文：全部选"短、卡在 _NEGATION_WINDOW 字符数以内"的词，
    # 不是完整语法结构(比如用 "not contain" 不用 "does not contain")——原因
    # 和中文列表完全一样(见上面注释)：这是"命中位置前 N 个字符内有没有出现
    # 这个子串"的窗口式启发式，不是语法分析。选短词的好处是"does not
    # contain"/"doesn't contain"/"do not contain"这些说法的公共后缀
    # "not contain"天然都能接住，不需要为每种语法变体单独列一条；同时故意
    # **不**把窗口(`_NEGATION_WINDOW`)从 12 放宽去将就更长的英文短语——放宽
    # 窗口会让上面"避开"这个词的辖域也跟着变宽，反而可能把"已经避开甲壳类了，
    # 但是我还是加了蚝油"这种句子里后面真实的命中也一起放过(见对应单测)。
    # 宁可漏掉几个写法更绕的英文否定句(结果是多一条不必要的安全提示，无害)，
    # 也不放宽窗口去冒"真命中被误判成已否定"的风险。
    "not contain", "not include", "free of", "without", "excluding",
    "excludes", "avoided", "removed", "not used", "omitted",
)
_NEGATION_WINDOW = 12

# "今天避开：花生、虾" 这类清单里，过敏原名字离标题可能隔了换行和顿号，
# 否定窗口接不住。标题必须比单字"避开"更具体，否则
# "已经避开甲壳类了，但是我还是加了蚝油"会把后面的真命中也放掉。
# 辖域只覆盖标题后一小段（空行 / 新小节 / 字数上限），不能一路吃到文末，
# 否则标题下面的真实推荐（「午餐：宫保鸡丁配花生碎」）也会被放过。
AVOIDANCE_SECTION_HEADERS = (
    "今天避开",
    "今日避开",
    "需要避开",
    "应当避开",
    "建议避开",
    "请避开",
    "全程不出现",
    "不要出现",
    "不宜食用",
    "忌口",
    # 2026-09-01 补英文小节标题——机制和中文一样，标题后一小段范围内的列举
    # 不算命中。
    "avoid today",
    "to avoid",
    "please avoid",
    "avoid the following",
    "do not include",
    "not suitable for you",
)
_AVOIDANCE_SECTION_MAX_CHARS = 120
_AVOIDANCE_SECTION_END_RE = re.compile(
    r"("
    r"\n#{1,6}\s"
    r"|"
    r"\n\s*\n"
    r"|"
    r"\n(?=(?:午餐|晚餐|早餐|加餐|推荐|做法|步骤|购物清单|"
    r"lunch|dinner|breakfast|snack|recommend|instructions?|steps?|shopping\s*list))"
    r")",
    re.IGNORECASE,
)


def _in_avoidance_section(text: str, pos: int) -> bool:
    """Hit sits in a short avoidance list after a section header, not a recommendation."""
    before = text[:pos].lower()  # 大小写不敏感，理由同 _term_present_unnegated
    header_at = -1
    for header in AVOIDANCE_SECTION_HEADERS:
        idx = before.rfind(header.lower())
        if idx > header_at:
            header_at = idx
    if header_at < 0:
        return False
    span = text[header_at:pos]
    if len(span) > _AVOIDANCE_SECTION_MAX_CHARS:
        return False
    return _AVOIDANCE_SECTION_END_RE.search(span) is None


def _term_present_unnegated(text: str, term: str) -> bool:
    """`term` 在 `text` 里出现过、且至少有一次出现不是紧跟在否定词之后、
    也不在避开清单小节里。

    大小写不敏感(`re.IGNORECASE` + 否定词窗口比对时统一转小写)——中文没有
    大小写，这个开关对中文条目是空操作，但英文回复里句首/标题惯例会把词
    首字母大写(比如"Shellfish is not used here")，term/否定词都按原样小写
    存在表里，不加这个开关的话英文这条命中路径基本等于没做。"""
    for m in re.finditer(re.escape(term), text, re.IGNORECASE):
        if _in_avoidance_section(text, m.start()):
            continue
        window = text[max(0, m.start() - _NEGATION_WINDOW) : m.start()].lower()
        if not any(marker.lower() in window for marker in NEGATION_MARKERS):
            return True
    return False


def check_allergens(text: str, user_allergens: Iterable[str] | None) -> list[AllergenFinding]:
    """扫描一段文本(推荐条目/菜谱/购物清单等)，返回命中的用户过敏原。

    两条命中路径，任一都算命中：
      1. 文本里直接出现过敏原类别本身的字面词(用户过敏原是"花生"，文本里出现"花生")
      2. 文本里出现某个隐藏来源词，该来源映射到用户过敏原类别(用户过敏原是"甲壳类"，
         文本里出现"蚝油")

    出现位置前紧跟否定词(见 `NEGATION_MARKERS`)的情形不算命中——"不含甲壳类"
    不应该被当成"含甲壳类"。落在 `AVOIDANCE_SECTION_HEADERS` 小节里的列举
    （「今天避开：花生、虾」）同样不算命中。

    不做形态学/同义词归一化(比如"虾"和"大虾")——这里只匹配子串，`user_allergens`
    里的每一项和 `HIDDEN_ALLERGEN_SOURCES` 的 key/value 都应该是这类比对能处理的
    规范化短词，归一化本身是 user_profile 录入时(PRD §10.2 人在环)该做的事，
    不是这个函数的职责。
    """
    allergens = {a.strip() for a in (user_allergens or []) if a and a.strip()}
    if not allergens or not text:
        return []

    findings: list[AllergenFinding] = []
    seen: set[tuple[str, str]] = set()

    for allergen in allergens:
        if _term_present_unnegated(text, allergen):
            key = (allergen, allergen)
            if key not in seen:
                findings.append(AllergenFinding(matched_term=allergen, allergen=allergen))
                seen.add(key)

    for term, category in HIDDEN_ALLERGEN_SOURCES.items():
        if category in allergens and _term_present_unnegated(text, term):
            key = (term, category)
            if key not in seen:
                findings.append(AllergenFinding(matched_term=term, allergen=category))
                seen.add(key)

    return findings


# ---------------------------------------------------------------------------
# 诊断性表述——PRD §10"诊断性表述 → 正则 + 分类器双层拦截，替换为免责模板"。
# 这里是正则那一层；"分类器"(LLM 软判定)不在本文件职责内。
# ---------------------------------------------------------------------------
_DIAGNOSTIC_PATTERN = re.compile(
    r"("
    r"你(是|患有|得了|确诊|可能是|可能患有)[^。，,！!？?]{0,12}(病|症|综合征|障碍|炎)"
    r"|"
    r"你的?病(是|就是)"
    r"|"
    r"(确诊|诊断)为"
    r"|"
    # 2026-09-01 补英文——同样覆盖"你是/患有/确诊为 XX 病"这几种诊断性
    # 断言的英文说法。中文能用"这个词后面跟病/症/炎/综合征/障碍就是病名"
    # 这条规律，英文病名词形五花八门(diabetes/cancer/IBS 都不共享任何后缀)，
    # 没有等价的干净后缀规律——分两支处理：①词本身以常见医学后缀结尾
    # (itis/osis/emia/algia，或独立单词 disease/syndrome/disorder/
    # condition/infection)；②明确列出的常见病名种子表(参照本文件其余
    # 种子表"不追求穷举，按需扩充"的一贯做法)。
    r"\byou\s+(?:have|might\s+have|may\s+have|likely\s+have)\s+"
    r"(?:a\s+|an\s+)?[a-zA-Z\s\-]{0,20}(?:itis|osis|emia|algia|disease|syndrome|disorder|condition|infection)\b"
    r"|"
    r"\byou\s+(?:have|might\s+have|may\s+have|likely\s+have)\s+"
    r"(?:a\s+|an\s+)?(?:diabetes|cancer|hypertension|depression|anxiety|asthma|eczema|"
    r"psoriasis|anemia|copd|ibs|gout|migraines?)\b"
    r"|"
    r"\byour\s+diagnosis\s+is\b"
    r"|"
    r"\bdiagnosed\s+with\s+[a-zA-Z\s\-]{0,30}\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DiagnosticFinding:
    matched_text: str


def check_diagnostic_statement(text: str) -> DiagnosticFinding | None:
    m = _DIAGNOSTIC_PATTERN.search(text)
    if not m:
        return None
    return DiagnosticFinding(matched_text=m.group(0))


# ---------------------------------------------------------------------------
# "多多益善"式表述——PRD §10"重生成，注入平衡约束"。检测层同样只负责标记，
# 不在这里做"重生成"(那是上游 retry 编排的事，本文件不做业务流程编排)。
# ---------------------------------------------------------------------------
_UNLIMITED_GOOD_PATTERN = re.compile(
    r"(多吃[^。，,！!？?]{0,10}(有益|越多越好|多多益善)|越多越好|多多益善|吃得越多越好"
    r"|"
    # 2026-09-01 补英文："the more the better"/"as much as you like/want"/
    # "no limit on how much"/"unlimited amount(s)"。
    r"\bthe\s+more\s+the\s+better\b"
    r"|"
    r"\beat\s+as\s+much\s+as\s+you\s+(?:like|want|can)\b"
    r"|"
    r"\b(?:no\s+limit|unlimited\s+amounts?)\s+(?:on|of|to)\s+how\s+much\b"
    r"|"
    r"\bunlimited\s+amounts?\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UnlimitedGoodFinding:
    matched_text: str


def check_unlimited_good_statement(text: str) -> UnlimitedGoodFinding | None:
    m = _UNLIMITED_GOOD_PATTERN.search(text)
    if not m:
        return None
    return UnlimitedGoodFinding(matched_text=m.group(0))


@dataclass
class OutputFilterResult:
    allergens: list[AllergenFinding] = field(default_factory=list)
    diagnostic: DiagnosticFinding | None = None
    unlimited_good: UnlimitedGoodFinding | None = None

    @property
    def blocked(self) -> bool:
        """过敏原命中 = 硬阻断(PRD §10)；诊断性表述同样是硬性拦截(替换为免责
        模板，不是"降级标注")。"多多益善"式表述的动作是重生成而不是移除，
        不计入 `blocked`——调用方需要单独检查 `unlimited_good` 字段决定是否
        触发重生成。"""
        return bool(self.allergens) or self.diagnostic is not None


def filter_output(text: str, user_allergens: Iterable[str] | None = ()) -> OutputFilterResult:
    """聚合三项输出侧检查，供 `backend/agents/verification.py` 的确定性检查
    调用。"""
    return OutputFilterResult(
        allergens=check_allergens(text, user_allergens),
        diagnostic=check_diagnostic_statement(text),
        unlimited_good=check_unlimited_good_statement(text),
    )
