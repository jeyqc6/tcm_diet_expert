#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键事实落库前置——跨路由分支的常驻检查（§4.3 总览图②）。

设计依据：docs/ARCHITECTURE.md §4.3
决策依据：docs/PRD.md §12.4("关键事实须在压缩发生前落库"，不局限于"记录"分支)
roadmap：阶段 7

用户可能在**任何**分支的对话里顺带提一句关键事实——比如在问"今天该吃什么"
(完整推荐分支)时说"对了我对虾过敏"。如果只在"记录"分支做落库检查，这句话
会被完整推荐分支当成普通输入，最终随会话历史一起被压缩掉(重现 PRD §7 那个
经典 bug 场景)。这个模块提供一次**确定性**关键词/规则扫描(过敏原声明句式 +
"我在吃/服用"补剂提及模式)，不调用 LLM——`api/main.py` 在路由判断之前对每一轮
原始输入调用一次，命中就同步调 `write_memory(critical)`(走 MCP client，不是
本模块直接连库，见 `api/main.py` `_stream_chat_inner` 里的调用点)。

本模块只做**纯扫描**，不碰数据库/MCP——`scan_critical_facts()` 的输入是文本 +
(可选的)当前 `UserProfileContext`，输出是"相对现有画像的增量"，方便单测
100% 覆盖，不需要 mock 任何 I/O。落库、SSE 通知、把新事实并入本轮请求剩余
流程用的 `profile` 对象，都是调用方(`api/main.py`)的职责。

**扫描到的是"类别"而不是原文用词**(比如用户说"虾"，落库写的是"甲壳类")——
这样才能真正触发 `backend/guardrails/output_filters.py` `check_allergens()`
的隐藏来源比对层(蚝油→甲壳类这类)，而不是只有原文逐字重复"虾"这个词才算
命中。那份文件的模块文档早就点名"归一化本身是 user_profile 录入时(人在环)该
做的事，不是 `check_allergens()` 的职责"——这里就是那个"该做归一化的地方"。
类别命名对齐 `output_filters.py` 的 `HIDDEN_ALLERGEN_SOURCES`(甲壳类/鱼类/
乳制品/麸质/芝麻/花生)，两者类别名不一致的话，这里新写进 `user_profile.allergens`
的类别不会被那边的隐藏来源比对认出来，防护会变成"看起来生效但实际不生效"。
另外三类(坚果/大豆/蛋类)补全 `knowledge/_raw/allergen/condiment-allergens-cn.jsonl`
里出现过的"中美并集九类"——`output_filters.py` 目前还没为这三类维护隐藏来源
表，但写进 `allergens` 之后，`check_allergens()` 的直接字面命中层(类别词本身
出现在文本里)已经能生效，不是白写。

⚠️ 这是一份种子词表，不是穷举权威过敏原/补剂数据库——参照 `output_filters.py`
同一条纪律：新增映射前确认来源，不是拍脑袋扩充。
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any

from backend.agents.user_context import UserProfileContext

# ---------------------------------------------------------------------------
# 过敏原声明扫描
# ---------------------------------------------------------------------------

# 类别命名对齐 backend/guardrails/output_filters.py 的 HIDDEN_ALLERGEN_SOURCES
# (甲壳类/鱼类/乳制品/麸质/芝麻/花生)，新增三类(坚果/大豆/蛋类)见模块文档。
#
# 2026-09-01 每个类别补了英文关键词——用户用英文声明过敏("I'm allergic to
# shrimp")时，扫描到的类别写进 user_profile.allergens 的仍然是这份表的中文
# key(比如"甲壳类"，归一化逻辑不变，见模块文档"扫描到的是类别而不是原文
# 用词"一节)，只是识别用户"提到了哪个类别"这一步现在中英文关键词都认。
ALLERGEN_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "甲壳类": (
        "甲壳类", "贝类", "龙虾", "螃蟹", "虾", "蟹",
        "shellfish", "crustacean", "shrimp", "prawn", "crab", "lobster",
    ),
    "鱼类": ("鱼类", "三文鱼", "鳕鱼", "带鱼", "鱼", "fish", "salmon", "cod"),
    "乳制品": (
        "乳制品", "奶制品", "牛奶", "奶酪", "芝士", "乳糖",
        "dairy", "milk", "cheese", "lactose",
    ),
    "麸质": ("麸质", "谷蛋白", "小麦", "面筋", "gluten", "wheat"),
    "芝麻": ("芝麻", "sesame"),
    "花生": ("花生", "peanut", "peanuts"),
    "坚果": (
        "坚果", "杏仁", "腰果", "核桃", "开心果", "榛子",
        "tree nut", "nuts", "almond", "cashew", "walnut", "pistachio", "hazelnut",
    ),
    "大豆": ("大豆", "黄豆", "豆制品", "soy", "soybean", "soybeans"),
    "蛋类": ("蛋类", "鸡蛋", "蛋清", "蛋黄", "egg", "eggs"),
}

# 否定语境：紧挨在命中位置前面出现时，判定这次出现是"明确说明不过敏"，不算
# 命中——"我不是对虾过敏"、"并非对芝麻过敏"。同 output_filters.py 的窗口式
# 启发式，但这里的否定词表是"否定过敏声明"专用，和"否定菜里含某成分"是两类
# 不同的否定语境，不共用同一份词表。
#
# 2026-09-01 补英文否定词——刻意选最短的("not"/"n't")：`_ALLERGY_WINDOW` 只有
# 6 个字符，比 output_filters.py 的否定窗口(12)还窄，装不下"does not"/
# "isn't allergic"这种完整短语，只能靠这两个词的公共子串生效("not"接住
# "am not"/"is not"，"n't"接住"don't"/"isn't"/"aren't"这类缩写)——和中文
# 列表选"没有"/"不是"这种短词是同一个理由，不追求语法完整，只要"紧挨着
# 命中位置前面出现"这个信号本身就够。
_ALLERGY_NEGATION_MARKERS = ("没有", "不是", "并不", "不对", "并非", "not", "n't")
# 2026-09-01: 6 → 10，给英文否定词留出空间(比如"not"紧跟在"allergic"前面
# 时刚好够，太紧的话连"is not"都装不下)。现有"否定词离命中太远不应该压制"
# 的单测用的前缀是 35 个字符，10 远小于 35，不会把那条测试变成假阳性；
# 现有"否定词紧贴命中"的单测词间距是 0，同样不受影响——加宽这一档不影响
# 任何已验证过的边界，只是给英文流出一点必要的余量。
_ALLERGY_WINDOW = 10


@dataclass(frozen=True)
class AllergenMention:
    category: str
    matched_term: str


def _build_allergy_patterns() -> dict[str, tuple[re.Pattern[str], ...]]:
    """每个类别三种常见中文句式("对X过敏"/"X过敏"/"过敏原是X")+ 三种对应的
    英文句式("allergic to X"/"X allergy"/"allergy: X")。关键词按长度降序
    排列在同一个交替分组里，保证"甲壳类"这种复合词优先于子串"蟹"被匹配到
    (虽然这里两者结果一样都算命中，但和项目别处同一个技巧——比如
    routing.py 的连接词分组——保持一致的写法习惯)。中英文关键词混在同一个
    分组里，中文模板碰到英文关键词(或反过来)本身不会误判——只是不会有人
    真的说"我对shrimp过敏"，两个模板×两种关键词的交叉组合不追求都对应
    自然语句，能命中的那一组(中文模板配中文词、英文模板配英文词)才是
    真正覆盖的场景。大小写不敏感(`re.IGNORECASE`)：英文句首/用户随手打字
    经常不注意大小写。"""
    patterns: dict[str, tuple[re.Pattern[str], ...]] = {}
    for category, keywords in ALLERGEN_CATEGORY_KEYWORDS.items():
        kw_group = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
        patterns[category] = (
            re.compile(rf"对({kw_group}).{{0,6}}过敏"),
            re.compile(rf"({kw_group}).{{0,4}}过敏"),
            re.compile(rf"过敏原?(?:是|有|包括|[:：])\s*.{{0,4}}({kw_group})"),
            re.compile(rf"\ballerg(?:y|ic)\s+to\s+({kw_group})\b", re.IGNORECASE),
            re.compile(rf"\b({kw_group})\s+allerg(?:y|ies)\b", re.IGNORECASE),
            re.compile(
                rf"\ballerg(?:y|ies)?\s*(?:is|are|include[s]?|:)\s*({kw_group})\b", re.IGNORECASE
            ),
        )
    return patterns


_ALLERGY_PATTERNS = _build_allergy_patterns()


def _has_nearby_negation(text: str, start: int, markers: tuple[str, ...], window: int) -> bool:
    # 大小写不敏感——中文没有大小写，对中文 marker 是空操作；英文
    # "Not allergic"这类句首大写不应该逃过否定检查。
    preceding = text[max(0, start - window) : start].lower()
    return any(marker.lower() in preceding for marker in markers)


def scan_allergen_mentions(text: str) -> tuple[AllergenMention, ...]:
    """扫描一段自由文本，返回其中出现的过敏原声明。

    只认"对XX过敏/XX过敏/过敏原是XX"这类明确的过敏声明句式，不是"提到某个
    食材"就命中——那会把每一句正常聊天(比如"今天吃了虾")都误判成过敏声明。
    这是刻意的保守取舍：宁可漏掉"我对虾这种东西一直不太耐受"这类不走标准
    句式的表述(仍然会被 §4.2 记录分支/核查 pass 的过敏原硬阻断兜底)，也不
    把每一次提到食材都当成过敏声明写库。"""
    if not text:
        return ()
    found: dict[str, str] = {}
    for category, patterns in _ALLERGY_PATTERNS.items():
        for pattern in patterns:
            matched = False
            for m in pattern.finditer(text):
                if _has_nearby_negation(text, m.start(), _ALLERGY_NEGATION_MARKERS, _ALLERGY_WINDOW):
                    continue
                found[category] = m.group(1)
                matched = True
                break
            if matched:
                break
    return tuple(AllergenMention(category=c, matched_term=t) for c, t in found.items())


# ---------------------------------------------------------------------------
# 补剂提及扫描
# ---------------------------------------------------------------------------

# 种子词表，同上——常见的非处方膳食补剂。处方药/慢性病用药走的是另一条完全
# 不同的检查(backend/guardrails/input_filters.py `detect_medical_intent()`，
# 触发受限模式而不是记忆写入)，两者关注点不同、词表也不共用：那边关心的是
# "有没有潜在药物-食物相互作用需要转诊"，这里关心的是"这条信息值不值得记进
# 画像供以后的建议参考"。
SUPPLEMENT_KEYWORDS = (
    "复合维生素", "维生素D", "维生素C", "维生素B", "维C", "维生素",
    "鱼油", "钙片", "益生菌", "蛋白粉", "褪黑素", "DHA", "欧米伽3",
    "铁剂", "叶酸", "辅酶Q10", "氨基酸片", "胶原蛋白",
    # 2026-09-01 补英文补剂名。
    "multivitamin", "vitamin D", "vitamin C", "vitamin B", "vitamin",
    "fish oil", "calcium", "probiotics", "protein powder", "melatonin",
    "omega-3", "omega 3", "iron", "folic acid", "folate", "coenzyme Q10",
    "CoQ10", "amino acid", "collagen",
)

_SUPPLEMENT_NEGATION_MARKERS = (
    "没", "不再", "已经不", "停了", "戒掉", "停用",
    # 2026-09-01 补英文——同上面 _ALLERGY_WINDOW 的理由，窗口宽到 10 才够
    # 装下"stopped"这类词。
    "not", "n't", "stopped", "quit",
)
_SUPPLEMENT_WINDOW = 10


def _build_supplement_pattern() -> re.Pattern[str]:
    # ⚠️ 英文动词故意只用 "taking"/"take"，不用 "on"——"on" 太常见("on a
    # diet"/"based on"/"focused on X")，当补剂提及的触发词误报率会很高，
    # 不值得为了接住"我 on 鱼油"这种口语说法冒这个险。
    kw_group = "|".join(re.escape(k) for k in sorted(SUPPLEMENT_KEYWORDS, key=len, reverse=True))
    return re.compile(rf"(?:吃|服用|taking|take)\s*({kw_group})", re.IGNORECASE)


_SUPPLEMENT_PATTERN = _build_supplement_pattern()


@dataclass(frozen=True)
class SupplementMention:
    name: str


def scan_supplement_mentions(text: str) -> tuple[SupplementMention, ...]:
    """扫描"我在吃/正在服用/吃着 XX"这类补剂提及模式。"停了鱼油"/"已经不吃钙片
    了"这类明确表示"不再服用"的表述不算命中——见 `_SUPPLEMENT_NEGATION_MARKERS`。"""
    if not text:
        return ()
    found: dict[str, None] = {}
    for m in _SUPPLEMENT_PATTERN.finditer(text):
        if _has_nearby_negation(text, m.start(), _SUPPLEMENT_NEGATION_MARKERS, _SUPPLEMENT_WINDOW):
            continue
        found.setdefault(m.group(1), None)
    return tuple(SupplementMention(name=n) for n in found)


# ---------------------------------------------------------------------------
# 聚合入口
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticalFactScanResult:
    """相对 `profile` 现有内容的**增量**——已经记录过的过敏原类别/补剂名称不
    会重复出现在这里，调用方不需要自己再去重一遍。"""

    new_allergens: tuple[str, ...] = ()
    new_supplements: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return bool(self.new_allergens or self.new_supplements)


def scan_critical_facts(
    text: str, profile: UserProfileContext | None = None
) -> CriticalFactScanResult:
    """本模块对外的唯一入口——`api/main.py` 只调这一个函数(以及命中时接着调
    下面的 `merge_into_profile()`)。纯函数，不碰数据库/MCP，方便单测不需要
    mock 任何 I/O(100% 覆盖要求见模块文档)。"""
    existing_allergens = set(profile.allergens) if profile else set()
    existing_supplements = {s.get("name") for s in (profile.supplements if profile else ())}

    allergen_categories = {m.category for m in scan_allergen_mentions(text)}
    new_allergens = tuple(sorted(allergen_categories - existing_allergens))

    supplement_names = {m.name for m in scan_supplement_mentions(text)}
    new_supplements = tuple(sorted(supplement_names - existing_supplements))

    return CriticalFactScanResult(new_allergens=new_allergens, new_supplements=new_supplements)


def merge_into_profile(
    result: CriticalFactScanResult,
    profile: UserProfileContext | None,
    *,
    user_id: str = "default_user",
) -> tuple[dict[str, Any], UserProfileContext]:
    """把扫描到的增量事实和当前画像合并，返回 `(write_memory 的 payload，
    更新后的 UserProfileContext)`。只在 `result.hit` 为真时才应该调用——调用
    方负责判断，这里不重复判断一遍。

    返回的 `UserProfileContext` 供**同一轮请求剩余流程**使用——比如这一轮
    消息本身就是"对了我对虾过敏，今天该吃什么"，SubAgent 生成建议时就该已经
    避开虾，不用等下一轮请求重新查一次数据库才生效(这也是 §4.3 举的原始例子:
    "在问'今天该吃什么'时顺带提一句'对了我对虾过敏'")。`allergens`/
    `supplements` 是 `write_memory(critical)` 的 UPSERT 语义(整列覆盖，不是
    数组追加)——这里必须传完整的合并后列表，不能只传新增的那部分，否则会把
    用户已经记过的过敏原/补剂覆盖掉。
    """
    merged_allergens = tuple(sorted({*(profile.allergens if profile else ()), *result.new_allergens}))
    merged_supplements = (
        *(profile.supplements if profile else ()),
        *({"name": name, "dose": None} for name in result.new_supplements),
    )

    payload: dict[str, Any] = {}
    if result.new_allergens:
        payload["allergens"] = list(merged_allergens)
    if result.new_supplements:
        payload["supplements"] = list(merged_supplements)

    if profile is not None:
        updated_profile = dataclasses.replace(
            profile, allergens=merged_allergens, supplements=merged_supplements
        )
    else:
        updated_profile = UserProfileContext(
            user_id=user_id, allergens=merged_allergens, supplements=merged_supplements
        )
    return payload, updated_profile
