#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三级查找：dish_ingredient_map → user_dish_aliases(仅已晋升) → LLM 兜底。

设计依据：docs/ARCHITECTURE.md §4.2
决策依据：docs/DECISIONS.md D27 修订一

⚠️ **`dish_ingredient_map` 不是 Postgres 表**——查了 ARCHITECTURE.md §1.2"待建表"
清单，里面没有这张表；§4.2 正文把它叫"表"是行文习惯，实际数据资产是
`knowledge/food/dish-decomposition.jsonl`(44 条种子数据)，本模块直接读这个
JSONL 到内存字典，不建 DB 表——44-200 条规模，建表反而是过度设计。

**三级查找的具体做法(不是按"整句话切成一个个菜名"这种脆弱的分词)**：
1. **全局表**：把已知的每个菜名当子串，在整句原始输入里扫——"晚上吃了番茄炒蛋
   加一小碗白米饭"里"番茄炒蛋"作为子串天然能被扫到，不需要先猜"这句话在哪里
   断句"。命中后把匹配到的文本从"剩余文本"里去掉，避免和后面的菜名重叠计数；
   长菜名优先扫(比如"蚝油炒芥蓝"先于"芥蓝")，避免短菜名子串抢先命中导致更具体
   的菜名信息丢失。
2. **个人别名**：整句"剩余文本"(去空白/标点归一化后)如果**整体**等于某条已
   晋升的 `user_dish_aliases.normalized_phrase`，直接取那一行的结构化结果——
   个人别名是"这个用户这样说时指的是什么"，天然是针对一整句自定义说法，不是
   单个菜名的子串匹配，所以这一级用整体相等，不是子串。
3. **LLM 兜底**：全局表和个人别名都没能覆盖的剩余文本，交给 LLM 结构化输出
   一次，标记 `confidence="low"`。这一步失败(网络错误等)会让异常往上抛，调用方
   (`api/main.py` 的 log_write 分支)决定怎么降级，不在本模块内吞掉。

顺带说明为什么不专门实现"切分成多个菜名"这一步：子串扫描已经隐式做到了
"这句话里提到了哪几个已知菜"，不需要一个独立的、容易切错的分词步骤；LLM 兜底
天然能处理"剩余文本里其实还包含不止一个东西"这种情况(结构化输出本身就是
一个列表)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn
from backend.i18n import apply_language_instruction, current_locale
from backend.llm.adapter import CompleteFn

_DISH_TABLE_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "food" / "dish-decomposition.jsonl"

CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"
SOURCE_GLOBAL_TABLE = "global_table"
SOURCE_USER_ALIAS = "user_alias"
SOURCE_LLM_FALLBACK = "llm_fallback"

_STRIP_RE = re.compile(r"[\s，,。.！!？?、；;：:]+")


def normalize_phrase(text: str) -> str:
    """去空白/标点后的原始说法——和 `user_dish_aliases.normalized_phrase` 的
    既有字段注释("去空白/标点后的原始说法")一致，标点/空白差异视为同一短语。"""
    return _STRIP_RE.sub("", text.strip())


@dataclass(frozen=True)
class DishMatch:
    dish: str
    ingredients: tuple[str, ...]
    tcm_nature: str | None
    allergens: tuple[str, ...]
    confidence: str
    source_tier: str


@dataclass(frozen=True)
class MealDecomposition:
    matches: tuple[DishMatch, ...]

    def all_ingredients(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for m in self.matches:
            for ing in m.ingredients:
                seen.setdefault(ing, None)
        return tuple(seen)

    def all_food_properties(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for m in self.matches:
            if m.tcm_nature:
                seen.setdefault(m.tcm_nature, None)
        return tuple(seen)

    def all_allergens(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for m in self.matches:
            for a in m.allergens:
                seen.setdefault(a, None)
        return tuple(seen)


@lru_cache(maxsize=1)
def _load_dish_table() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    if not _DISH_TABLE_PATH.is_file():
        return table
    with _DISH_TABLE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            table[row["dish"]] = row
    return table


def clear_dish_table_cache() -> None:
    """测试用：`_load_dish_table` 是 `lru_cache`，测试如果要注入临时文件内容
    需要先清缓存。"""
    _load_dish_table.cache_clear()


def _record_to_match(record: dict[str, Any]) -> DishMatch:
    return DishMatch(
        dish=record["dish"],
        ingredients=tuple(record.get("ingredients") or ()),
        tcm_nature=record.get("tcm_nature") or None,
        allergens=tuple(record.get("allergens") or ()),
        confidence=CONFIDENCE_HIGH,
        source_tier=SOURCE_GLOBAL_TABLE,
    )


def match_global_table(text: str) -> tuple[tuple[DishMatch, ...], str]:
    """返回(命中的 DishMatch 列表, 去掉命中片段后的剩余文本)。长菜名优先扫，
    命中后把该片段从剩余文本里删掉，避免被后面更短的菜名子串重复计数。"""
    table = _load_dish_table()
    remaining = text
    matches: list[DishMatch] = []
    for dish_name in sorted(table, key=len, reverse=True):
        if dish_name and dish_name in remaining:
            matches.append(_record_to_match(table[dish_name]))
            remaining = remaining.replace(dish_name, "")
    return tuple(matches), remaining


def _fetch_promoted_alias(
    user_id: str, normalized_phrase: str, dsn: str | None
) -> DishMatch | None:
    """只命中 `promoted_at IS NOT NULL` 的行——未晋升的候选行不该被当作"确定
    可信"的结果使用，那是给 dish_alias_promotion.py 计数用的候选状态。"""
    if not normalized_phrase or psycopg2 is None:
        return None
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        return None
    try:
        conn = psycopg2.connect(resolved_dsn)
    except Exception:
        return None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT dishes, ingredients FROM user_dish_aliases
            WHERE user_id = %s AND normalized_phrase = %s AND promoted_at IS NOT NULL
            """,
            (user_id, normalized_phrase),
        )
        row = cur.fetchone()
        cur.close()
    except Exception:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    dishes = row.get("dishes") or []
    dish_names = "+".join(d.get("dish", "?") for d in dishes) if dishes else normalized_phrase
    return DishMatch(
        dish=dish_names,
        ingredients=tuple(row.get("ingredients") or ()),
        tcm_nature=None,
        allergens=(),
        confidence=CONFIDENCE_HIGH,
        source_tier=SOURCE_USER_ALIAS,
    )


_LLM_SYSTEM_PROMPT = (
    "你是菜品拆解助手。给定一段用户描述吃了什么的文字，把里面**没有被识别过**的"
    "部分拆解成菜品/食材，只输出这一个 JSON 对象本身，前后不要有任何解释文字：\n"
    '{"dishes":[{"dish":"菜名或食物名","ingredients":["食材1","食材2"],'
    '"tcm_nature":"温|寒|凉|平|热|null","allergens":["可能的过敏原，没有就空数组"]}]}\n'
    "如果这段文字里其实什么食物都没提到，返回 {\"dishes\":[]}。"
)


def _strip_json_fences(text: str) -> str:
    """同 `backend/agents/router.py`/`verification.py` 里同名的既有做法——
    真实跑真实模型时发现的问题：即便 system prompt 明确写了"只输出严格 JSON，
    不要任何解释文字"，Haiku 仍然会把 JSON 包在 ```json ... ``` 代码块里，
    直接 `json.loads` 会报错。这三处目前各自复制一份而不是共用一个函数——
    和既有代码保持同一种"重复三行小片段"的风格，不是这次新引入的模式。

    另一个独立观察到的问题：模型有时会在 JSON 后面附加解释文字（尽管 prompt
    已经写了"只输出严格 JSON，不要任何解释文字"）。原先的实现只处理"整段被
    ```包住"这一种情况，JSON 后面拖着自然语言时 `json.loads` 直接失败。这里
    在去掉代码块围栏之后，再定位第一个括号平衡的 {...} 片段，尾部多出来的
    文字就不会影响解析了。"""
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


def _parse_llm_dishes(raw_text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(_strip_json_fences(raw_text or ""))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    dishes = data.get("dishes")
    return dishes if isinstance(dishes, list) else []


async def llm_fallback_decompose(
    text: str, complete: CompleteFn, locale: str | None = None
) -> tuple[DishMatch, ...]:
    """LLM 兜底——只在全局表/个人别名都没能覆盖剩余文本时调用。失败(网络/
    格式错误)不在这里吞掉，让调用方决定怎么降级(比如只展示已经用确定性方式
    识别出的部分，剩余部分诚实报告"未能识别")。"""
    result = await complete(
        [
            {
                "role": "system",
                "content": apply_language_instruction(
                    _LLM_SYSTEM_PROMPT, locale if locale is not None else current_locale()
                ),
            },
            {"role": "user", "content": text},
        ],
        force_prod_tier=False,
    )
    raw_dishes = _parse_llm_dishes(result.text or "")
    return tuple(
        DishMatch(
            dish=d.get("dish", text),
            ingredients=tuple(d.get("ingredients") or ()),
            tcm_nature=d.get("tcm_nature") or None,
            allergens=tuple(d.get("allergens") or ()),
            confidence=CONFIDENCE_LOW,
            source_tier=SOURCE_LLM_FALLBACK,
        )
        for d in raw_dishes
        if isinstance(d, dict)
    )


async def decompose_meal(
    raw_input: str,
    *,
    user_id: str = "default_user",
    dsn: str | None = None,
    complete: CompleteFn,
    locale: str | None = None,
) -> MealDecomposition:
    """三级查找的编排入口——`api/main.py` 的 log_write 分支调这一个函数。

    2026-08-31 修订：命中全局表之后，只要还有非空残留文本就继续查(个人别名→
    LLM 兜底)，不再靠一份连接词关键词表("加"/"和"/"还有"...)去猜"残留文本
    里是不是还提到了别的东西"。真实数据发现这份关键词表覆盖不了用户实际
    使用的所有分隔习惯("鸡爪+牛肉南瓜炖年糕+半个小西瓜"用"+"分隔，不在表
    里)，命中"南瓜"/"西瓜"这两个全局表词条后就直接短路，"鸡爪"和"牛肉南瓜
    炖年糕"整个丢了，且这类关键词表本质上永远会有覆盖不到的分隔方式(换行、
    "/"、纯空格分隔...)，不是加几个词就能穷举完的。

    这个决定不是没有代价：早先(2026-08-27 之前)的实现就是"只要有残留就查
    LLM"，结果发现"帮我记录一下，中午吃了麻婆豆腐"这类纯叙述性填充词也会
    触发一次 LLM 调用，几乎每条 log_write 消息都要多打一次——用关键词表短路
    正是为了避免这个成本才加的。这次改回"无条件继续查"，等于把这个成本换
    回来：这类残留是纯填充词的场景，多打的这次 LLM 调用会正确返回空
    dishes(见 `_LLM_SYSTEM_PROMPT`"如果这段文字里其实什么食物都没提到，
    返回 {"dishes":[]}")，结果不会错，但确实多花一次 LLM 调用——这是主动
    权衡"宁可稍微多花一点，也不要漏记" (2026-08-31 决定)，不是没考虑到
    这个成本。
    """
    global_matches, remaining = match_global_table(raw_input)
    normalized_remaining = normalize_phrase(remaining)

    if not normalized_remaining:
        return MealDecomposition(matches=global_matches)

    alias_match = _fetch_promoted_alias(user_id, normalized_remaining, dsn)
    if alias_match is not None:
        return MealDecomposition(matches=(*global_matches, alias_match))

    llm_matches = await llm_fallback_decompose(remaining, complete=complete, locale=locale)
    return MealDecomposition(matches=(*global_matches, *llm_matches))
