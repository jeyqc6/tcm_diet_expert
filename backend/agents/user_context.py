#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中枢从 `user_profile` 读取当前用户画像，供 SubAgent / 调和层 / 核查 pass 使用。

设计依据：docs/ARCHITECTURE.md §5.2 步骤 2("中枢准备派发上下文:读 user_profile(常驻)")
决策依据：docs/DECISIONS.md D28(体质主/次/来源字段)、D25(preferences/goal_tags 区分)

⚠️ 这个模块存在的理由：`api/main.py` 此前(阶段4任务11)完全没有查询过 `user_profile`
——`run_tcm_subagent()` 一直是不带 `constitution` 参数调用的，永远走 D28 的"体质未知"
降级路径，哪怕数据库里已经有一行填好体质的画像也读不到。这是 2026-08-26 真实跑通
`/api/chat` 之后审计发现的接线缺口，本模块 + `api/main.py` 里对应的改动是补上这一步。

2026-08-30 补充：从"预留参数但只有一行"升级成真的多用户——`user_profile.user_id`
本来就是 UNIQUE 列，`list_users()`/`create_user()`是这次新增的用户管理入口，
`DEFAULT_USER_ID` 仍然是没显式传 user_id 时的兜底值(向后兼容旧调用点/旧测试)，
不是唯一合法值了。`display_name` 是纯展示字段，不参与任何隔离逻辑——`user_id`
才是真正贯穿 `conversation_sessions`/`messages`/`diet_log`/MCP 工具调用的隔离键。

失败即静默降级——查不到画像(表空/未建档/连不上库)不应该让整条 `/api/chat` 请求
失败:TCM SubAgent 本身已经实现了"体质未知"降级(D28)，调和层/核查 pass 也都能接受
`user_profile=None`/空摘要，读画像失败就按"用户还没有画像"处理，不是系统性故障。
这条原则和 `backend/mcp_server/tools/query_diet_log.py` 里 `_fetch_user_timezone()`
的既有先例一致，不是本模块独创。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn
from backend.i18n import DEFAULT_LOCALE, normalize_locale

DEFAULT_USER_ID = "default_user"

_PROFILE_COLUMNS = (
    "display_name",
    "constitution",
    "constitution_secondary",
    "constitution_source",
    "allergens",
    "supplements",
    "goal_tags",
    "preferences",
    "city",
    "timezone",
    "onboarding_done",
    "locale",
)


@dataclass(frozen=True)
class UserProfileContext:
    user_id: str
    display_name: str | None = None
    constitution: str | None = None
    constitution_secondary: tuple[str, ...] = ()
    constitution_source: str | None = None
    allergens: tuple[str, ...] = ()
    # In-use supplements: [{"name": "维生素D", "dose": "1000IU/day"}, ...].
    # Scanner writes this field (dose is usually None). After confirm (D34)
    # these values are injected into SubAgent / reconcile / verify prompts.
    supplements: tuple[dict[str, Any], ...] = ()
    goal_tags: tuple[str, ...] = ()
    preferences: dict[str, Any] = field(default_factory=dict)
    city: str | None = None
    timezone: str | None = None
    # Set True when the first-conversation intro finishes (including skip-all).
    onboarding_done: bool = False
    # UI / conversation language. Default zh matches ChatRequest and schema.
    locale: str = DEFAULT_LOCALE

    def constitutions(self) -> tuple[str, ...]:
        """主 + 次体质合并，供 conflict_rules 匹配用——体质夹杂(D28)时，次要体质
        命中的规则也该被考虑,不能只按主体质过滤。"""
        if self.constitution:
            return (self.constitution, *self.constitution_secondary)
        return self.constitution_secondary

    def to_reconciliation_dict(self) -> dict[str, Any]:
        """喂给 `reconcile()` 的 `user_profile` 参数。D14 的边界不在这里——那条
        边界是"调和层不接收原始检索内容"，和用户画像是两回事;这里只是把画像
        序列化成调和层 prompt(`_format_user_profile`)能读的 dict。"""
        return {
            "constitution": self.constitution,
            "constitution_secondary": list(self.constitution_secondary),
            "allergens": list(self.allergens),
            "supplements": [self._supplement_label(item) for item in self.supplements],
            "goal_tags": list(self.goal_tags),
            "preferences": self.preferences,
        }

    def to_verification_summary(self) -> str:
        """喂给 `verify()` 的 `user_profile_summary` 参数——一句话摘要，不是整份
        JSON:核查 pass 的 prompt 预算(§5.2 步骤6 ≤12k)比调和层更紧。过敏原/
        体质仍是硬边界；补剂与偏好必须出现，否则核查第 5 条(补剂交互)和忌口
        对不上生成侧已经注入的信息。"""
        parts = []
        if self.constitution:
            parts.append(f"体质:{self.constitution}")
        if self.allergens:
            parts.append(f"过敏原:{','.join(self.allergens)}")
        supplement_labels = [self._supplement_label(item) for item in self.supplements]
        if supplement_labels:
            parts.append(f"在服补剂:{','.join(supplement_labels)}")
        if self.preferences:
            parts.append(f"偏好:{json.dumps(self.preferences, ensure_ascii=False)}")
        if not parts:
            return ""
        return "；".join(parts)

    def profile_prompt_notes(self) -> str:
        """Injected into TCM/Nutrition SubAgent system prompts (not the hub).

        ⚠️ `city` 必须显式写进这句话，不能假设 SubAgent 自己知道用户在哪个
        城市——之前没写这一条时，SubAgent 调 `query_weather(city=...)` 会
        编一个占位字符串(比如 "用户当前城市" 这种描述性文字本身)当城市名传
        进去，Open-Meteo 地理编码当然查不到，工具原样按"接口没连上"降级成
        节气兜底，看起来像是天气 API 挂了，其实是画像里明明有 `city` 字段
        但从没喂给过模型(ARCHITECTURE.md §1.1"`city` 字段直接取，不用每次
        对话都问"这句话默认了"喂给谁用"，但漏了这一步)。"""
        parts: list[str] = []
        if self.city:
            parts.append(
                f"用户当前城市:{self.city}。调用 query_weather 的 city 参数时"
                "直接传这个值，不要传占位符或猜测的城市名。"
            )
        names = [self._supplement_label(item) for item in self.supplements]
        if names:
            parts.append(
                "用户在服补剂:"
                + "、".join(names)
                + "。检索不到药食/补剂交互依据时，不得编造交互；"
                "必须声明不确定，并提示咨询医生（E8 / PRD §9 Critical，不是医嘱）。"
            )
        if self.preferences:
            parts.append(
                "用户饮食偏好:" + json.dumps(self.preferences, ensure_ascii=False)
            )
        return "\n".join(parts)

    @staticmethod
    def _supplement_label(item: Any) -> str:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            dose = str(item.get("dose") or "").strip()
            if name and dose:
                return f"{name}({dose})"
            return name
        return str(item).strip()


def _row_to_context(user_id: str, row: dict[str, Any]) -> UserProfileContext:
    preferences = row.get("preferences") or {}
    if isinstance(preferences, str):
        # psycopg2 通常已经把 JSONB 解成 dict；这里兜底极少数返回原始字符串的情况。
        try:
            preferences = json.loads(preferences)
        except json.JSONDecodeError:
            preferences = {}
    supplements = row.get("supplements") or ()
    if isinstance(supplements, str):
        # 同上一段 preferences 的兜底：psycopg2 通常已经把 JSONB 解成
        # list[dict]，这里兜底极少数返回原始字符串的情况。
        try:
            supplements = json.loads(supplements)
        except json.JSONDecodeError:
            supplements = ()
    return UserProfileContext(
        user_id=user_id,
        display_name=row.get("display_name"),
        constitution=row.get("constitution"),
        constitution_secondary=tuple(row.get("constitution_secondary") or ()),
        constitution_source=row.get("constitution_source"),
        allergens=tuple(row.get("allergens") or ()),
        supplements=tuple(supplements),
        goal_tags=tuple(row.get("goal_tags") or ()),
        preferences=preferences,
        city=row.get("city"),
        timezone=row.get("timezone"),
        onboarding_done=bool(row.get("onboarding_done")),
        locale=normalize_locale(row.get("locale")),
    )


def ensure_user_profile(
    user_id: str = DEFAULT_USER_ID, dsn: str | None = None
) -> bool:
    """Insert a stub `user_profile` row if none exists.

    Called when onboarding is first offered so a user with no row yet still
    has one after this call. `should_trigger` keys off `onboarding_done`,
    not row existence (`create_user` already inserts a stub with the flag FALSE).
    Failure is silent for the same reason as `fetch_user_profile`: not being
    able to stamp the row must not block the onboarding prompts themselves.
    Returns True when the row exists afterwards (inserted or already there).
    """
    if psycopg2 is None:
        return False
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        return False
    try:
        conn = psycopg2.connect(resolved_dsn)
    except Exception:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_profile (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING",
            (user_id,),
        )
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def fetch_user_profile(
    user_id: str = DEFAULT_USER_ID, dsn: str | None = None
) -> UserProfileContext | None:
    """查不到就返回 None，不抛异常——调用方按"用户还没有画像"处理。"""
    if psycopg2 is None:
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
        try:
            cur.execute(
                f"SELECT {', '.join(_PROFILE_COLUMNS)} FROM user_profile WHERE user_id = %s",
                (user_id,),
            )
        except Exception:
            # Pre-locale schema: retry without the new column so chat still
            # works until schema.sql has been re-applied.
            conn.rollback()
            fallback = tuple(c for c in _PROFILE_COLUMNS if c != "locale")
            cur.execute(
                f"SELECT {', '.join(fallback)} FROM user_profile WHERE user_id = %s",
                (user_id,),
            )
        row = cur.fetchone()
        cur.close()
    except Exception:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    return _row_to_context(user_id, dict(row))


def list_users(dsn: str | None = None) -> list[dict[str, str]]:
    """前端用户切换器的数据源——现在库里有几行 `user_profile` 就是几个"用户"。
    查不到/连不上库时返回空列表，同本模块其余读路径的静默降级原则；前端把
    空列表当"还没有任何用户"处理，走一次创建用户的引导。"""
    if psycopg2 is None:
        return []
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        return []
    try:
        conn = psycopg2.connect(resolved_dsn)
    except Exception:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id, display_name FROM user_profile ORDER BY id ASC")
        rows = cur.fetchall()
        cur.close()
    except Exception:
        return []
    finally:
        conn.close()
    return [{"user_id": r[0], "name": r[1] or r[0]} for r in rows]


def create_user(name: str, dsn: str | None = None) -> dict[str, str] | None:
    """新建一个用户——生成一个新的 `user_id`(前缀 `u_` 避免和历史上手写的
    `default_user` 撞到一起)，`display_name` 就是调用方传的名字，纯展示用，
    不参与任何隔离逻辑。失败(连不上库/写入异常)返回 None，调用方(API 层)
    决定怎么报错——这条是写路径，不能像读路径那样静默吞掉再假装成功。"""
    if psycopg2 is None:
        return None
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        return None
    new_user_id = f"u_{uuid.uuid4().hex[:12]}"
    try:
        conn = psycopg2.connect(resolved_dsn)
    except Exception:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_profile (user_id, display_name) VALUES (%s, %s)",
            (new_user_id, name),
        )
        conn.commit()
        cur.close()
    except Exception:
        return None
    finally:
        conn.close()
    return {"user_id": new_user_id, "name": name}


def persist_user_locale(
    user_id: str, locale: str, dsn: str | None = None
) -> bool:
    """Best-effort write of user_profile.locale.

    Chat must not fail if the DSN is missing, the column isn't migrated yet,
    or the UPDATE errors — same silent-degrade rule as fetch_user_profile.
    """
    if psycopg2 is None:
        return False
    resolved = normalize_locale(locale)
    resolved_dsn = get_pg_dsn(dsn)
    if not resolved_dsn:
        return False
    try:
        conn = psycopg2.connect(resolved_dsn)
    except Exception:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE user_profile SET locale = %s, updated_at = now() WHERE user_id = %s",
            (resolved, user_id),
        )
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False
    finally:
        conn.close()
