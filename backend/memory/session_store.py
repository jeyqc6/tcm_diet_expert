#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 `backend/memory/compression.py` 的压缩算法接进真实的 `/api/chat` 请求
生命周期——`conversation_sessions`/`messages` 的读写、Tier1→Tier2 归档的
实际执行、会话空闲后 Tier2→Tier3 的折叠、组装喂给下一轮请求的会话历史文本。

设计依据：docs/ARCHITECTURE.md §4.4.1(两级触发的具体检查点)
决策依据：docs/DECISIONS.md D27 修订二

## 和 compression.py 的分工

`compression.py` 只做纯函数(数据结构、优先级表、触发判断、模板组装)，不碰
数据库——这个文件是它和真实 Postgres 之间唯一的桥梁，负责"什么时候读、什么
时候写、写成什么形状"，不重新实现任何压缩逻辑本身。DB 访问模式照抄
`backend/agents/user_context.py`/`backend/mcp_server/tools/write_memory.py`
的既有先例：`psycopg2.connect()` 按调用建连接(不是连接池，这个项目目前所有
写路径都是这样)，读路径查不到/连不上一律静默降级为空结果，绝不抛异常阻塞
`/api/chat`——压缩/归档是锦上添花的能力，不能因为它失败拖垮或污染已经算好
要返回给用户的这次响应。

## 两级触发在这里的落地

- **步骤9(响应发出之后)**：`record_turn()`——写入这一轮的原文(Tier1)，写完
  顺带检查 `should_archive_tier1()`，命中就用 `select_turns_to_archive()`
  挑出最旧的几轮，`build_archived_summary()` 渲染成结构化摘要，原地把这些
  行的 `compression_tier` 从 0(原文)改成 1(Tier2 摘要，会话仍在进行)。
  调用方(`api/main.py`)用 `asyncio.create_task()` 触发、不 `await`——这是
  这个项目里"不占用当前请求响应时间"的实际实现方式，没有 Celery 这类真正
  的后台任务队列，`create_task` 是 FastAPI 原生等价物。
- **会话空闲判定(Tier2→Tier3)**：没有真正的后台调度器，折叠检查挂在"新消息
  到达时"这个天然触发点上——`maybe_fold_idle_session()` 在处理一条新消息
  之前调用，如果这个 `session_id` 距上次活跃已经超过
  `compression.SESSION_IDLE_THRESHOLD_SECONDS`，就把它现有的全部 Tier2 摘要
  折叠成 Tier3(只改 `compression_tier`，摘要内容本身不变——折叠是存储层面
  的生命周期状态变化，不是数据变换，见 `compression.is_session_idle()` 文档)。
- **步骤2(下一轮请求组装上下文，同步兜底)**：`load_session_history()`——读
  Tier1 原文 + Tier2/Tier3 结构化摘要按时间顺序拼接；如果摘要部分本身已经
  超预算(说明步骤9的归档任务因为某种原因没跟上)，用
  `compression.drop_oldest_until_within_budget()` 就地丢弃最旧的摘要，不
  等待、不调用 LLM。

## 已知限制(如实记录)

- 多任务(D32)一句话拆成几个子任务时，这里只记一整条合并后的轮次(`branch`
  用 `+` 连接涉及到的分支名)，不是每个子任务单独一行——`messages` 表按
  `(session_id, turn_index)` 唯一，这个粒度对应"一条用户消息"而不是"一个
  子任务"，见 `api/main.py` 里构造 `TurnRecord` 的地方。
- ED 拦截 / 疾病受限模式 / 首次引导对话不经过这里——它们本来就不产出 D27
  意义上的"结论"(直接返回模板话术)，也不属于七选一分支，不记录进会话历史。
"""
from __future__ import annotations

import logging
import time
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from backend.env import get_pg_dsn
from backend.memory.compression import (
    TIER_ARCHIVED_ACTIVE,
    TIER_ARCHIVED_IDLE,
    TIER_RAW,
    ArchivedSummary,
    SESSION_IDLE_THRESHOLD_SECONDS,
    TurnRecord,
    build_archived_summary,
    drop_oldest_until_within_budget,
    select_turns_to_archive,
    should_archive_tier1,
)

logger = logging.getLogger("diet_expert.memory.session_store")

DEFAULT_USER_ID = "default_user"

# compression_tier 列值约定，见 db/schema.sql messages 表注释——数字本身
# 定义在 compression.py(`ArchivedSummary.tier` 需要用到)，这里 re-export
# 只是保留既有的 `from session_store import TIER_RAW` 这类调用方写法不用改。


def _connect(dsn: str | None = None):
    if psycopg2 is None:
        return None
    resolved = get_pg_dsn(dsn)
    if not resolved:
        return None
    try:
        return psycopg2.connect(resolved)
    except Exception:
        return None


def maybe_fold_idle_session(session_id: str, *, dsn: str | None = None) -> None:
    """新消息到达时的天然触发点——没有后台调度器，折叠检查只能挂在这里
    (见模块文档"会话空闲判定"一节)。距上次活跃超过
    `SESSION_IDLE_THRESHOLD_SECONDS` 才折叠；没有这个会话/连不上库时静默
    跳过，不影响这次请求正常处理。"""
    conn = _connect(dsn)
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT extract(epoch from updated_at) FROM conversation_sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            cur.close()
            return
        last_updated_ts = float(row[0])
        if (time.time() - last_updated_ts) <= SESSION_IDLE_THRESHOLD_SECONDS:
            cur.close()
            return
        cur.execute(
            "UPDATE messages SET compression_tier = %s WHERE session_id = %s AND compression_tier = %s",
            (TIER_ARCHIVED_IDLE, session_id, TIER_ARCHIVED_ACTIVE),
        )
        folded = cur.rowcount
        conn.commit()
        cur.close()
        if folded:
            logger.info("folded %d tier2 summary(ies) into tier3 · session=%s", folded, session_id)
    except Exception:
        logger.exception("idle-session fold check failed · session=%s", session_id)
    finally:
        conn.close()


def record_turn(
    session_id: str,
    turn: TurnRecord,
    *,
    user_id: str = DEFAULT_USER_ID,
    model: str | None = None,
    dsn: str | None = None,
) -> None:
    """§4.4.1 步骤9的写入部分——响应已经发出之后调用(`api/main.py` 用
    `asyncio.create_task()` 触发，不 `await`，见模块文档)。写完这一轮原文，
    顺带检查是否需要把最旧的若干轮归档成 Tier2 摘要。"""
    conn = _connect(dsn)
    if conn is None:
        logger.warning("record_turn skipped: db unavailable · session=%s", session_id)
        return
    try:
        cur = conn.cursor()
        # `SELECT MAX(turn_index)+1` 后面紧跟一条独立的 INSERT，两条语句
        # 之间没有锁——同一个 session_id 如果有两次 record_turn() 并发执行
        # (手快连发两条消息、追问重试撞上上一轮的后台写入)，两边可能读到
        # 同一个 MAX、算出同一个 turn_index，其中一条会撞 UNIQUE
        # (session_id, turn_index) 而被下面的 except 静默吞掉——那一轮对话
        # 就永久性地没有落库。用 session_id 哈希出的 key 做一次事务级
        # advisory lock，把同一个 session_id 的并发写入序列化；不同
        # session_id 哈希不同，互不阻塞；锁在事务 commit/rollback 时自动
        # 释放，不需要额外的清理代码。
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
        cur.execute(
            "INSERT INTO conversation_sessions (session_id, user_id) VALUES (%s, %s) "
            "ON CONFLICT (session_id) DO UPDATE SET updated_at = now()",
            (session_id, user_id),
        )
        cur.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM messages WHERE session_id = %s",
            (session_id,),
        )
        turn_index = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO messages
                (session_id, turn_index, role, content, branch, conclusion,
                 cited_source_ids, rejected_suggestions, triggered_guardrails, compression_tier)
            VALUES (%s, %s, 'assistant', %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                turn_index,
                turn.raw_text,
                turn.branch,
                turn.conclusion,
                list(turn.cited_source_ids),
                list(turn.rejected_suggestions),
                list(turn.triggered_guardrails),
                TIER_RAW,
            ),
        )
        conn.commit()
        cur.close()
    except Exception:
        logger.exception("record_turn failed · session=%s", session_id)
        return
    finally:
        conn.close()

    _maybe_archive_tier1(session_id, model=model, dsn=dsn)


def _load_tier1_turns(cur, session_id: str) -> list[TurnRecord]:
    cur.execute(
        "SELECT turn_index, content, branch, conclusion, cited_source_ids, "
        "rejected_suggestions, triggered_guardrails, extract(epoch from created_at) AS ts "
        "FROM messages WHERE session_id = %s AND compression_tier = %s ORDER BY turn_index ASC",
        (session_id, TIER_RAW),
    )
    rows = cur.fetchall()
    return [
        TurnRecord(
            turn_id=str(r["turn_index"]),
            branch=r["branch"] or "",
            raw_text=r["content"],
            conclusion=r["conclusion"] or "",
            cited_source_ids=tuple(r["cited_source_ids"] or []),
            rejected_suggestions=tuple(r["rejected_suggestions"] or []),
            triggered_guardrails=tuple(r["triggered_guardrails"] or []),
            timestamp=float(r["ts"]),
        )
        for r in rows
    ]


def _maybe_archive_tier1(session_id: str, *, model: str | None, dsn: str | None) -> None:
    conn = _connect(dsn)
    if conn is None:
        return
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        turns = _load_tier1_turns(cur, session_id)
        if not should_archive_tier1(turns, model=model):
            cur.close()
            return
        to_archive, _remaining = select_turns_to_archive(turns, model=model)
        for t in to_archive:
            summary = build_archived_summary(t)
            cur.execute(
                "UPDATE messages SET content = %s, compression_tier = %s "
                "WHERE session_id = %s AND turn_index = %s",
                (summary.render(), TIER_ARCHIVED_ACTIVE, session_id, int(t.turn_id)),
            )
        conn.commit()
        cur.close()
        if to_archive:
            logger.info(
                "archived %d turn(s) into tier2 · session=%s · turn_ids=%s",
                len(to_archive), session_id, [t.turn_id for t in to_archive],
            )
    except Exception:
        logger.exception("tier1 archive check failed · session=%s", session_id)
    finally:
        conn.close()


def load_session_history(session_id: str, *, model: str | None = None, dsn: str | None = None) -> str:
    """组装喂给下一轮请求的会话历史文本——Tier1 原文 + Tier2/Tier3 结构化
    摘要按轮次顺序拼接。查不到/连不上库/还没有任何历史时返回空字符串，
    调用方按"没有历史"处理，不阻塞请求(同 `fetch_user_profile` 的静默降级
    原则)。"""
    conn = _connect(dsn)
    if conn is None:
        return ""
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT turn_index, content, compression_tier, branch, conclusion, "
            "cited_source_ids, rejected_suggestions, triggered_guardrails "
            "FROM messages WHERE session_id = %s ORDER BY turn_index ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        cur.close()
    except Exception:
        logger.exception("load_session_history failed · session=%s", session_id)
        return ""
    finally:
        conn.close()

    if not rows:
        return ""

    summaries: list[ArchivedSummary] = []
    raw_lines: list[str] = []
    for r in rows:
        if r["compression_tier"] == TIER_RAW:
            raw_lines.append(r["content"])
        else:
            summaries.append(
                ArchivedSummary(
                    turn_id=str(r["turn_index"]),
                    branch=r["branch"] or "",
                    conclusion=r["conclusion"] or "",
                    cited_source_ids=tuple(r["cited_source_ids"] or []),
                    rejected_suggestions=tuple(r["rejected_suggestions"] or []),
                    triggered_guardrails=tuple(r["triggered_guardrails"] or []),
                    tier=r["compression_tier"],
                )
            )

    # 步骤2的同步紧急兜底(D27修订二)：只丢已经是摘要的部分，不动还没来得及
    # 归档的 Tier1 原文——不等待、不调用 LLM，见 compression.py 该函数文档。
    summaries = list(drop_oldest_until_within_budget(summaries, model=model))

    parts = [s.render() for s in summaries] + raw_lines
    return "\n".join(parts)


_USER_TEXT_PREFIX = "用户: "
_ASSISTANT_TEXT_SEP = "\n助手: "


def load_session_messages(session_id: str, *, dsn: str | None = None) -> list[dict[str, Any]]:
    """§10.1 `GET /api/sessions/{session_id}/messages`——按 turn_index 顺序
    返回这个 session 每一轮的结构化数据，供前端刷新页面后重建聊天气泡。

    和 `load_session_history()` 的区别：那个函数是为了拼给 LLM 当上下文用的
    纯文本，这里要的是给前端渲染用的结构化字段，不能复用。

    Tier1(原文,`compression_tier=0`)行的 `content` 是 `api/main.py`
    `_TurnAccumulator.build()` 拼的 `"用户: {user}\\n助手: {assistant}"` 文本，
    这里解析出 `user_text`；Tier2/Tier3(已归档)行的 `content` 已经被替换成
    结构化摘要的 `render()` 结果(D27)，原始用户提问已经不在数据库里了——
    `user_text` 为 `None`，前端不该为这类轮次伪造一条用户气泡，应该展示成
    一条摘要提示。`conclusion` 列在两种 tier 下都完整保留最终结论文本，直接
    用它做 `assistant_text`，不用再去解析 `content`。

    查不到/连不上库时返回空列表，同其它读路径的静默降级原则。"""
    conn = _connect(dsn)
    if conn is None:
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT turn_index, content, compression_tier, branch, conclusion, "
            "cited_source_ids, rejected_suggestions, triggered_guardrails, "
            "extract(epoch from created_at) AS created_at "
            "FROM messages WHERE session_id = %s ORDER BY turn_index ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        cur.close()
    except Exception:
        logger.exception("load_session_messages failed · session=%s", session_id)
        return []
    finally:
        conn.close()

    return [_message_row_to_dict(r) for r in rows]


def _message_row_to_dict(r: Any) -> dict[str, Any]:
    """`load_session_messages`/`load_all_messages` 共用的行→字典映射，两个
    函数的 SELECT 列表完全一致(`load_all_messages` 只是多 JOIN 了一张表做
    user_id 过滤)，解析逻辑不应该抄两份。"""
    tier = r["compression_tier"]
    content = r["content"] or ""
    user_text = None
    if tier == TIER_RAW and content.startswith(_USER_TEXT_PREFIX):
        user_text = content[len(_USER_TEXT_PREFIX):].split(_ASSISTANT_TEXT_SEP, 1)[0]
    return {
        "turn_index": r["turn_index"],
        "compression_tier": tier,
        "archived": tier != TIER_RAW,
        "branch": r["branch"],
        "user_text": user_text,
        "assistant_text": r["conclusion"] or "",
        "cited_source_ids": list(r["cited_source_ids"] or []),
        "rejected_suggestions": list(r["rejected_suggestions"] or []),
        "triggered_guardrails": list(r["triggered_guardrails"] or []),
        "created_at": r["created_at"],
    }


def load_all_messages(user_id: str = DEFAULT_USER_ID, *, dsn: str | None = None) -> list[dict[str, Any]]:
    """跨 session 拉这个用户的全部历史轮次，按发生时间顺序返回。

    背景：V1 单用户、单个中枢 agent(见 ARCHITECTURE.md)，`session_id` 只是
    压缩/归档算法的记账单位(§4.4.1 空闲折叠判定用它)，不是"多个独立对话"
    的用户概念——用户点"新对话"只是给以后的轮次开一个新的压缩记账窗口，
    不代表之前说过的话应该从历史里消失。所以前端加载历史时不该按当前
    `session_id` 过滤，应该看这个用户名下所有 session 的全部轮次。

    每条结果多带一个 `session_id` 字段(`load_session_messages` 没有，因为
    调用方已经知道 session_id 了)，方便以后前端要按会话分组展示时用；当前
    UI 不需要也可以直接忽略这个字段。跨多个 session 的 turn_index 不是全局
    唯一的，排序只能用 `created_at`，用它做主排序键，`turn_index` 仅在同一
    毫秒内落库(理论上不会发生，加上纯粹是防御)时兜底。"""
    conn = _connect(dsn)
    if conn is None:
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT m.session_id, m.turn_index, m.content, m.compression_tier, m.branch, "
            "m.conclusion, m.cited_source_ids, m.rejected_suggestions, m.triggered_guardrails, "
            "extract(epoch from m.created_at) AS created_at "
            "FROM messages m JOIN conversation_sessions cs ON cs.session_id = m.session_id "
            "WHERE cs.user_id = %s ORDER BY m.created_at ASC, m.turn_index ASC",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close()
    except Exception:
        logger.exception("load_all_messages failed · user=%s", user_id)
        return []
    finally:
        conn.close()

    result = []
    for r in rows:
        item = _message_row_to_dict(r)
        item["session_id"] = r["session_id"]
        result.append(item)
    return result
