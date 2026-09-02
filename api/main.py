#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 入口：`/api/chat`，SSE 流式输出；`/api/profile`、`/api/onboarding/*`。

设计依据：docs/ARCHITECTURE.md §10
决策依据：docs/DECISIONS.md D26
roadmap：阶段 4.2 任务 11

硬约束（BUILD_PLAN 阶段4 #11 完成判据）：**核查 pass 必须在第一条 `token` 事件
之前完成**——这里靠代码结构强制，不是靠约定：`backend/agents/dispatch.py` 里
每一条分支都是先把路由/派发/调和/核查全部跑完、拿到最终确定的文本，才第一次
`yield` 任何 SSE 事件；`verify()` 之后如果全部条目被拒绝，直接吐 `guardrail` +
`done`，从不吐 `token`。

⚠️ 诚实说明"流式"这两个字目前的真实程度：`backend/llm/adapter.py` 的
`complete()` 是一次性拿到完整响应，不是逐 token 流式返回——真正的模型级流式
需要 adapter/provider 两层都加 streaming 支持（OpenAI/Anthropic 的 stream=True
增量协议），是比这次任务大得多的另一块工作，不在阶段4任务11范围内。这里把
"整段已经算好的文本"切成若干块、依次 `yield` 多个 `token` 事件，实现的是
**SSE 传输层的多事件机制**（事件顺序、格式、trace_id 贯穿），不是模型生成过程
本身的流式——如实标注,不假装这是逐 token 生成。

⚠️ 首字节延迟（M10 <4s）本身依赖真实模型响应速度，不是这一层代码能保证的事，
这里能做到的只是"不在中间插入不必要的等待"——真正达标与否要用真实 eval 测。

2026-08-28：本文件原来同时装着 FastAPI 路由/DI wiring 和七个分支各自的完整
业务逻辑（1447 行）。后者和 FastAPI 完全无关，已按自然边界拆到：
  - `backend/agents/log_write.py`（`log_write` 分支：菜品拆解→过敏原即时警示→
    幂等写入）
  - `backend/agents/log_review.py`（`log_review` 分支：查 `query_diet_log`→
    确定性格式化）
  - `backend/agents/dispatch.py`（`fact_query`/`single_domain`/`candidate_eval`/
    `full_recommend`/`other` 五条分支 + 调和 + 过敏原重试 + 核查 pass + 多任务
    逐个分发的编排）
  - `backend/agents/sse.py`（`sse_event`/`chunk_text` 两个 SSE 格式化工具函数）
这里现在只剩：FastAPI app/DI wiring、输入防护/onboarding/追问恢复/路由分类
这条顶层链路（`_stream_chat_inner`）、以及 `/api/chat`、`/api/profile`、
`/api/onboarding/*` 三组路由声明。纯粹搬文件，不改变任何行为；对应测试也
按同样的边界重新组织（见 tests/unit/api/、tests/integration/test_api_chat_sse.py）。

分支覆盖：
  - log_review：不经过 SubAgent/LLM，直接查 query_diet_log、确定性格式化
  - log_write：§4.2 三级查找(dish_ingredient_map → user_dish_aliases → LLM 兜底)
    + 过敏原即时警示(不阻断写入,记录的是已经发生的事)+ write_memory(daily_log)
    幂等写入
  - fact_query / single_domain：派发 `decision.domain_hint` 指向的单个 SubAgent
  - candidate_eval / full_recommend：双派发 + `reconcile_subagent_results` +
    ENGINEERING §2 坑一(单边失败要能单边输出，`return_exceptions=True`) +
    坑二(45s SubAgent / 90s 整链 `wait_for`，取消传到还在跑的一侧) +
    坑三(请求级 token/`cost_est` 按两侧加总，不按墙钟取 max)

`user_profile`/`conflict_rules`(§5.2 步骤2/5)：log_review 之外的所有分支都会查
`user_profile`(log_write 要做过敏原比对，其余需要派发 SubAgent 的分支需要
constitution/画像)——`get_user_profile_fetcher()` 注入的是一个**函数**而不是
预先算好的值(同 `get_complete_fn` 的既有模式)，log_review 是唯一不会触发这次
数据库往返的分支。查不到画像/规则(表空、连不上库)一律静默降级为 `None`/`[]`，
不让请求失败——`constitution=None` 时 TCM SubAgent 走 D28 既有的"体质未知"
降级路径，这是正常路径的一部分，不是新加的错误处理。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.schemas import (
    ChatRequest,
    CreateUserRequest,
    CriticalFactDecisionRequest,
    OnboardingAnswerRequest,
    ProfileUpdateRequest,
)
from backend.agents.clarification import (
    ClarificationStore,
    InMemoryClarificationStore,
    default_clarification_store,
)
from backend.agents.conflict_rules_lookup import fetch_matched_conflict_rules
from backend.agents.dispatch import ConflictRulesFetcher, dispatch_branch, stream_multi_task
from backend.agents.routing import RouteBranch, RouteDecision, classify_turn
from backend.agents.sse import chunk_text, sse_event
from backend.agents.timeouts import aiter_with_timeout, chain_timeout_s
from backend.agents.user_context import (
    DEFAULT_USER_ID,
    UserProfileContext,
    create_user,
    ensure_user_profile,
    fetch_user_profile,
    list_users,
    persist_user_locale,
)
from backend.exceptions import ChainTimeoutError, DietExpertError
from backend.guardrails import ed_protection
from backend.guardrails.input_filters import detect_medical_intent, filter_input
from backend.i18n import normalize_locale, set_current_locale, t
from backend.llm import adapter as llm_adapter
from backend.llm.adapter import CompleteFn
from backend.logging_config import configure_logging
from backend.mcp_server.roles import CallerRole
from backend.mcp_server.safe_call import safe_call_tool
from backend.mcp_server.server import DietExpertMcpServer
from backend.mcp_server.tools._retrieval_common import warm_embedder, warm_embedder_enabled
from backend.memory import session_store
from backend.memory.compression import TurnRecord
from backend.memory.critical_fact_scanner import (
    CriticalFactScanResult,
    merge_into_profile,
    scan_critical_facts,
)
from backend.memory.pending_critical_facts import (
    InMemoryPendingCriticalFactStore,
    PendingCriticalFact,
    PendingCriticalFactStore,
    default_pending_store,
    new_pending_id,
)
from backend.observability.cost import record_llm_call, request_cost_scope
from backend.observability.redact import redact_text
from backend.observability.tracing import (
    current_trace_id,
    flush_tracing,
    observation,
    start_request_trace,
    stage_log,
    update_current,
)
from backend.onboarding.flow import (
    OnboardingResult,
    advance_chat_onboarding,
    apply_answer,
    should_trigger,
    start_onboarding,
)
from backend.onboarding.session_store import (
    InMemoryOnboardingSessionStore,
    OnboardingSessionStore,
    default_onboarding_store,
)

# 2026-08-30 补：真实多用户后，这两个都要能接受 `user_id=` 关键字参数覆盖
# 默认值(`fetch_user_profile`/`ensure_user_profile` 本来就支持，只是这里的
# 类型别名之前写成零参数——调用方现在会传 `user_id=request.user_id`)。
ProfileFetcher = Callable[..., "UserProfileContext | None"]
ProfileEnsurer = Callable[..., bool]
# backend/memory/session_store.py 接线(D27 补充，2026-08-28)——三个都注入成
# **函数**而不是直接在 `_stream_chat_inner` 里 import 模块调用，同
# `get_complete_fn`/`get_user_profile_fetcher` 的既有模式：测试用
# `app.dependency_overrides` 换成空操作的假函数，不会真的写真实 Postgres、
# 也不会在测试之间通过真实数据库表互相污染(不同测试文件反复复用同一个
# 硬编码 session_id，比如 "s1" —— 这条踩过 `_clarification_store_singleton`
# 的教训，见 tests/integration/test_api_chat_sse.py `_clear_overrides`)。
SessionHistoryLoader = Callable[[str], str]
# `user_id=` 关键字覆盖(`record_turn()` 本来就支持)，同上面 ProfileFetcher 的理由。
TurnRecorder = Callable[..., None]
IdleSessionFolder = Callable[[str], None]
# `GET /api/sessions/{session_id}/messages`(§10.1，之前一直没写)的注入点——
# 同上，注入 `session_store.load_session_messages()` 这个函数而不是预先查好
# 的列表，测试换成假函数。
SessionMessagesFetcher = Callable[[str], "list[dict[str, object]]"]
# `GET /api/messages`——2026-08-30 起真实多用户，按 `user_id` 过滤（不再是
# V1 单用户假设），见 `session_store.load_all_messages()` 文档。
AllMessagesFetcher = Callable[..., "list[dict[str, object]]"]
# `GET /api/users` / `POST /api/users`——用户切换器的数据源，见
# `backend/agents/user_context.py` `list_users()`/`create_user()`。
UserLister = Callable[[], "list[dict[str, str]]"]
UserCreator = Callable[[str], "dict[str, str] | None"]

_onboarding_store_singleton = InMemoryOnboardingSessionStore()
_clarification_store_singleton = InMemoryClarificationStore()
_pending_critical_store_singleton: PendingCriticalFactStore = InMemoryPendingCriticalFactStore()

# `_dispatch_and_record()` 落库用的 fire-and-forget 后台任务在这里注册强
# 引用——asyncio 文档明确警告：事件循环只持有 task 的弱引用，没有别处保留
# 强引用的 task 有可能在跑完之前就被垃圾回收，导致这一轮的落库+归档悄无
# 声息地根本没执行、且没有任何报错。`add_done_callback` 在任务结束(不管
# 成功/失败)后把它从这个集合里摘掉，避免集合无限增长。
_background_tasks: set[asyncio.Task] = set()


def _log_background_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("background task failed")


def _uses_in_memory_stores() -> bool:
    return (
        isinstance(_pending_critical_store_singleton, InMemoryPendingCriticalFactStore)
        or isinstance(_clarification_store_singleton, InMemoryClarificationStore)
        or isinstance(_onboarding_store_singleton, InMemoryOnboardingSessionStore)
    )

logger = logging.getLogger("diet_expert.api")


def _metered_complete(complete: CompleteFn) -> CompleteFn:
    """ENGINEERING §2 pit 3: every LLM call in this request adds to the
    request-scoped total. Injected test stubs go through the same wrap, so
    the chat span total is the sum even when adapter.complete is not used."""

    async def wrapped(messages, **kwargs):
        result = await complete(messages, **kwargs)
        record_llm_call(
            usage=getattr(result, "usage", None),
            cost_est=getattr(result, "cost_est", None),
        )
        return result

    return wrapped


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ENGINEERING §6.2: explicit init, not an import-time basicConfig side effect.
    configure_logging()
    global _pending_critical_store_singleton
    global _onboarding_store_singleton
    global _clarification_store_singleton
    _pending_critical_store_singleton = default_pending_store()
    _onboarding_store_singleton = default_onboarding_store()
    _clarification_store_singleton = default_clarification_store()
    if warm_embedder_enabled():
        try:
            logger.info("warming BGE-M3 embedder at startup")
            await asyncio.to_thread(warm_embedder)
        except Exception:
            logger.warning(
                "BGE-M3 embedder warmup failed; first retrieval may be slow",
                exc_info=True,
            )
    yield
    flush_tracing()


app = FastAPI(title="diet_expert", lifespan=lifespan)


def _error_payload(error_type: str) -> dict:
    body: dict = {"error": {"type": error_type}}
    trace_id = current_trace_id()
    if trace_id:
        body["error"]["trace_id"] = trace_id
    return body


@app.exception_handler(DietExpertError)
async def handle_known_error(_request: Request, exc: DietExpertError) -> JSONResponse:
    """Non-streaming JSON endpoints (`/api/profile`, `/api/onboarding/*`).

    `/api/chat` SSE already started sending — HTTP status cannot change.
    That path keeps the try/except → `guardrail` + `done` in `_stream_chat`.
    """
    logger.warning("known error · type=%s", type(exc).__name__)
    status = int(getattr(exc, "http_status", 400) or 400)
    error_type = getattr(exc, "error_type", None) or "diet_expert_error"
    return JSONResponse(_error_payload(error_type), status_code=status)


@app.exception_handler(Exception)
async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
    # FastAPI mounts Exception/500 handlers on ServerErrorMiddleware: the
    # JSON body is sent, then the exception is re-raised so the process can
    # log it. The client still sees this 500; TestClient needs
    # raise_server_exceptions=False to observe the body.
    logger.exception("unhandled")
    return JSONResponse(_error_payload("internal_error"), status_code=500)


# frontend/(Next.js dev server, 默认 :3000)和这个 API(默认 :8123)不同源，
# 浏览器的 fetch 会被 CORS 挡掉——V1 单机部署、前后端就服务同一个人，允许所有
# origin 不是安全隐患，允许列表本身才是维护负担；生产部署时如果前后端真的分开
# 部署到不同域名，应该收紧成显式白名单，不在这次最小闭环范围内处理。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_server_singleton = DietExpertMcpServer()


def get_mcp_server() -> DietExpertMcpServer:
    """FastAPI 依赖注入点——测试用 `app.dependency_overrides` 换成注入过 stub
    handler 的 server，不用真的连数据库/外部服务。"""
    return _server_singleton


def get_complete_fn() -> CompleteFn:
    """同上，测试换成 `_ScriptedComplete`，不打真实网络（同
    tests/unit/agents/test_agent_loop.py 的一贯模式）。"""
    return llm_adapter.complete


def get_user_profile_fetcher() -> ProfileFetcher:
    """返回**函数**而不是预先查好的画像（同 `get_complete_fn` 的模式）——这样
    log_write/log_review 这类不需要用户画像的分支不会触发多余的数据库往返：
    只有真的调用这个函数时才会查 `user_profile`。测试换成返回固定
    `UserProfileContext` 的假函数，不连真实数据库。"""
    return fetch_user_profile


def get_user_profile_ensurer() -> ProfileEnsurer:
    """Stamp a stub `user_profile` row the first time onboarding is offered.
    Tests replace this with a no-op so `/api/onboarding/start` / first chat
    does not hit the DB."""
    return ensure_user_profile


def get_onboarding_store() -> OnboardingSessionStore:
    """In-progress onboarding for `/api/chat`. Tests inject a fresh store."""
    return _onboarding_store_singleton


def get_clarification_store() -> ClarificationStore:
    """待补充问题的状态(D20 五处 agent 行为点第3条 + 扩展覆盖 SubAgent 四分支，
    见 backend/agents/clarification.py 模块文档)。Tests inject a fresh store，
    同 `get_onboarding_store` 的既有模式。"""
    return _clarification_store_singleton


def get_pending_critical_store() -> PendingCriticalFactStore:
    """Scanner hits wait here until the user confirms (PRD §10.2 / D34)."""
    return _pending_critical_store_singleton


def get_conflict_rules_fetcher() -> ConflictRulesFetcher:
    """同上，注入函数而不是预先查好的规则列表。"""
    return fetch_matched_conflict_rules


def get_session_history_loader() -> SessionHistoryLoader:
    """D27 补充(2026-08-28)：`backend/memory/session_store.py` `load_session_history()`
    的注入点。查不到/连不上库时该函数本身已经静默降级为空字符串，这里不用
    再包一层 try/except。"""
    return session_store.load_session_history


def get_turn_recorder() -> TurnRecorder:
    """同上，`record_turn()` 的注入点——`_stream_chat_inner` 在响应发出之后
    用 `asyncio.to_thread()` 调用这个函数，不阻塞当前请求(见该函数调用处的
    说明)。"""
    return session_store.record_turn


def get_idle_session_folder() -> IdleSessionFolder:
    """同上，`maybe_fold_idle_session()` 的注入点。"""
    return session_store.maybe_fold_idle_session


def get_session_messages_fetcher() -> SessionMessagesFetcher:
    """`GET /api/sessions/{session_id}/messages` 的注入点——
    `session_store.load_session_messages()` 查不到/连不上库时已经静默降级
    为空列表，同 `get_session_history_loader` 的既有模式。"""
    return session_store.load_session_messages


def get_all_messages_fetcher() -> AllMessagesFetcher:
    """`GET /api/messages` 的注入点——`session_store.load_all_messages()`
    同样静默降级为空列表，同上。"""
    return session_store.load_all_messages


def get_user_lister() -> UserLister:
    """`GET /api/users` 的注入点——`user_context.list_users()`。"""
    return list_users


def get_user_creator() -> UserCreator:
    """`POST /api/users` 的注入点——`user_context.create_user()`。"""
    return create_user


@app.get("/healthz")
async def healthz():
    """docker-compose 健康检查（ENGINEERING §9）。

    默认只做进程存活（单测 / 本地 uvicorn 不连库）。compose 给 API 设
    `HEALTHZ_CHECK_DB=1` 之后才查 Postgres：能连上、pgvector 扩展在、
    `knowledge_chunks` 的 HNSW 索引在。表空不算不健康——ingest 按 §9
    明确不放在启动路径上，冷启动 90s 判据是 healthz 200，不是知识库灌完。
    """
    if os.environ.get("HEALTHZ_CHECK_DB") != "1":
        payload: dict[str, object] = {"status": "ok"}
        if _uses_in_memory_stores():
            payload["degraded"] = True
        return payload

    dsn = os.environ.get("DIET_EXPERT_PG_DSN")
    if not dsn:
        return JSONResponse(
            {"status": "unhealthy", "detail": "DIET_EXPERT_PG_DSN missing"},
            status_code=503,
        )
    try:
        import psycopg2

        conn = psycopg2.connect(dsn, connect_timeout=3)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone() is None:
                return JSONResponse(
                    {"status": "unhealthy", "detail": "pgvector extension missing"},
                    status_code=503,
                )
            cur.execute(
                """
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = 'knowledge_chunks'
                  AND indexname = 'idx_knowledge_chunks_embedding_hnsw'
                """
            )
            if cur.fetchone() is None:
                return JSONResponse(
                    {"status": "unhealthy", "detail": "vector index not ready"},
                    status_code=503,
                )
        finally:
            conn.close()
    except Exception:
        # Do not echo the exception: DSN may contain credentials.
        return JSONResponse(
            {"status": "unhealthy", "detail": "db unreachable"},
            status_code=503,
        )
    return {"status": "ok", **({"degraded": True} if _uses_in_memory_stores() else {})}


# Medical-intent disclaimer copy lives in backend.i18n (`api.medical_disclaimer`).


# ---------------------------------------------------------------------------
# D27 补充(2026-08-28)：把这一轮"发生了什么"重建成 backend/memory/
# compression.py 的 `TurnRecord`，喂给 session_store.record_turn()。
#
# 不改 backend/agents/dispatch.py 任何函数签名——`dispatch_branch()`/
# `stream_multi_task()` 已经把"这一轮发生了什么"完整编码进标准化的 SSE 事件
# 协议里(token=结论文本、source=引用、guardrail=触发的guardrail/被拒建议)，
# 直接在这一层观察已经产出的事件流重建结构化记录，比改 dispatch.py 内部一路
# 传一个"累加器"对象下去更不侵入、和那个模块的既有职责边界(它只管产出 SSE
# 事件，不管这些事件之后要被怎么消费)更一致。
# ---------------------------------------------------------------------------

_SSE_CHUNK_RE = re.compile(r"^event:\s*(\S+)\ndata:\s*(.*)\n\n$", re.DOTALL)


def _parse_sse_chunk(chunk: str) -> tuple[str | None, dict]:
    m = _SSE_CHUNK_RE.match(chunk)
    if not m:
        return None, {}
    try:
        data = json.loads(m.group(2)) if m.group(2) else {}
    except json.JSONDecodeError:
        data = {}
    return m.group(1), data


class _TurnAccumulator:
    """观察 `dispatch_branch()`/`stream_multi_task()` 吐出的原始 SSE 字符串流，
    重建这一轮的 `TurnRecord`。多任务(D32)场景下会看到多组 `task`/`token`/
    `source`/`guardrail` 事件——这里不按子任务拆分成多条记录(`messages` 表
    按 `(session_id, turn_index)` 唯一，这个粒度对应"一条用户消息"，见
    `session_store.py` 模块文档"已知限制"一节)，而是合并成一条：`branch`
    字段用命中过的所有分支名拼接。"""

    def __init__(self) -> None:
        self._token_chunks: list[str] = []
        self._source_ids: list[str] = []
        self._rejected: list[str] = []
        self._guardrails: list[str] = []
        self._branches: list[str] = []

    def observe(self, sse_chunk: str) -> None:
        event, data = _parse_sse_chunk(sse_chunk)
        if event == "task":
            branch = data.get("branch")
            if branch and branch not in self._branches:
                self._branches.append(branch)
        elif event == "token":
            self._token_chunks.append(data.get("text", ""))
        elif event == "source":
            source_id = data.get("source_id")
            if source_id and source_id not in self._source_ids:
                self._source_ids.append(source_id)
        elif event == "guardrail":
            guardrail_type = data.get("type")
            if guardrail_type and guardrail_type not in self._guardrails:
                self._guardrails.append(guardrail_type)
            if guardrail_type == "rejected_item":
                reason = data.get("reason")
                if reason:
                    self._rejected.append(reason)

    def build(self, *, branch_fallback: str, user_text: str) -> TurnRecord:
        conclusion = "".join(self._token_chunks)
        branch = "+".join(self._branches) if self._branches else branch_fallback
        return TurnRecord(
            turn_id="",  # session_store.record_turn() 自己分配真正的 turn_index
            branch=branch,
            raw_text=f"用户: {user_text}\n助手: {conclusion}",
            conclusion=conclusion,
            cited_source_ids=tuple(self._source_ids),
            rejected_suggestions=tuple(self._rejected),
            triggered_guardrails=tuple(self._guardrails),
        )


async def _dispatch_and_record(
    inner_stream: AsyncIterator[str],
    *,
    session_id: str,
    branch_fallback: str,
    user_text: str,
    turn_recorder: TurnRecorder,
    user_id: str = DEFAULT_USER_ID,
) -> AsyncIterator[str]:
    """包一层 `dispatch_branch()`/`stream_multi_task()` 的输出——原样转发
    每个 SSE chunk(调用方看不出区别)，全部转发完之后用
    `asyncio.create_task(asyncio.to_thread(...))` 触发落库+归档检查，不
    `await`——这是 D27 修订二"步骤9异步、不占用当前请求响应时间"在这个项目
    (没有 Celery 这类真正的后台任务队列)里的实际实现方式：生成器已经把所有
    chunk 交给 ASGI 层，`record_turn()` 是一次同步的 psycopg2 调用，用
    `to_thread` 避免阻塞事件循环，`create_task` 避免等它跑完才让这次请求
    结束。压缩/归档失败(见 `record_turn()` 内部的静默降级)不会影响这次已经
    发出的响应。task 强引用存进 `_background_tasks`(见该模块级变量的注释)，
    避免 GC 提前回收导致落库悄悄没发生。"""
    accumulator = _TurnAccumulator()
    async for chunk in inner_stream:
        accumulator.observe(chunk)
        yield chunk
    turn = accumulator.build(branch_fallback=branch_fallback, user_text=user_text)
    task = asyncio.create_task(
        asyncio.to_thread(turn_recorder, session_id, turn, user_id=user_id)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_log_background_task_result)


async def _stream_chat(
    request: ChatRequest,
    server: DietExpertMcpServer,
    complete: CompleteFn,
    profile_fetcher: ProfileFetcher,
    conflict_rules_fetcher: ConflictRulesFetcher,
    trace_id: str,
    profile_ensurer: ProfileEnsurer,
    onboarding_store: OnboardingSessionStore,
    clarification_store: ClarificationStore,
    session_history_loader: SessionHistoryLoader,
    turn_recorder: TurnRecorder,
    idle_session_folder: IdleSessionFolder,
    pending_critical_store: PendingCriticalFactStore,
) -> AsyncIterator[str]:
    outcome: dict = {"status": "ok"}
    with start_request_trace(
        trace_id,
        name="chat",
        session_id=request.session_id,
        user_id=request.user_id,
        input={"message": redact_text(request.message), "session_id": request.session_id},
    ) as root:
        with request_cost_scope() as request_cost:
            metered = _metered_complete(complete)
            try:
                async for chunk in aiter_with_timeout(
                    _stream_chat_inner(
                        request,
                        server,
                        metered,
                        profile_fetcher,
                        conflict_rules_fetcher,
                        trace_id,
                        outcome,
                        profile_ensurer,
                        onboarding_store,
                        clarification_store,
                        session_history_loader,
                        turn_recorder,
                        idle_session_folder,
                        pending_critical_store,
                    ),
                    timeout=chain_timeout_s(),
                ):
                    yield chunk
            except ChainTimeoutError:
                # wait_for already cancelled the generator, which cancelled
                # leftover SubAgent tasks. Tell the HTTP client via SSE —
                # the stream has started, so we cannot change the status code.
                outcome["status"] = "chain_timeout"
                logger.warning("chat chain timed out · trace_id=%s", trace_id)
                yield sse_event(
                    "guardrail",
                    {"type": "chain_timeout", "detail": t("api.chain_timeout", request.locale)},
                )
                yield sse_event("done", {"trace_id": trace_id})
            except asyncio.CancelledError:
                outcome["status"] = "cancelled"
                raise
            except Exception:
                outcome["status"] = "error"
                logger.exception("chat pipeline failed · trace_id=%s", trace_id)
                yield sse_event("guardrail", {"type": "internal_error", "detail": t("api.internal_error", request.locale)})
                yield sse_event("done", {"trace_id": trace_id})
            finally:
                outcome["tokens"] = request_cost.total_tokens
                outcome["cost_est"] = request_cost.cost_est
                outcome["llm_calls"] = request_cost.calls
                outcome["cost_incomplete"] = request_cost.cost_incomplete
                stage_log(
                    logger,
                    "chat",
                    tokens=request_cost.total_tokens,
                    cost_est=request_cost.cost_est,
                    llm_calls=request_cost.calls,
                    cost_incomplete=request_cost.cost_incomplete,
                    status=outcome.get("status"),
                )
                update_current(output=outcome)
                root.update(output=outcome)


async def _stream_chat_inner(
    request: ChatRequest,
    server: DietExpertMcpServer,
    complete: CompleteFn,
    profile_fetcher: ProfileFetcher,
    conflict_rules_fetcher: ConflictRulesFetcher,
    trace_id: str,
    outcome: dict,
    profile_ensurer: ProfileEnsurer,
    onboarding_store: OnboardingSessionStore,
    clarification_store: ClarificationStore,
    session_history_loader: SessionHistoryLoader,
    turn_recorder: TurnRecorder,
    idle_session_folder: IdleSessionFolder,
    pending_critical_store: PendingCriticalFactStore,
) -> AsyncIterator[str]:
    set_current_locale(request.locale)

    # 总览图①：输入防护——截断超长输入 + 剥离指令注入片段(PRD §10;
    # THREAT_MODEL.md E1)，在路由判断之前跑，剥离后的文本才进
    # classify_route_async 和后续所有分支。之前这一步完全不存在——用户输入
    # 未经任何处理直接进模型上下文，是这次(阶段5)要补的缺口。
    filtered = filter_input(request.message)
    if filtered.instruction_injection_flagged or filtered.was_truncated:
        with observation(
            "input.guardrail",
            as_type="guardrail",
            metadata={
                "instruction_injection": filtered.instruction_injection_flagged,
                "truncated": filtered.was_truncated,
            },
        ):
            if filtered.instruction_injection_flagged:
                logger.warning(
                    "input guardrail: instruction injection stripped · trace_id=%s · spans=%s",
                    trace_id, filtered.instruction_injection_spans,
                )
            if filtered.was_truncated:
                logger.warning(
                    "input guardrail: message truncated %d -> %d chars · trace_id=%s",
                    filtered.original_length, len(filtered.text), trace_id,
                )
            update_current(
                output={
                    "spans": filtered.instruction_injection_spans,
                    "original_length": filtered.original_length,
                }
            )
    request = request.model_copy(update={"message": filtered.text})

    # ED 防护规则 2/3/4(极端限制性表述 / 索要数值目标 / 自述极低摄入或体重
    # 焦虑)——在剥离后的文本上跑。命中就直接吐审阅过的模板回复(见
    # backend/guardrails/ed_protection.py 的 `canned_response`，来源
    # docs/prompts/disclaimers.md §9/§10)，不继续走路由/生成
    # (THREAT_MODEL.md E3)。`scan_user_input` 已经把优先级排好：体重焦虑
    # 高于索要数值目标，高于极端限制性表述。
    ed_result = ed_protection.scan_user_input(request.message)
    if ed_result.blocked:
        primary = ed_result.primary
        outcome["status"] = "ed_blocked"
        outcome["rule"] = primary.rule.value
        logger.warning(
            "input guardrail: ED rule triggered · trace_id=%s · rule=%s · matched=%r",
            trace_id, primary.rule.value, primary.matched,
        )
        with observation(
            "input.ed_protection",
            as_type="guardrail",
            metadata={"rule": primary.rule.value},
            level="WARNING",
        ):
            update_current(output={"rule": primary.rule.value, "reason": primary.reason})
        yield sse_event(
            "guardrail",
            {"type": "ed_protection", "rule": primary.rule.value, "detail": primary.reason},
        )
        for chunk in chunk_text(ed_result.canned_response_for_locale(request.locale) or ""):
            yield sse_event("token", {"text": chunk})
        yield sse_event("done", {"trace_id": trace_id})
        return

    # 疾病/用药咨询意图——切到受限模式：仅通用信息 + 免责声明，不派发
    # SubAgent 生成个性化建议(PRD §10；THREAT_MODEL.md E8)。
    if detect_medical_intent(request.message):
        outcome["status"] = "medical_restricted"
        logger.warning(
            "input guardrail: medical intent detected, restricted mode · trace_id=%s", trace_id
        )
        with observation("input.medical_intent", as_type="guardrail", level="WARNING"):
            update_current(output={"mode": "restricted"})
        yield sse_event(
            "guardrail",
            {
                "type": "medical_intent_restricted",
                "detail": t("api.medical_intent_detail", request.locale),
            },
        )
        for chunk in chunk_text(t("api.medical_disclaimer", request.locale)):
            yield sse_event("token", {"text": chunk})
        yield sse_event("done", {"trace_id": trace_id})
        return

    # §11.1: first conversation with no user_profile — the agent itself asks
    # the onboarding questions over `/api/chat`. Do not wait for the frontend
    # to call `/api/onboarding/*`. Safety filters above still run first.
    profile = profile_fetcher(user_id=request.user_id)
    if profile is not None:
        try:
            persist_user_locale(request.user_id, request.locale)
        except Exception:
            logger.warning("persist locale failed · user_id=%s", request.user_id, exc_info=True)
    onboarding_turn = advance_chat_onboarding(
        request.message, profile, onboarding_store, user_id=request.user_id, locale=request.locale
    )
    if onboarding_turn is not None:
        outcome["status"] = "onboarding"
        outcome["branch"] = "onboarding"
        if onboarding_turn.started:
            profile_ensurer(user_id=request.user_id)
            try:
                persist_user_locale(request.user_id, request.locale)
            except Exception:
                logger.warning("persist locale failed · user_id=%s", request.user_id, exc_info=True)
        if onboarding_turn.done and onboarding_turn.profile_updates is not None:
            session = server.open_session(CallerRole.ROUTER, user_id=request.user_id)
            write_result = safe_call_tool(
                session,
                "write_memory",
                {"category": "critical", "payload": onboarding_turn.profile_updates},
            )
            if not write_result.ok:
                logger.warning(
                    "onboarding write_memory failed · trace_id=%s · error=%s",
                    trace_id,
                    write_result.detail,
                )
        logger.info(
            "chat onboarding · trace_id=%s · started=%s · done=%s",
            trace_id, onboarding_turn.started, onboarding_turn.done,
        )
        for chunk in chunk_text(onboarding_turn.prompt):
            yield sse_event("token", {"text": chunk})
        yield sse_event("done", {"trace_id": trace_id})
        return

    # D27 补充(2026-08-28)：新消息到达是这个项目里"会话是不是空闲了"唯一
    # 天然的检查时机(没有后台调度器)，在真正处理这条消息之前先跑一次折叠
    # 检查(把上一次判定为"还在进行中"的 Tier2 摘要，如果距上次活跃已经超过
    # 空闲阈值，折叠成 Tier3)，再组装喂给这一轮 SubAgent 的会话历史文本——
    # 两者都静默降级(查不到/连不上库不影响这次请求正常处理)，见
    # backend/memory/session_store.py 模块文档。
    idle_session_folder(request.session_id)
    session_history = session_history_loader(request.session_id)

    # §4.3 / D34: deterministic scan runs every turn; pending emit may be deferred
    # to profile_write when that branch owns LLM merge (see below after classify_turn).
    fact_scan = scan_critical_facts(request.message, profile)

    # D20 五处 agent 行为点第3条(2026-08-27 实现)：上一轮问过用户一个澄清
    # 问题，这一轮消息是回答——不重新走路由，直接把补充信息拼回原文本，
    # 重新分发到当初触发追问的那个分支。`clear()` 在重新分发之前就调用，
    # 天然保证"追问一次，仍模糊则记为 unspecified"(PRD §11)这条单次上限：
    # 不管这次分发结果如何，pending 状态都已经被消费掉，不会再问第三次。
    pending = clarification_store.get(request.session_id)
    if pending is not None:
        clarification_store.clear(request.session_id)
        combined_message = f"{pending.original_text}，{request.message}"
        logger.info(
            "clarification retry · trace_id=%s · session=%s · branch=%s",
            trace_id, request.session_id, pending.branch.value,
        )
        outcome["status"] = "clarification_retry"
        outcome["branch"] = pending.branch.value
        retry_decision = RouteDecision(
            pending.branch,
            reason="clarification_retry",
            domain_hint=pending.domain_hint,
            rule_matched=True,
        )
        # 2026-09-01：追问重试轮的分支其实早就定了(上一轮触发追问时)，这里
        # 仍然吐一条 routing stage 事件——前端的"处理阶段"指示条不该因为走的
        # 是重试路径就少一截，用户体感上这仍然是"这一轮请求"的第一步。
        yield sse_event(
            "stage",
            {
                "stage": "routing",
                "status": "done",
                "detail": t("dispatch.stage_routing", request.locale),
                "branch": pending.branch.value,
            },
        )
        async for chunk in _dispatch_and_record(
            dispatch_branch(
                request.model_copy(update={"message": combined_message}),
                retry_decision, server, complete, trace_id, profile, conflict_rules_fetcher,
                clarification_store, allow_clarification=False, session_history=session_history,
                pending_critical_store=pending_critical_store,
            ),
            session_id=request.session_id,
            branch_fallback=pending.branch.value,
            user_text=combined_message,
            turn_recorder=turn_recorder,
            user_id=request.user_id,
        ):
            yield chunk
        return

    # D32/§5.1.1(2026-08-27 补充 LLM 兜底)：一句话包含多个独立意图(比如"帮我
    # 记录一下中午吃了麻婆豆腐，另外阳虚质应该吃什么")时，六分支路由此前只会
    # 命中优先级最高的那一个，整条原文喂给那一个分支，其余意图直接消失、用户
    # 毫无感知。`classify_turn()` 一次性决定这一轮该拆成几个任务：优先用确定性
    # 连接词切分(零 LLM 调用)，只有规则确实拿不准(完全没命中，或者有连接词
    # 但切分/分类不confident)时才打一次 LLM 去判断——比只做纯规则多覆盖了
    # "连接词覆盖不到的说法"和"压根没用连接词的隐式多意图"这两类，见
    # DECISIONS.md D32 补充说明。返回值恒为非空元组，长度为 1 就是原来的
    # 单任务场景，`dispatch_branch` 走法和这条设计生效前完全一样。
    tasks = await classify_turn(request.message, complete=complete)
    runs_profile_write = any(t.decision.branch is RouteBranch.PROFILE_WRITE for t in tasks)
    if fact_scan.hit and not runs_profile_write:
        pending = PendingCriticalFact(
            pending_id=new_pending_id(),
            user_id=request.user_id,
            session_id=request.session_id,
            allergens=fact_scan.new_allergens,
            supplements=fact_scan.new_supplements,
        )
        try:
            pending_critical_store.put(pending)
        except Exception:
            logger.exception("failed to persist pending critical fact · trace_id=%s", trace_id)
            yield sse_event(
                "guardrail",
                {
                    "type": "pending_critical_store_failed",
                    "detail": t("api.pending_critical_store_failed", request.locale),
                },
            )
        else:
            logger.info(
                "critical fact pending · trace_id=%s · pending_id=%s · allergens=%s · supplements=%s",
                trace_id, pending.pending_id, fact_scan.new_allergens, fact_scan.new_supplements,
            )
            yield sse_event("critical_fact_pending", pending.to_event_dict(locale=request.locale))

    outcome["branch"] = tasks[0].decision.branch.value if len(tasks) == 1 else None
    outcome["multi_task"] = len(tasks) > 1
    outcome["branches"] = [t.decision.branch.value for t in tasks]
    logger.info(
        "chat · trace_id=%s · session=%s · tasks=%d · branches=%s",
        trace_id, request.session_id, len(tasks), outcome["branches"],
    )
    stage_log(logger, "chat", task_count=len(tasks), branches=outcome["branches"], session_id=request.session_id)

    # 2026-09-01：路由已确定——这是"过程可见性"这条新增能力(见
    # backend/agents/dispatch.py `_stage_event()`)里唯一一个不属于
    # `dispatch_branch()`/`stream_multi_task()` 内部、必须在这里吐的事件：
    # 路由判断本身就是在这两个函数被调用*之前*做完的。多任务场景下
    # `branch` 字段复用下面 `branch_fallback` 同款的 "+".join(...) 记法，
    # 单任务原样给分支名，不为了这一条 stage 事件专门发明第二套多分支表示法。
    yield sse_event(
        "stage",
        {
            "stage": "routing",
            "status": "done",
            "detail": t("dispatch.stage_routing", request.locale),
            "branch": outcome["branch"] if outcome["branch"] is not None else "+".join(outcome["branches"]),
        },
    )

    # Profile already loaded above for the onboarding check. log_review is the
    # only branch that does not use it.
    logger.info(
        "user_profile fetched · trace_id=%s · has_profile=%s · constitution=%s",
        trace_id, profile is not None, profile.constitution if profile else None,
    )
    update_current(
        metadata={
            "has_profile": profile is not None,
            "constitution": profile.constitution if profile else None,
        }
    )

    if len(tasks) > 1:
        async for chunk in _dispatch_and_record(
            stream_multi_task(
                request, tasks, server, complete, profile, conflict_rules_fetcher, trace_id,
                clarification_store, session_history=session_history,
                pending_critical_store=pending_critical_store,
                prefetched_fact_scan=fact_scan if runs_profile_write else None,
            ),
            session_id=request.session_id,
            branch_fallback="+".join(outcome["branches"]),
            user_text=request.message,
            turn_recorder=turn_recorder,
            user_id=request.user_id,
        ):
            yield chunk
        return

    async for chunk in _dispatch_and_record(
        dispatch_branch(
            request, tasks[0].decision, server, complete, trace_id, profile, conflict_rules_fetcher,
            clarification_store, session_history=session_history,
            pending_critical_store=pending_critical_store,
            prefetched_fact_scan=fact_scan
            if tasks[0].decision.branch is RouteBranch.PROFILE_WRITE
            else None,
        ),
        session_id=request.session_id,
        branch_fallback=tasks[0].decision.branch.value,
        user_text=request.message,
        turn_recorder=turn_recorder,
        user_id=request.user_id,
    ):
        yield chunk


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    server: DietExpertMcpServer = Depends(get_mcp_server),
    complete: CompleteFn = Depends(get_complete_fn),
    profile_fetcher: ProfileFetcher = Depends(get_user_profile_fetcher),
    conflict_rules_fetcher: ConflictRulesFetcher = Depends(get_conflict_rules_fetcher),
    profile_ensurer: ProfileEnsurer = Depends(get_user_profile_ensurer),
    onboarding_store: OnboardingSessionStore = Depends(get_onboarding_store),
    clarification_store: ClarificationStore = Depends(get_clarification_store),
    session_history_loader: SessionHistoryLoader = Depends(get_session_history_loader),
    turn_recorder: TurnRecorder = Depends(get_turn_recorder),
    idle_session_folder: IdleSessionFolder = Depends(get_idle_session_folder),
    pending_critical_store: PendingCriticalFactStore = Depends(get_pending_critical_store),
) -> StreamingResponse:
    # ENGINEERING §6.1: trace_id on the HTTP response header AND in the SSE
    # `done` event. Generated here so the header is available before the
    # generator starts yielding.
    trace_id = uuid.uuid4().hex
    return StreamingResponse(
        _stream_chat(
            request,
            server,
            complete,
            profile_fetcher,
            conflict_rules_fetcher,
            trace_id,
            profile_ensurer,
            onboarding_store,
            clarification_store,
            session_history_loader,
            turn_recorder,
            idle_session_folder,
            pending_critical_store,
        ),
        media_type="text/event-stream",
        headers={"X-Trace-Id": trace_id},
    )


# ---------------------------------------------------------------------------
# §10.1 剩余三条路由：`GET /api/sessions/{session_id}/messages`、`/api/profile`、
# `/api/onboarding/*`。都不经过 SubAgent/LLM——会话历史/画像读写是确定性 CRUD，
# 引导对话是 backend/onboarding/flow.py 的确定性步骤机(CCMQ 是固定选择题，
# 不需要生成式理解自由文本)，没有理由为这三条路由引入不必要的模型调用。
# ---------------------------------------------------------------------------


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    messages_fetcher: SessionMessagesFetcher = Depends(get_session_messages_fetcher),
):
    """§10.1：前端刷新页面/重新打开后拉取这个 session 之前的对话轮次，重建
    聊天气泡。已归档(Tier2/Tier3)的轮次原始用户提问已经不在库里了(D27 归档
    设计如此)，对应行 `user_text` 为 `null`、`archived` 为 `true`——前端不该
    为这类轮次伪造一条用户气泡，用一条摘要提示展示即可，见
    `backend/memory/session_store.py` `load_session_messages()` 的文档。"""
    return {"messages": messages_fetcher(session_id)}


@app.get("/api/messages")
async def get_all_messages(
    user_id: str = DEFAULT_USER_ID,
    messages_fetcher: AllMessagesFetcher = Depends(get_all_messages_fetcher),
):
    """跨 session 拉这个用户的全部历史轮次，按发生时间顺序返回。`session_id`
    只是压缩/归档的记账单位，不是"多个独立对话"——前端切换用户/刷新页面
    重建聊天气泡应该用这条而不是按当前 session_id 过滤的
    `GET /api/sessions/{session_id}/messages`，见
    `backend/memory/session_store.py` `load_all_messages()` 的文档。"""
    return {"messages": messages_fetcher(user_id=user_id)}


@app.get("/api/users")
async def get_users(user_lister: UserLister = Depends(get_user_lister)):
    """前端用户切换器的数据源——现在库里有几行 `user_profile` 就是几个"用户"。
    见 `backend/agents/user_context.py` `list_users()` 的文档。"""
    return {"users": user_lister()}


@app.post("/api/users")
async def create_new_user(
    request: CreateUserRequest, user_creator: UserCreator = Depends(get_user_creator)
):
    """新建一个用户——纯展示名字，不是登录凭证/没有密码，本来就没有认证层
    (V1 单机个人使用)，这里只是让"同一台机器给不同的人/不同的画像分别记录"
    这件事有一个真实的 `user_id` 可用，见 `create_user()` 的文档。"""
    created = user_creator(request.name)
    if created is None:
        return JSONResponse({"detail": t("api.create_user_failed")}, status_code=500)
    return created


_PROFILE_PATCHABLE_FIELDS = frozenset(
    {
        "constitution", "constitution_secondary", "constitution_source",
        "allergens", "supplements", "goal_tags", "preferences", "city", "timezone",
        "locale",
    }
)


@app.get("/api/profile")
async def get_profile(
    user_id: str = DEFAULT_USER_ID,
    profile_fetcher: ProfileFetcher = Depends(get_user_profile_fetcher),
):
    """§10.1：`GET /api/profile` 读取指定用户(`user_id` query param，默认
    `default_user` 向后兼容)的 `user_profile`。额外带一个
    `onboarding_recommended`字段(不在 §10.1 原始响应形状里)，把 §11.1 的触发
    条件判断结果暴露给前端——`onboarding_done` 为 false（含新建用户 stub、
    体质全空）时为 true。引导走完或「全部跳过」后写入 `onboarding_done=true`，
    不再自动建议。`/api/chat` 在同样的条件下会由中枢自己插入引导对话。"""
    profile = profile_fetcher(user_id=user_id)
    if profile is None:
        return {"exists": False, "onboarding_recommended": should_trigger(None)}
    return {
        "exists": True,
        "onboarding_recommended": should_trigger(profile),
        "constitution": profile.constitution,
        "constitution_secondary": list(profile.constitution_secondary),
        "constitution_source": profile.constitution_source,
        "allergens": list(profile.allergens),
        "supplements": list(profile.supplements),
        "goal_tags": list(profile.goal_tags),
        "preferences": profile.preferences,
        "city": profile.city,
        "timezone": profile.timezone,
        "onboarding_done": profile.onboarding_done,
        "locale": profile.locale,
    }


@app.patch("/api/profile")
async def patch_profile(
    request: ProfileUpdateRequest, server: DietExpertMcpServer = Depends(get_mcp_server)
):
    """§10.1：`PATCH /api/profile`，承载人在环确认后的写入(PRD §10.2)。
    `confirmed` 必须显式为 true——见 api/schemas.py `ProfileUpdateRequest` 的说明。"""
    if not request.confirmed:
        return JSONResponse(
            {"detail": t("api.profile_unconfirmed")}, status_code=400
        )
    if request.field not in _PROFILE_PATCHABLE_FIELDS:
        return JSONResponse(
            {
                "detail": t(
                    "api.profile_field_unsupported",
                    fields=", ".join(sorted(_PROFILE_PATCHABLE_FIELDS)),
                    field=request.field,
                )
            },
            status_code=400,
        )
    value = (
        normalize_locale(request.value)
        if request.field == "locale" and isinstance(request.value, str)
        else request.value
    )
    session = server.open_session(CallerRole.ROUTER, user_id=request.user_id)
    write_result = safe_call_tool(
        session,
        "write_memory",
        {"category": "critical", "payload": {request.field: value}},
    )
    if not write_result.ok:
        return JSONResponse(
            {"detail": t("api.tool_call_failed")},
            status_code=503,
        )
    result = write_result.result
    return {"ok": result.ok, "fields_written": list(result.fields_written)}


@app.post("/api/profile/critical-facts/confirm")
async def confirm_critical_fact(
    request: CriticalFactDecisionRequest,
    server: DietExpertMcpServer = Depends(get_mcp_server),
    profile_fetcher: ProfileFetcher = Depends(get_user_profile_fetcher),
    pending_critical_store: PendingCriticalFactStore = Depends(get_pending_critical_store),
):
    """PRD §10.2: only after the user confirms do we UPSERT into user_profile.

    `user_id` 来自 `pending.user_id`(这条 fact 被扫描出来时就已经记下是哪个
    用户说的，见 `_stream_chat_inner` 里 `PendingCriticalFact(user_id=request.user_id, ...)`
    的构造)，不是从这个请求本身取——`CriticalFactDecisionRequest` 只有
    `pending_id`，这样更简单也更对：confirm 操作本来就该落到"当初触发它的
    那个用户"身上，不需要客户端在确认时重复声明一遍。"""
    pending = pending_critical_store.get(request.pending_id)
    if pending is None:
        return JSONResponse({"detail": t("api.pending_not_found")}, status_code=404)
    scan = CriticalFactScanResult(
        new_allergens=pending.allergens,
        new_supplements=pending.supplements,
        new_preferences=pending.preferences,
    )
    payload, _updated = merge_into_profile(
        scan, profile_fetcher(user_id=pending.user_id), user_id=pending.user_id
    )
    session = server.open_session(CallerRole.ROUTER, user_id=pending.user_id)
    write_result = safe_call_tool(
        session,
        "write_memory",
        {"category": "critical", "payload": payload},
    )
    if not write_result.ok:
        return JSONResponse(
            {"detail": t("api.tool_call_failed")},
            status_code=503,
        )
    result = write_result.result
    pending_critical_store.delete(request.pending_id)
    return {
        "ok": result.ok,
        "fields_written": list(result.fields_written),
        "pending_id": request.pending_id,
    }


@app.post("/api/profile/critical-facts/revoke")
async def revoke_critical_fact(
    request: CriticalFactDecisionRequest,
    pending_critical_store: PendingCriticalFactStore = Depends(get_pending_critical_store),
):
    deleted = pending_critical_store.delete(request.pending_id)
    if deleted is None:
        return JSONResponse({"detail": t("api.pending_not_found")}, status_code=404)
    return {"ok": True, "pending_id": request.pending_id, "revoked": True}


@app.post("/api/onboarding/start")
async def onboarding_start(
    user_id: str = DEFAULT_USER_ID,
    locale: str = "zh",
    profile_ensurer: ProfileEnsurer = Depends(get_user_profile_ensurer),
):
    """§10.1：`POST /api/onboarding/start`，触发首次使用引导(§11)。

    Inserts a stub `user_profile` row immediately so a user with no row yet
    still has one after this call. Chat onboarding keys off `onboarding_done`
    (FALSE until the intro finishes, including skip-all).
    """
    locale = normalize_locale(locale)
    profile_ensurer(user_id=user_id)
    persist_user_locale(user_id, locale)
    step = start_onboarding(locale=locale)
    return {"step_id": step.step_id, "prompt": step.prompt, "state": step.state}


@app.post("/api/onboarding/answer")
async def onboarding_answer(
    request: OnboardingAnswerRequest, server: DietExpertMcpServer = Depends(get_mcp_server)
):
    """§10.1：`POST /api/onboarding/answer`，提交引导对话中的一轮回答。走到
    最后一步(`OnboardingResult`)时，把收集到的字段通过 `write_memory(critical)`
    落进 `user_profile`——引导流程本身的问答/确认步骤就是"人在环确认"，见
    `write_memory.py` 模块文档，这里不再加一层。"""
    set_current_locale(request.locale)
    persist_user_locale(request.user_id, request.locale)
    result = apply_answer(request.step_id, request.answer, request.state, locale=request.locale)
    if isinstance(result, OnboardingResult):
        session = server.open_session(CallerRole.ROUTER, user_id=request.user_id)
        write_result = safe_call_tool(
            session,
            "write_memory",
            {"category": "critical", "payload": result.profile_updates},
        )
        return {
            "step_id": "done",
            "summary": result.summary,
            "profile_updates": result.profile_updates,
            "written": bool(write_result.ok and write_result.result.ok),
        }
    return {"step_id": result.step_id, "prompt": result.prompt, "state": result.state}
