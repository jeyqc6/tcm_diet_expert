#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieve_tcm / retrieve_nutrition 共享的检索实现（内部模块，不是 MCP 工具本身）。

两个工具在 docs/ARCHITECTURE.md §2.2 是分开声明的，且在 §2.3 的权限模型里必须是
两个可以独立授予/收回的工具（TCM SubAgent 看不到 retrieve_nutrition，反之亦然）。
但底层"向量检索 + 结构化预筛"的逻辑除了 domain 不一样、完全相同，所以把这部分
抽到这个内部模块里共享，retrieve_tcm.py / retrieve_nutrition.py 各自只是锁定
domain 的薄封装——这样两处工具签名保持独立（架构要求），实现不用抄两遍（避免
两份检索逻辑慢慢长歪）。

混合检索（结构化预筛 + 向量排序，§2.2）：
    WHERE domain = %s [AND source_type = ANY(...)] [AND metadata @> ...]
    ORDER BY embedding <=> $1

⚠️ 和 §2.2 举的例子有一处出入：例子写的是 "source_status='verified'"，但
knowledge_chunks 表目前没有 source_status 列——这个字段目前只在 conflict_rules
表里存在，语义是"这条冲突规则有没有外部引用"，不是通用知识库 chunk 的概念。
这里不假造一个不存在的字段；`filters` 的真实能力是"任意 metadata 键值精确匹配"
（比例子更通用，但不包含 source_status）。像"体质匹配"这种预筛，实际靠的是
chunk 自带的 metadata（比如 core-tables.md 体质九分类表的每一行 chunk 都带
{"代码": "qi_xu", ...} 这样的字段，ingest.py 解析 markdown 表格时写进去的）。
如果确实需要"资料是否已核实"这个维度，需要先决定这个字段该怎么定义、由谁在
ingest 阶段打标，这是一个还没做的设计决策，不是这里能顺手补的。

MCP 协议层（server.py 按角色声明可见工具，§2.3）还没做；这里先实现工具本身的
检索逻辑，可以被直接 import 调用、单测，也是 server.py 未来要接的东西。

## 检索评分方法优化（2026-08-30，见 ARCHITECTURE §2.6 / EVALUATION §7.7）

之前这里是纯稠密向量 cosine 相似度排序，没有别的信号——`RetrievedChunk.score`
算出来之后没有任何地方用它做阈值判断，BGE-M3 自带的 sparse(词法)输出也没启用。
2026-08-30 生产检索评测（EVALUATION.md §7.7）显示官方 context_recall=73.3%
（刚过 Launch 线 70%，离 Target 85% 还差不少），失败案例的共同模式是"一句话
里混了多个信息点(比如体质+季节)，单一向量检索被其中一个信息点主导，另一个
被挤掉"——`ARCHITECTURE.md` §2.6 早就点名这类"词汇鸿沟"问题，提出 MQE/HyDE
两个方向，当时明确"等 baseline 数字稳定再考虑"，现在数字有了、失败模式也
对上了，这次补上两个方向的第一个（MQE）+ 混合检索（dense+sparse）：

1. **混合检索（dense+sparse，`use_hybrid`，默认开）**：BGE-M3 一次
   `model.encode()` 顺带产出稠密向量和词法(sparse)权重两路输出，之前只存了
   前者。词法权重对"疏肝""祛湿"这类专业术语的精确命中比纯语义向量更稳。
   两路分别在 Postgres 里各自排序取候选池，用 Reciprocal Rank Fusion(RRF)
   融合——两路原始分数量纲不同（cosine vs inner product），融合排名比融合
   分数本身更稳妥，不需要给两路分数找一个共同的校准方式。
2. **MQE(`use_mqe`，`retrieve_tcm`/`retrieve_nutrition` 默认开)**：用一次
   轻量 LLM 调用把 query 拆解成最多 `MQE_MAX_VARIANTS` 个不同角度的版本，
   每个版本各自跑一遍上面那套混合检索，所有版本的结果一起参与最终的 RRF
   融合。这是"检索工具实现内部的增强"（§2.6 原文），不是新增一个 agent
   决策点——SubAgent 完全不知道这一层存在，还是只看到"调用一次 retrieve_*
   拿到一批结果"。
3. **score 语义调整**：融合之后返回的 `RetrievedChunk.score` 是这个 chunk
   在所有参与融合的列表里出现过的最高原始 cosine/inner-product 值，**不是**
   RRF 融合分数本身——RRF 分数只有相对大小意义(不同 query 之间不可比)，
   不适合直接暴露给 SubAgent 做"这条资料靠不靠谱"的判断；原始相似度好歹
   还有"越接近1越相关"这个直观含义。SubAgent 该怎么用这个字段，见
   `backend/agents/citation.py` 的 score 使用指引。

## 依赖方向的一处新增说明

`search_knowledge_chunks()` 新增的 `complete: CompleteFn | None` 参数，惰性
默认到 `backend.llm.adapter.complete`——和 `backend/agents/agent_loop.py`
`run_agent_loop()` 的 `complete = complete or llm_adapter.complete` 完全
同一个模式。`backend/mcp_server/tools/` 依赖 `backend/llm/` 不算破坏依赖
方向（`backend/llm/` 是比 `backend/mcp_server/`/`backend/agents/` 更底层的
基础设施，两者都已经在依赖它，参见 `backend/agents/agent_loop.py` 本身）。

真正需要注意的是同步/异步边界：这个模块从头到尾是同步的（psycopg2 是同步
库），但 MQE 需要调用异步的 `complete()`，调用栈可能已经身处
`run_agent_loop()` 一个运行中的 event loop 里面——不能直接 `asyncio.run()`
（嵌套会抛 `RuntimeError`），也不该为了这一处改动把 MCP 协议层/Agent Loop
的同步调用约定整个改成 async（那会牵动全部六个工具 handler 的签名，风险和
收益不成比例）。`_run_coroutine_sync()` 用一个独立线程起一份全新的 event
loop 跑这个协程来绕开这个限制，见该函数文档。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# db/ 目前不是通过 pip 安装的包，是同一个 repo 里的兄弟目录；直接把项目根目录
# 加进 sys.path，复用 db/embed_bge_m3.py 里已经写好、且被验证过的 BgeM3Embedder
# 和 connect()——检索时的 query 向量必须和入库时用同一个模型/同一套归一化逻辑，
# 自己在这里重新实现一遍会有两份 embedding 逻辑悄悄drift 的风险。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.env import get_pg_dsn  # noqa: E402 — 必须排在 sys.path 修补之后，见上面注释
from backend.i18n import apply_language_instruction, current_locale  # noqa: E402
from backend.llm import adapter as llm_adapter  # noqa: E402
from backend.llm.adapter import CompleteFn  # noqa: E402

logger = logging.getLogger("diet_expert.mcp_server.retrieval")

# MQE：原始 query 之外最多再改写出几个版本(含原始共 MQE_MAX_VARIANTS+1 路)。
# 数字越大覆盖的角度越多，但检索延迟/DB 往返次数跟着线性涨，2 是"明显能覆盖
# 双信息点查询、又不至于让一次检索变成好几秒"的折中值，不是精确调过的超参。
MQE_MAX_VARIANTS = 2

# 每一路(每个 query 变体 × dense/sparse)候选池取多大再送进 RRF 融合——比最终
# 的 top_k 大几倍，给融合留排序空间(如果每路只取 top_k 条，一个只在某一路
# 排第 4、5 名的强相关 chunk 可能在还没机会被其他路"接住"之前就被截断掉)。
HYBRID_FETCH_MULTIPLIER = 3

# Reciprocal Rank Fusion 的平滑常数——业界惯用默认值(来自 RRF 原始论文的
# 经验设置)，不是针对这个项目调过的超参。
RRF_K = 60

_MQE_SYSTEM_PROMPT = (
    "你是检索前的查询改写助手，服务于中医/营养知识库的向量检索。"
    "给定用户的一句提问，如果它包含多个独立的信息点（比如同时提到体质和季节、"
    "同时提到食材和健康状况），把它拆解成最多 {n} 个不同角度的查询，"
    "分别覆盖每个信息点，帮助向量检索不会被其中一个信息点主导而漏掉另一个；"
    "如果问题本来就只有一个信息点，不要为了凑数量硬拆，返回一个改写版本即可。"
    "只输出一个 JSON 字符串数组（例如 [\"改写1\", \"改写2\"]），不要输出任何"
    "其他文字，不要用 markdown 代码块包裹。"
    "知识库原文是中文：如果用户问题是英文，改写查询也请用中文，以便命中中文 chunk。"
)


@dataclass
class RetrievedChunk:
    """source_id 就是 knowledge_chunks.chunk_id，是溯源(citation grounding，
    backend/agents/citation.py)引用的唯一标识。

    `score`：融合检索(2026-08-30)之后，这是该 chunk 在所有参与融合的列表里
    出现过的最高原始相似度(dense 用 cosine，sparse 用 inner product)——不是
    RRF 融合分数本身。数值越接近 1 代表某一路检索认为它越相关；这个字段的
    使用指引见 `backend/agents/citation.py`。"""

    source_id: str
    domain: str
    source_file: str
    source_type: str | None
    text: str
    metadata: dict[str, Any]
    score: float


def build_filter_sql(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """把 filters dict 拼成可以直接接在 `WHERE domain = %s` 后面的 SQL 片段。

    纯函数，不碰数据库，方便单测（tests/unit/mcp_server/test_retrieval_filters.py）。

    规则：
      - "source_type" 是保留键，值可以是 str 或 list[str]，映射成 source_type = ANY(...)
      - 其余任意键值对，收进一个 dict，映射成单个 `metadata @> %s::jsonb` 包含查询
        （多个 metadata 键会被 psycopg2 一次性以一个 JSON 对象传入，@> 天然对多键做 AND，
        不需要每个键单独一条子句）
      - 值为 None 的键会被忽略（视为"没有传这个过滤条件"，不是"过滤出 null"）
      - filters 为 None/空 dict 时返回 ("", [])
    """
    if not filters:
        return "", []

    remaining = dict(filters)
    clauses: list[str] = []
    params: list[Any] = []

    source_type = remaining.pop("source_type", None)
    if source_type:
        if isinstance(source_type, str):
            source_type = [source_type]
        clauses.append("source_type = ANY(%s)")
        params.append(list(source_type))

    metadata_filter = {k: v for k, v in remaining.items() if v is not None}
    if metadata_filter:
        clauses.append("metadata @> %s::jsonb")
        params.append(json.dumps(metadata_filter, ensure_ascii=False))

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def reciprocal_rank_fusion(
    ranked_id_lists: Sequence[Sequence[str]], *, k: int = RRF_K
) -> dict[str, float]:
    """把多路已排序的 id 列表（比如"稠密检索的结果"、"稀疏检索的结果"、
    "每个 MQE 改写版本各自的检索结果"）融合成一个分数——业界标准的
    Reciprocal Rank Fusion：某个 id 在第 i 路结果里排第 r 名(从 1 开始)，
    贡献 `1/(k+r)` 分，同一个 id 出现在多路里分数累加。

    只按排名算分，不需要不同路的原始分数(cosine 相似度 vs 词法内积)在同一个
    量纲上可比——这正是要融合两种数值含义完全不同的分数时，比直接加权求和
    更稳妥的地方：不管两路的分数分布差多远，"排第一"永远贡献 `1/(k+1)`。

    纯函数，不碰数据库/模型，只需要 id 列表就能单测。"""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, source_id in enumerate(ranked_ids, start=1):
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (k + rank)
    return scores


def _run_coroutine_sync(coro: Any) -> Any:
    """在同步函数里安全地跑一个协程——`search_knowledge_chunks()` 是同步函数
    (psycopg2 是同步库)，但 MQE 需要调用异步的 `complete()`。调用栈可能已经
    身处 `backend/agents/agent_loop.py` `run_agent_loop()` 一个运行中的
    event loop 里面——不能直接 `asyncio.run()`(嵌套会抛
    `RuntimeError: asyncio.run() cannot be called from a running event loop`)。
    用一个独立线程起一份全新的 event loop 跑这个协程，不管调用方所在线程
    有没有正在运行的 loop 都能正常工作；没有运行中 loop 时(比如单测、CLI
    直接调用)直接 `asyncio.run()`，不额外开线程。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def generate_query_variants(
    query: str,
    *,
    complete: CompleteFn,
    max_variants: int = MQE_MAX_VARIANTS,
    locale: str | None = None,
) -> list[str]:
    """MQE(ARCHITECTURE §2.6)：用一次轻量 LLM 调用把 query 拆解成最多
    `max_variants` 个不同角度的改写版本，缓解"一句话里混了多个信息点，单一
    向量检索被其中一个信息点主导"的问题(2026-08-30 EVALUATION.md §7.7 的
    失败案例分析——比如"平和质的人在春天饮食上要注意什么"，体质/季节两个
    信息点之间没有任何标点或连接词，规则式拆分做不到，必须理解语义)。

    失败（LLM 调用异常、返回不是合法 JSON 字符串数组）时静默降级为空
    列表——调用方在此基础上还有原始 query 本身可用，不因为这一步失败就让
    整次检索失败，同 `fetch_user_profile`/`load_session_history` 的静默
    降级原则一致。"""
    messages = [
        {
            "role": "system",
            "content": apply_language_instruction(
                _MQE_SYSTEM_PROMPT.format(n=max_variants),
                locale if locale is not None else current_locale(),
            ),
        },
        {"role": "user", "content": query},
    ]
    try:
        result = _run_coroutine_sync(complete(messages))
        text = (result.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
        variants = json.loads(text)
        if not isinstance(variants, list):
            return []
        cleaned = [str(v).strip() for v in variants if str(v).strip()]
        return cleaned[:max_variants]
    except Exception:
        logger.warning("MQE query rewrite failed, falling back to original query only", exc_info=True)
        return []


_EMBEDDER = None
# 2026-08-31：`backend/agents/agent_loop.py` 把一轮里的多个工具调用改成
# `asyncio.to_thread` 并发执行之后（同一轮内的并发 + 两个 SubAgent 本来就并行
# 派发，双重叠加），第一次真实检索请求可能有两个以上线程同时跑到这里、同时看到
# `_EMBEDDER is None`——这是经典的惰性单例竞态：不加锁的话，多个线程会并发
# 各自跑一遍 `BgeM3Embedder()`（模型加载，非幂等、非线程安全的一次性初始化，
# 可能涉及从 HuggingFace Hub 下载/写入同一份本地缓存），实测直接表现为两个
# SubAgent 的检索调用一起卡死、双双撞上 45s SubAgent 超时。双重检查锁定
# （`threading.Lock`，不是 `asyncio.Lock`——这个函数从 `asyncio.to_thread` 派生
# 的工作线程里调用，不在事件循环线程上）保证真正的模型加载只发生一次，其余
# 线程等锁、拿到已经建好的单例，不重复加载。"""
_EMBEDDER_LOCK = threading.Lock()


def _get_embedder():
    """惰性单例：模型加载有真实成本（首次几十秒），SubAgent 一次任务里可能
    多次调用检索工具（D20 行为点#1），不能每次调用都重新 from_pretrained 一遍。
    加锁的原因见上面 `_EMBEDDER_LOCK` 的注释——不是防御性编程，是真实复现过
    的并发 bug。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                from db.embed_bge_m3 import BgeM3Embedder

                _EMBEDDER = BgeM3Embedder()
    return _EMBEDDER


def _row_to_chunk(row: Sequence[Any]) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=row[0],
        domain=row[1],
        source_file=row[2],
        source_type=row[3],
        text=row[4],
        metadata=row[5],
        score=float(row[6]),
    )


def _run_dense_query(
    cur: Any, domain: str, qvec: list[float], filter_sql: str, filter_params: list[Any], limit: int
) -> list[RetrievedChunk]:
    sql = f"""
        SELECT chunk_id, domain, source_file, source_type, text, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM knowledge_chunks
        WHERE domain = %s{filter_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [qvec, domain, *filter_params, qvec, limit]
    cur.execute(sql, params)
    return [_row_to_chunk(row) for row in cur.fetchall()]


def _run_sparse_query(
    cur: Any, domain: str, qsparse: Any, filter_sql: str, filter_params: list[Any], limit: int
) -> list[RetrievedChunk]:
    """`sparse_embedding` 可空(旧数据/还没重新 ingest 的行)，`IS NOT NULL`
    这条条件让这些行天然不参与稀疏这一路的排序，不需要在应用层额外过滤。"""
    sql = f"""
        SELECT chunk_id, domain, source_file, source_type, text, metadata,
               1 - (sparse_embedding <=> %s::sparsevec) AS score
        FROM knowledge_chunks
        WHERE domain = %s AND sparse_embedding IS NOT NULL{filter_sql}
        ORDER BY sparse_embedding <=> %s::sparsevec
        LIMIT %s
    """
    params = [qsparse, domain, *filter_params, qsparse, limit]
    cur.execute(sql, params)
    return [_row_to_chunk(row) for row in cur.fetchall()]


def search_knowledge_chunks(
    domain: str,
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    dsn: str | None = None,
    *,
    use_mqe: bool = False,
    use_hybrid: bool = True,
    complete: CompleteFn | None = None,
    locale: str | None = None,
) -> list[RetrievedChunk]:
    """domain 由调用方（retrieve_tcm/retrieve_nutrition）锁定，不对外暴露成参数——
    这正是它们必须是两个独立工具而不是一个带 domain 参数的工具的原因（§2.3 权限
    分层要求"协议层就不存在"，而不是"这个工具存在但业务代码判断你传的 domain
    合不合法"）。

    `use_mqe`（默认关）/`use_hybrid`（默认开）：见模块文档"检索评分方法优化"
    一节。这一层的默认值偏保守（`use_mqe=False`）——直接调用这个函数的既有
    调用方（单测、`db/embed_bge_m3.py` 之外的脚本）行为不因为这次改动而变化，
    除非显式传 `use_mqe=True`。真正面向 SubAgent 的 `retrieve_tcm`/
    `retrieve_nutrition` 才把 `use_mqe` 也默认打开——那才是"检索工具内部的
    增强，SubAgent 不需要知道"这句话真正生效的地方。

    `complete`：MQE 用的 LLM 调用，惰性默认到 `backend.llm.adapter.complete`
    （不在函数签名默认值里直接写 `llm_adapter.complete`，而是运行时才解析，
    同 `run_agent_loop()` 的既有模式——方便单测注入假实现，不用打真实网络）。
    `use_mqe=False` 时完全不会被用到。
    """
    from db.embed_bge_m3 import EMBED_DIM_SPARSE, connect

    dsn = get_pg_dsn(dsn)
    if not dsn:
        raise RuntimeError(
            "没有连接串。传 dsn 参数、export DIET_EXPERT_PG_DSN=...，"
            "或在项目根目录 .env 里设置 DIET_EXPERT_PG_DSN（见 .env.example）"
        )

    embedder = _get_embedder()
    queries = [query]
    if use_mqe:
        resolved_complete = complete or llm_adapter.complete
        for variant in generate_query_variants(
            query,
            complete=resolved_complete,
            locale=locale if locale is not None else current_locale(),
        ):
            if variant not in queries:
                queries.append(variant)

    filter_sql, filter_params = build_filter_sql(filters)
    fetch_limit = max(top_k * HYBRID_FETCH_MULTIPLIER, 10)

    ranked_lists: list[list[str]] = []
    best_chunk: dict[str, RetrievedChunk] = {}

    def _absorb(chunks: list[RetrievedChunk]) -> None:
        ranked_lists.append([c.source_id for c in chunks])
        for c in chunks:
            existing = best_chunk.get(c.source_id)
            if existing is None or c.score > existing.score:
                best_chunk[c.source_id] = c

    conn = connect(dsn)
    try:
        cur = conn.cursor()
        for q in queries:
            if use_hybrid:
                dense_vecs, sparse_maps = embedder.encode_hybrid([q])
                dense_vec = dense_vecs[0].tolist()
                sparse_map = sparse_maps[0]
            else:
                dense_vec = embedder.encode([q])[0].tolist()
                sparse_map = None

            _absorb(_run_dense_query(cur, domain, dense_vec, filter_sql, filter_params, fetch_limit))

            if use_hybrid and sparse_map:
                from pgvector import SparseVector

                sparse_vec = SparseVector(sparse_map, EMBED_DIM_SPARSE)
                _absorb(_run_sparse_query(cur, domain, sparse_vec, filter_sql, filter_params, fetch_limit))
        cur.close()
    finally:
        conn.close()

    fused_scores = reciprocal_rank_fusion(ranked_lists)
    ranked_ids = sorted(fused_scores, key=lambda sid: fused_scores[sid], reverse=True)
    return [best_chunk[sid] for sid in ranked_ids[:top_k]]
