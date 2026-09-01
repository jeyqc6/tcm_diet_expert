"""MQE retrieval latency breakdown (ASYNC_DESIGN.md §6.2).

Measures:
  1. MQE LLM rewrite (`generate_query_variants`)
  2. Per-query sequential embed + SQL inside `search_knowledge_chunks`
  3. End-to-end `use_mqe=False` vs `use_mqe=True`

Requires Postgres (`DIET_EXPERT_PG_DSN`) and a configured dev-tier LLM when
`run_mqe=True`. See `evals/run_mqe_retrieval_bench.py`.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

DEFAULT_BENCH_QUERIES: tuple[str, ...] = (
    "平和质的人在春天饮食上要注意什么",
    "气虚质冬天适合喝什么汤",
    "糖尿病患者能吃红枣吗",
)

CompleteFn = Callable[[list[dict[str, str]]], Any]


@dataclass(frozen=True)
class QueryLoopTiming:
    index: int
    text: str
    embed_s: float
    sql_s: float
    dense_rows: int
    sparse_rows: int

    @property
    def total_s(self) -> float:
        return self.embed_s + self.sql_s


@dataclass
class QueryBenchResult:
    query: str
    domain: str
    mqe_s: float
    variants: list[str]
    queries: list[str]
    loop_timings: list[QueryLoopTiming] = field(default_factory=list)
    e2e_no_mqe_s: float | None = None
    e2e_mqe_s: float | None = None

    @property
    def embed_total_s(self) -> float:
        return sum(t.embed_s for t in self.loop_timings)

    @property
    def sql_total_s(self) -> float:
        return sum(t.sql_s for t in self.loop_timings)

    @property
    def loop_total_s(self) -> float:
        return self.embed_total_s + self.sql_total_s

    @property
    def mqe_plus_loop_s(self) -> float:
        return self.mqe_s + self.loop_total_s


def merge_queries(original: str, variants: Sequence[str]) -> list[str]:
    """Original query first, then deduped MQE variants."""
    out = [original]
    for variant in variants:
        cleaned = variant.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def summarize_results(results: Sequence[QueryBenchResult]) -> dict[str, Any]:
    if not results:
        return {"n": 0}

    def _avg(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    mqe_vals = [r.mqe_s for r in results]
    loop_vals = [r.loop_total_s for r in results]
    combined_vals = [r.mqe_plus_loop_s for r in results]
    e2e_no = [r.e2e_no_mqe_s for r in results if r.e2e_no_mqe_s is not None]
    e2e_yes = [r.e2e_mqe_s for r in results if r.e2e_mqe_s is not None]

    return {
        "n": len(results),
        "avg_mqe_s": _avg(mqe_vals),
        "avg_loop_s": _avg(loop_vals),
        "avg_mqe_plus_loop_s": _avg(combined_vals),
        "avg_e2e_no_mqe_s": _avg(e2e_no) if e2e_no else None,
        "avg_e2e_mqe_s": _avg(e2e_yes) if e2e_yes else None,
        "avg_query_count": _avg([len(r.queries) for r in results]),
    }


def format_report(
    results: Sequence[QueryBenchResult],
    *,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("MQE retrieval benchmark (ASYNC_DESIGN §6.2)")
    if provider or model:
        lines.append(f"LLM: {provider or '?'} / {model or '?'}")
    lines.append("")

    for result in results:
        lines.append("=" * 72)
        lines.append(f"QUERY: {result.query}")
        lines.append(
            f"  MQE LLM: {result.mqe_s:.2f}s | {len(result.queries)} queries"
        )
        for variant in result.variants:
            lines.append(f"    variant: {variant}")
        for timing in result.loop_timings:
            lines.append(
                f"    q{timing.index}: embed {timing.embed_s * 1000:.0f}ms | "
                f"sql {timing.sql_s * 1000:.0f}ms | "
                f"dense={timing.dense_rows} sparse={timing.sparse_rows}"
            )
        lines.append(
            f"  loop total: embed {result.embed_total_s:.3f}s + "
            f"sql {result.sql_total_s:.3f}s = {result.loop_total_s:.3f}s"
        )
        lines.append(f"  MQE+loop: {result.mqe_plus_loop_s:.2f}s")
        if result.e2e_no_mqe_s is not None:
            lines.append(f"  e2e use_mqe=False: {result.e2e_no_mqe_s:.2f}s")
        if result.e2e_mqe_s is not None:
            lines.append(f"  e2e use_mqe=True: {result.e2e_mqe_s:.2f}s")

    summary = summarize_results(results)
    if summary["n"]:
        lines.append("")
        lines.append("=" * 72)
        lines.append("SUMMARY (averages)")
        lines.append(f"  queries: {summary['n']}")
        lines.append(f"  MQE LLM: {summary['avg_mqe_s']:.2f}s")
        lines.append(f"  embed+SQL loop: {summary['avg_loop_s']:.3f}s")
        lines.append(f"  MQE+loop: {summary['avg_mqe_plus_loop_s']:.2f}s")
        if summary["avg_e2e_no_mqe_s"] is not None:
            lines.append(f"  e2e no MQE: {summary['avg_e2e_no_mqe_s']:.2f}s")
        if summary["avg_e2e_mqe_s"] is not None:
            lines.append(f"  e2e with MQE: {summary['avg_e2e_mqe_s']:.2f}s")
        lines.append(f"  query variants per item: {summary['avg_query_count']:.1f}")
    return "\n".join(lines)


def benchmark_query_loop(
    queries: Sequence[str],
    *,
    domain: str,
    dsn: str,
    fetch_limit: int = 15,
) -> list[QueryLoopTiming]:
    """Mirror `search_knowledge_chunks` sequential embed + SQL loop."""
    from pgvector import SparseVector

    from backend.mcp_server.tools._retrieval_common import (
        _get_embedder,
        _run_dense_query,
        _run_sparse_query,
        build_filter_sql,
    )
    from db.embed_bge_m3 import EMBED_DIM_SPARSE, connect

    embedder = _get_embedder()
    filter_sql, filter_params = build_filter_sql(None)
    timings: list[QueryLoopTiming] = []

    conn = connect(dsn)
    try:
        cur = conn.cursor()
        for index, query in enumerate(queries):
            t_embed = time.perf_counter()
            dense_vecs, sparse_maps = embedder.encode_hybrid([query])
            dense_vec = dense_vecs[0].tolist()
            sparse_map = sparse_maps[0]
            embed_s = time.perf_counter() - t_embed

            t_sql = time.perf_counter()
            dense_rows = _run_dense_query(
                cur, domain, dense_vec, filter_sql, filter_params, fetch_limit
            )
            sparse_vec = SparseVector(sparse_map, EMBED_DIM_SPARSE)
            sparse_rows = _run_sparse_query(
                cur, domain, sparse_vec, filter_sql, filter_params, fetch_limit
            )
            sql_s = time.perf_counter() - t_sql
            timings.append(
                QueryLoopTiming(
                    index=index,
                    text=query,
                    embed_s=embed_s,
                    sql_s=sql_s,
                    dense_rows=len(dense_rows),
                    sparse_rows=len(sparse_rows),
                )
            )
        cur.close()
    finally:
        conn.close()
    return timings


def benchmark_one_query(
    query: str,
    *,
    domain: str,
    dsn: str,
    complete: CompleteFn,
    run_mqe: bool = True,
    run_e2e: bool = True,
    fetch_limit: int = 15,
    fixed_variants: Sequence[str] | None = None,
) -> QueryBenchResult:
    from backend.llm import adapter as llm_adapter
    from backend.mcp_server.tools._retrieval_common import (
        generate_query_variants,
        search_knowledge_chunks,
    )

    mqe_s = 0.0
    if fixed_variants is not None:
        variants = list(fixed_variants)
    elif run_mqe:
        t0 = time.perf_counter()
        variants = generate_query_variants(query, complete=complete)
        mqe_s = time.perf_counter() - t0
    else:
        variants = []

    queries = merge_queries(query, variants)
    loop_timings = benchmark_query_loop(
        queries, domain=domain, dsn=dsn, fetch_limit=fetch_limit
    )

    result = QueryBenchResult(
        query=query,
        domain=domain,
        mqe_s=mqe_s,
        variants=variants,
        queries=queries,
        loop_timings=loop_timings,
    )

    if run_e2e:
        t0 = time.perf_counter()
        search_knowledge_chunks(domain, query, top_k=5, use_mqe=False, use_hybrid=True)
        result.e2e_no_mqe_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        search_knowledge_chunks(
            domain,
            query,
            top_k=5,
            use_mqe=True,
            use_hybrid=True,
            complete=complete or llm_adapter.complete,
        )
        result.e2e_mqe_s = time.perf_counter() - t0

    return result


def results_to_json(results: Sequence[QueryBenchResult]) -> dict[str, Any]:
    return {
        "summary": summarize_results(results),
        "items": [
            {
                **{k: v for k, v in asdict(result).items() if k != "loop_timings"},
                "embed_total_s": result.embed_total_s,
                "sql_total_s": result.sql_total_s,
                "loop_total_s": result.loop_total_s,
                "mqe_plus_loop_s": result.mqe_plus_loop_s,
                "loop_timings": [asdict(t) for t in result.loop_timings],
            }
            for result in results
        ],
    }
