#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 on the production retriever: BGE-M3 + pgvector.

Official metric: labeled Recall@k against e1_retrieval_gold.jsonl
(constitution chapter + season table for 体质×季节 items).

Usage:
    python3 evals/run_vector_m1.py --root .
    python3 evals/run_vector_m1.py --root . --judge   # optional RAGAS-style check
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("LANGFUSE_ENABLED", "0")

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(ROOT_DEFAULT))

from evals.phase8_scoring import load_jsonl  # noqa: E402
from evals.phase8_judge import strip_json_fences  # noqa: E402
from evals.vector_m1 import (  # noqa: E402
    e1_items,
    load_retrieval_gold,
    score_retrieval,
)


def _chunk_to_dict(chunk) -> dict:
    return {
        "source_id": chunk.source_id,
        "domain": chunk.domain,
        "score": chunk.score,
        "text": chunk.text,
    }


def retrieve_domain(item: dict, k: int, dsn: str | None) -> list[dict]:
    from backend.mcp_server.tools.retrieve_nutrition import retrieve_nutrition
    from backend.mcp_server.tools.retrieve_tcm import retrieve_tcm

    query = item.get("query") or ""
    domain = item.get("domain") or "tcm"
    if domain == "nutrition":
        chunks = retrieve_nutrition(query, top_k=k, dsn=dsn)
    else:
        chunks = retrieve_tcm(query, top_k=k, dsn=dsn)
    return [_chunk_to_dict(c) for c in chunks]


def retrieve_all(item: dict, k: int, dsn: str | None) -> list[dict]:
    """Same SQL as search_knowledge_chunks, without the domain lock.

    B0 indexed tcm+nutrition together. This pool is the vector analogue.
    """
    from backend.env import get_pg_dsn
    from backend.mcp_server.tools._retrieval_common import _get_embedder
    from db.embed_bge_m3 import connect

    dsn = get_pg_dsn(dsn)
    if not dsn:
        raise RuntimeError("DIET_EXPERT_PG_DSN is not set")
    query = item.get("query") or ""
    qvec = _get_embedder().encode([query])[0].tolist()
    sql = """
        SELECT chunk_id, domain, source_file, source_type, text, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM knowledge_chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    conn = connect(dsn)
    try:
        cur = conn.cursor()
        cur.execute(sql, [qvec, qvec, k])
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return [
        {
            "source_id": row[0],
            "domain": row[1],
            "score": float(row[6]),
            "text": row[4],
        }
        for row in rows
    ]


CONTEXT_JUDGE_PROMPT = """你是 RAG 检索评审员，只评「召回的资料够不够用来回答」，不评最终回答文笔。
标准要点只以【检索金标】为准：那是知识库里实际有的段落方向。
不要用 derived 交叉表的合成句（例如「平和质春天少酸多甘」必须出现在同一段）。
不要要求冲突规则里的调和细节（例如补铁「餐前后1小时」）——那是回答评测，不是检索金标。
允许同义：干姜可以覆盖生姜，温补可以覆盖温阳。

只返回 JSON：
- context_recall (0/1/2)：仅根据这些资料，能否写出检索金标的方向。2=各组要点基本齐；1=只覆盖一半；0=核心缺失或跑题。
- context_precision (0/1/2)：召回的段落是否大部分和问题相关。2=大部分相关；1=参半；0=大部分跑题。
- rationale：一句话。

{"context_recall":2,"context_precision":2,"rationale":"..."}
"""


def _parse_context_scores(text: str) -> dict | None:
    cleaned = strip_json_fences(text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", cleaned or "", re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(parsed, dict):
        return None

    def _clamp(key, default=0):
        try:
            v = int(parsed.get(key))
        except (TypeError, ValueError):
            return default
        return max(0, min(2, v))

    return {
        "context_recall": _clamp("context_recall"),
        "context_precision": _clamp("context_precision"),
        "rationale": str(parsed.get("rationale") or "")[:300],
    }


def _context_blob(detail: dict, per_chunk: int = 500) -> str:
    parts = []
    for i, c in enumerate(detail.get("context") or [], start=1):
        parts.append(f"[{i} {c.get('source_id')}] {(c.get('text') or '')[:per_chunk]}")
    return "\n---\n".join(parts) or "(空)"


def _count_embeddings(dsn: str) -> dict:
    from db.embed_bge_m3 import connect

    conn = connect(dsn)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT domain, COUNT(*) FROM knowledge_chunks "
            "WHERE embedding IS NOT NULL GROUP BY domain"
        )
        by_domain = {row[0]: int(row[1]) for row in cur.fetchall()}
        cur.close()
    finally:
        conn.close()
    return by_domain


async def judge_pool(pool_result: dict, complete) -> dict:
    from evals.phase8_scoring import threshold_band

    details = pool_result.get("details") or []
    scored = 0
    recall2 = 0
    prec_sum = 0
    for detail in details:
        gt = detail.get("retrieval_gt") or "(无检索金标)"
        user = (
            f"用户问题:{detail.get('query')}\n\n【检索金标】\n{gt}\n\n"
            f"【检索到的资料 top-{pool_result.get('k', 5)}】\n{_context_blob(detail)}"
        )
        parsed = None
        for _ in range(2):
            result = await complete(
                [
                    {"role": "system", "content": CONTEXT_JUDGE_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            parsed = _parse_context_scores(result.text or "")
            if parsed:
                break
        detail["judge"] = parsed
        ok = bool(parsed and parsed["context_recall"] >= 2)
        detail["context_recall_pass"] = ok
        if parsed:
            scored += 1
            recall2 += int(ok)
            prec_sum += parsed["context_precision"]
        print(f"  judge {detail['id']} {'PASS' if ok else 'FAIL'}", flush=True)
    rate = (recall2 / scored) if scored else None
    pool_result["context_recall"] = {
        "n": scored,
        "passed": recall2,
        "rate": rate,
        "mean_precision": round(prec_sum / scored, 3) if scored else None,
        "band": threshold_band("M1", rate),
    }
    return pool_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pools", default="domain", help="comma: domain and/or all")
    ap.add_argument(
        "--judge",
        action="store_true",
        help="optional LLM context-recall against retrieval_gt (not official M1)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    from backend.env import get_pg_dsn, load_env

    load_env()
    dsn = get_pg_dsn()
    if not dsn:
        print("DIET_EXPERT_PG_DSN missing", file=sys.stderr)
        sys.exit(1)

    counts = _count_embeddings(dsn)
    total = sum(counts.values())
    if total == 0:
        print("knowledge_chunks has no embeddings", file=sys.stderr)
        sys.exit(1)

    items = e1_items(load_jsonl(root / "evals" / "dataset.jsonl"))
    gold_by_id = load_retrieval_gold(
        root / "evals" / "reference_tables" / "e1_retrieval_gold.jsonl"
    )
    missing = [r["id"] for r in items if r["id"] not in gold_by_id]
    if missing:
        print(f"gold file missing ids: {missing}", file=sys.stderr)
        sys.exit(1)
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    t0 = time.perf_counter()
    results: dict = {
        "date": str(date.today()),
        "retriever": "BGE-M3 + pgvector (cosine)",
        "official_metric": "labeled Recall@k (group any_of) on domain pool",
        "gold_file": "evals/reference_tables/e1_retrieval_gold.jsonl",
        "k": args.k,
        "n_e1": len(items),
        "embeddings_by_domain": counts,
        "n_embedded": total,
        "notes": [
            "Official = mean Recall@k over gold groups; a group hits if any_of is in top-k.",
            "体质×季节 gold is 体质专章 + 节气表, not derived crossover sentences.",
            "E1-14 gold is tea-tannin vs iron only; S08 time window is answer-side.",
            "strict/union remain keyword diagnostics. LLM --judge is optional.",
        ],
        "pools": {},
    }

    for pool in pools:
        print(f"pool={pool} n={len(items)} k={args.k}", flush=True)
        if pool == "domain":
            retrieve = lambda row, k, _dsn=dsn: retrieve_domain(row, k, _dsn)
        elif pool == "all":
            retrieve = lambda row, k, _dsn=dsn: retrieve_all(row, k, _dsn)
        else:
            print(f"unknown pool {pool}", file=sys.stderr)
            sys.exit(1)
        blob = score_retrieval(items, retrieve, k=args.k, gold_by_id=gold_by_id)
        results["pools"][pool] = blob
        print(
            f"  recall@{args.k}={blob['recall_at_k']:.1%}  "
            f"all_hit={blob['gold_hit_rate']:.1%}  "
            f"union={blob['union_recall']:.1%}  {blob['band']}",
            flush=True,
        )
        for miss in blob["misses"]:
            gold = miss.get("gold") or {}
            print(
                f"    miss {miss['id']} missing_roles={gold.get('missing_roles')}",
                flush=True,
            )

    if args.judge:
        import asyncio

        from backend.llm import adapter as llm_adapter

        async def _run_judges():
            for pool, blob in results["pools"].items():
                print(f"judging pool={pool}", flush=True)
                await judge_pool(blob, llm_adapter.complete)

        asyncio.run(_run_judges())

    results["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out = root / "evals" / "results" / f"vector_m1_{date.today()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written -> {out}")
    for pool, blob in results["pools"].items():
        cr = blob.get("context_recall") or {}
        print(
            f"{pool}: recall_at_k={blob.get('recall_at_k')}  "
            f"all_hit={blob.get('gold_hit_rate')}  "
            f"union={blob.get('union_recall')}  "
            f"context_recall={cr.get('rate')}  "
            f"band={blob.get('band')}"
        )


if __name__ == "__main__":
    main()
