#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQE 检索延迟拆解（ASYNC_DESIGN.md §6.2）。

对比 MQE LLM 改写 vs 顺序 embed+SQL 循环 vs 端到端 `use_mqe` 开关。
需要 Postgres + dev 档 LLM（OpenRouter/Ollama/…）；BGE-M3 首次加载会偏慢，
脚本会先 warmup 一次 embed。

用法:
    python3 evals/run_mqe_retrieval_bench.py
    python3 evals/run_mqe_retrieval_bench.py --query "气虚质冬天适合喝什么汤"
    python3 evals/run_mqe_retrieval_bench.py --loop-only   # 不调 LLM，只测 embed+SQL
    python3 evals/run_mqe_retrieval_bench.py --json-out evals/results/mqe_bench.json

产出: 终端报告；可选 JSON（含每项明细 + summary）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("LANGFUSE_ENABLED", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.env import get_pg_dsn, load_env  # noqa: E402
from backend.llm import adapter as llm_adapter  # noqa: E402
from evals.mqe_retrieval_bench import (  # noqa: E402
    DEFAULT_BENCH_QUERIES,
    benchmark_one_query,
    format_report,
    results_to_json,
)
from backend.mcp_server.tools._retrieval_common import _get_embedder  # noqa: E402


def _warmup_embedder() -> None:
    embedder = _get_embedder()
    embedder.encode_hybrid(["warmup"])


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description="MQE retrieval latency benchmark")
    ap.add_argument(
        "--domain",
        default="tcm",
        choices=("tcm", "nutrition"),
        help="knowledge_chunks.domain filter (default: tcm)",
    )
    ap.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="bench query (repeatable; default: 3 built-in samples)",
    )
    ap.add_argument(
        "--loop-only",
        action="store_true",
        help="skip MQE LLM; only measure single-query embed+SQL loop",
    )
    ap.add_argument(
        "--skip-e2e",
        action="store_true",
        help="skip end-to-end search_knowledge_chunks timing",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        help="write machine-readable results (default: evals/results/mqe_bench_<date>.json)",
    )
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip BGE-M3 warmup encode before timing",
    )
    args = ap.parse_args(argv)

    dsn = get_pg_dsn()
    if not dsn:
        print("DIET_EXPERT_PG_DSN is not set (see .env.example)", file=sys.stderr)
        return 1

    queries = args.queries or list(DEFAULT_BENCH_QUERIES)
    if not args.no_warmup:
        print("Warming up BGE-M3 embedder…", file=sys.stderr)
        _warmup_embedder()

    provider = os.environ.get("LLM_PROVIDER_DEV") or os.environ.get("LLM_PROVIDER")
    model = os.environ.get("LLM_MODEL_DEV")

    results = []
    for query in queries:
        results.append(
            benchmark_one_query(
                query,
                domain=args.domain,
                dsn=dsn,
                complete=llm_adapter.complete,
                run_mqe=not args.loop_only,
                run_e2e=not args.skip_e2e,
            )
        )

    report = format_report(results, model=model, provider=provider)
    print(report)

    out_path = args.json_out
    if out_path is None:
        out_dir = ROOT / "evals" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mqe_bench_{date.today().isoformat()}.json"
    elif not out_path.is_absolute():
        out_path = ROOT / out_path

    payload = results_to_json(results)
    payload["meta"] = {
        "domain": args.domain,
        "provider": provider,
        "model": model,
        "loop_only": args.loop_only,
        "skip_e2e": args.skip_e2e,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
