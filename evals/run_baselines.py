#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 3 baseline 跑分：B0(BM25 检索) / B1(单次 LLM) / B3(通用助手风格 LLM)

用法：
    python3 evals/run_baselines.py --root .
    python3 evals/run_baselines.py --root . --smoke   # 只跑 smoke.jsonl
    python3 evals/run_baselines.py --root . --b0-only
    python3 evals/run_baselines.py --root . --smoke --b0-only --check-launch-threshold
        # 对**真实** knowledge/_processed 语料检查 M1 Launch(70%)。
        # 当前全量 B0 是 53.3%，这条门槛今天会红——本地对照用，不是 CI 默认。
    python3 evals/run_baselines.py --root . --smoke --b0-only \\
        --chunks evals/fixtures/bm25_smoke_chunks.jsonl --check-ci-floor 0.6
        # CI 默认：用仓库内最小 fixture，保证 job 每次都跑。
        # fixture 分数 ≠ 真实 M1。Launch 70% 仍是产品门槛，见 EVALUATION.md §5。

产出：evals/results/baseline_<date>.json

Chunk 解析顺序：`--chunks` 显式路径 > `knowledge/_processed/{tcm,nutrition}_chunks.jsonl`
> `evals/fixtures/bm25_smoke_chunks.jsonl`。三者都没有则 **exit(1)**（fail-closed）。
M3/M5 仍无自动化 eval，门槛只覆盖 M1/B0。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# BM25 scorer lives under evals/ (planning/ is gitignored and absent in CI).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from build_and_eval_bm25 import BM25, hit, load_chunks  # noqa: E402

# docs/EVALUATION.md §4 / PRD §8.3：M1 检索 recall@5 的 Launch 阈值。
# Real-corpus B0 (2026-08-26) is 53.3% — this bar is not the CI default.
M1_RECALL_AT_5_LAUNCH_THRESHOLD = 0.70
# Fixture smoke floor: BM25 + keyword rubric still work on vendored chunks.
# Not a claim that production recall is 70%.
CI_FIXTURE_RECALL_FLOOR = 0.60
DEFAULT_FIXTURE_CHUNKS = Path("evals") / "fixtures" / "bm25_smoke_chunks.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_chunk_paths(root: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        paths = [(root / p).resolve() if not Path(p).is_absolute() else Path(p) for p in explicit]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"缺少 --chunks 文件: {missing}", file=sys.stderr)
            sys.exit(1)
        return paths
    processed = [
        root / "knowledge" / "_processed" / "tcm_chunks.jsonl",
        root / "knowledge" / "_processed" / "nutrition_chunks.jsonl",
    ]
    if all(p.exists() for p in processed):
        return processed
    fixture = root / DEFAULT_FIXTURE_CHUNKS
    if fixture.exists():
        print(f"knowledge/_processed 不在，改用 fixture {fixture}", file=sys.stderr)
        return [fixture]
    print(
        "缺少 chunk 文件：knowledge/_processed/{tcm,nutrition}_chunks.jsonl "
        f"和 {DEFAULT_FIXTURE_CHUNKS} 都不存在",
        file=sys.stderr,
    )
    sys.exit(1)


def e1_retrieval_items(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("subset") != "E1":
            continue
        kw = (r.get("expect") or {}).get("retrieval_keywords")
        if kw:
            out.append(r)
    return out


def run_b0(bm25: BM25, chunks: list[dict], items: list[dict], k: int = 5) -> dict:
    hits, misses = 0, []
    for r in items:
        q = r["query"]
        kws = r["expect"]["retrieval_keywords"]
        results = bm25.search(q, k=k)
        hit_chunks = [chunks[i] for i, _ in results]
        ok = any(hit(c["text"], kws) for c in hit_chunks)
        hits += int(ok)
        if not ok:
            misses.append(
                {
                    "id": r["id"],
                    "query": q,
                    "expect": kws,
                    "top": [
                        {
                            "chunk_id": chunks[i].get("chunk_id"),
                            "score": round(s, 3),
                            "head": chunks[i]["text"][:80],
                        }
                        for i, s in results[:3]
                    ],
                }
            )
    n = len(items) or 1
    return {
        "recall_at_k": hits / n,
        "k": k,
        "n": len(items),
        "hits": hits,
        "misses": misses,
    }


def llm_available() -> bool:
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def run_b1_llm(items: list[dict]) -> dict:
    """Single-shot LLM, no retrieval. Keyword rubric on answer_keywords."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"status": "skipped", "reason": "openai not installed"}

    if not llm_available():
        return {"status": "skipped", "reason": "LLM_API_KEY / OPENAI_API_KEY not set"}

    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    system = (
        "你是饮食与中医食养助手。只根据常识回答，不要假装查过数据库。"
        "涉及疾病、用药、过敏时提醒咨询专业人士。"
    )

    scored = []
    for r in items:
        if r.get("subset") not in ("E1", "E2a", "E2b"):
            continue
        exp = r.get("expect") or {}
        must = exp.get("answer_keywords") or exp.get("resolution_keywords") or []
        must_not = exp.get("must_not") or []
        if not must:
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": r["query"]},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = resp.choices[0].message.content or ""
        ok_must = all(k in text for k in must)
        ok_not = all(k not in text for k in must_not)
        scored.append(
            {
                "id": r["id"],
                "pass": ok_must and ok_not,
                "answer_preview": text[:200],
            }
        )
    if not scored:
        return {"status": "skipped", "reason": "no scorable E1/E2 items with answer keywords"}
    passed = sum(1 for s in scored if s["pass"])
    return {
        "status": "ok",
        "model": model,
        "keyword_pass_rate": passed / len(scored),
        "n": len(scored),
        "passed": passed,
        "details": scored,
    }


def run_b3_llm(items: list[dict]) -> dict:
    """Generic assistant: same API, minimal system prompt (no domain framing)."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"status": "skipped", "reason": "openai not installed"}
    if not llm_available():
        return {"status": "skipped", "reason": "LLM_API_KEY / OPENAI_API_KEY not set"}

    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    system = "You are a helpful assistant."

    scored = []
    for r in items:
        if r.get("subset") not in ("E1", "E2a"):
            continue
        exp = r.get("expect") or {}
        must = exp.get("answer_keywords") or exp.get("resolution_keywords") or []
        if not must:
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": r["query"]},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = resp.choices[0].message.content or ""
        passed = all(k in text for k in must)
        scored.append({"id": r["id"], "pass": passed})
    if not scored:
        return {"status": "skipped", "reason": "no scorable items"}
    passed = sum(1 for s in scored if s["pass"])
    return {
        "status": "ok",
        "model": model,
        "keyword_pass_rate": passed / len(scored),
        "n": len(scored),
        "passed": passed,
        "note": "B3 uses generic system prompt; same rubric as B1 for rough comparison",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="项目根目录")
    ap.add_argument("--smoke", action="store_true", help="用 smoke.jsonl 代替全量")
    ap.add_argument("--b0-only", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--chunks",
        nargs="+",
        default=None,
        help="chunk jsonl 路径(可多个)。不传则 knowledge/_processed，再退到 evals/fixtures。",
    )
    ap.add_argument(
        "--check-launch-threshold",
        action="store_true",
        help="M1 recall@k 跌破 docs/EVALUATION.md §4 的 Launch 阈值就 exit(1)（真实语料对照，不是 CI 默认）",
    )
    ap.add_argument(
        "--check-ci-floor",
        nargs="?",
        const=CI_FIXTURE_RECALL_FLOOR,
        type=float,
        default=None,
        help="fixture/CI 地板：recall@k 低于该值 exit(1)。默认 0.6。不是 Launch 70%。",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    eval_path = root / "evals" / ("smoke.jsonl" if args.smoke else "dataset.jsonl")
    rows = load_jsonl(eval_path)

    chunk_paths = _resolve_chunk_paths(root, args.chunks)
    chunks = load_chunks([str(p) for p in chunk_paths])
    bm25 = BM25().fit([c["text"] for c in chunks])

    e1_items = e1_retrieval_items(rows)
    b0 = run_b0(bm25, chunks, e1_items, k=args.k)

    out = {
        "date": str(date.today()),
        "dataset": eval_path.name,
        "n_cases": len(rows),
        "chunk_paths": [str(p) for p in chunk_paths],
        "B0_BM25_retrieval": {
            "description": "PRD §8.4 B0 — BM25 recall@k on E1 items with retrieval_keywords",
            **b0,
        },
    }

    if not args.b0_only:
        b1_items = [r for r in rows if r.get("subset") in ("E1", "E2a", "E2b")]
        out["B1_single_LLM"] = run_b1_llm(b1_items)
        out["B3_generic_assistant"] = run_b3_llm(b1_items)

    results_dir = root / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"baseline_{date.today()}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwritten -> {out_path}")

    recall = out["B0_BM25_retrieval"]["recall_at_k"]
    if args.check_launch_threshold:
        if recall < M1_RECALL_AT_5_LAUNCH_THRESHOLD:
            print(
                f"\n❌ M1 recall@{args.k}={recall:.1%} 跌破 Launch 阈值"
                f"({M1_RECALL_AT_5_LAUNCH_THRESHOLD:.0%})，见 docs/EVALUATION.md §4",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"\n✅ M1 recall@{args.k}={recall:.1%} 达到 Launch 阈值"
            f"({M1_RECALL_AT_5_LAUNCH_THRESHOLD:.0%})"
        )
    if args.check_ci_floor is not None:
        floor = args.check_ci_floor
        if recall < floor:
            print(
                f"\n❌ CI fixture recall@{args.k}={recall:.1%} 低于地板 {floor:.0%} "
                f"(这不是 Launch 70%；真实 B0 全量仍是 53.3%)",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"\n✅ CI fixture recall@{args.k}={recall:.1%} ≥ 地板 {floor:.0%} "
            f"(fixture 回归，不是真实 M1)"
        )


if __name__ == "__main__":
    main()
