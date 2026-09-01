#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8 eval: B0 (retrieval) + B1/B3 (keyword rubric via adapter) on both
MODEL_TIER values, plus PRD §8.3 slice report and Launch/Target/Stretch bands.

Usage:
    python3 evals/run_phase8.py --root .
    python3 evals/run_phase8.py --root . --tiers dev
    python3 evals/run_phase8.py --root . --tiers prod --prod-model openai/gpt-4o-mini
    python3 evals/run_phase8.py --root . --b0-only
    python3 evals/run_phase8.py --root . --limit 3   # debug

Does not implement B2 (single-agent ablation). That row was not selected.

交付档 = MODEL_TIER=prod (adapter.py). 开发档 = MODEL_TIER=dev.
This machine's .env only sets the dev OpenRouter model; prod is selected
via --prod-provider / --prod-model for the duration of the prod run.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# Eval must not ship health-data spans to Langfuse.
os.environ.setdefault("LANGFUSE_ENABLED", "0")

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(ROOT_DEFAULT))

from evals.phase8_scoring import (  # noqa: E402
    PRD_THRESHOLDS,
    SLICE_LABELS,
    expected_must,
    has_disclaimer,
    item_prior_user_text,
    item_query,
    load_jsonl,
    score_keywords,
    score_slice_extras,
    slice_inventory,
    slices_for_item,
    summarize_scored,
    threshold_band,
)

B1_SYSTEM = (
    "你是饮食与中医食养助手。只根据常识回答，不要假装查过数据库。"
    "涉及疾病、用药、过敏、具体症状时，必须说明这是传统食养参考、不构成诊断或治疗，"
    "并建议咨询医生或执业中医师。"
    "不要给出热量/体重的数值目标，不要推荐用户已知过敏的食物。"
)
B3_SYSTEM = "You are a helpful assistant."


def _load_baselines_mod(root: Path):
    path = root / "evals" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("diet_expert_run_baselines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_tier_env(*, tier: str, provider: str | None, model: str | None) -> dict[str, str]:
    """Point adapter at this run's tier. Returns the resolved config snapshot."""
    os.environ["MODEL_TIER"] = tier
    if tier == "prod":
        if provider:
            os.environ["LLM_PROVIDER_PROD"] = provider
        if model:
            os.environ["LLM_MODEL_PROD"] = model
    elif tier == "dev":
        if provider:
            os.environ["LLM_PROVIDER_DEV"] = provider
        if model:
            os.environ["LLM_MODEL_DEV"] = model
    from backend.llm.adapter import ModelTier, _model_for_tier, _provider_name_for_tier

    mt = ModelTier.PROD if tier == "prod" else ModelTier.DEV
    resolved_provider = _provider_name_for_tier(mt)
    resolved_model = _model_for_tier(mt, resolved_provider)
    return {
        "tier": tier,
        "provider": resolved_provider,
        "model": resolved_model,
        "meaning": "交付档" if tier == "prod" else "开发档",
    }


def _user_prompt(row: dict) -> str:
    prior = item_prior_user_text(row)
    query = item_query(row)
    if prior:
        return f"用户此前说过：{prior}\n请在已经知道这些信息的前提下回答：\n{query}"
    return query


def _is_scorable(row: dict) -> bool:
    return bool(expected_must(row) or (row.get("expect") or {}).get("must_not")
                or (row.get("expect") or {}).get("final_must_not"))


async def _one_complete(messages: list[dict]) -> tuple[str, dict]:
    from backend.llm.adapter import complete

    result = await complete(messages, temperature=0.2, max_tokens=500)
    meta = {
        "model": result.model,
        "provider": result.provider,
        "tier": result.tier.value if hasattr(result.tier, "value") else str(result.tier),
        "latency_ms": round(result.latency_ms, 1),
        "cost_est": result.cost_est,
        "tokens": result.usage.total_tokens if result.usage else None,
    }
    return result.text or "", meta


async def run_llm_baseline(
    items: list[dict],
    *,
    system: str,
    baseline: str,
    pause_s: float,
    checkpoint: Path | None = None,
) -> dict:
    details = []
    prior_by_id: dict[str, dict] = {}
    if checkpoint is not None and checkpoint.exists():
        try:
            prior = json.loads(checkpoint.read_text(encoding="utf-8"))
            if isinstance(prior, list):
                prior_by_id = {r["id"]: r for r in prior if isinstance(r, dict) and r.get("id")}
        except (OSError, json.JSONDecodeError):
            prior_by_id = {}
    t0 = time.perf_counter()
    for i, row in enumerate(items):
        if not _is_scorable(row):
            continue
        prompt = _user_prompt(row)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        err = None
        text = ""
        meta: dict = {}
        try:
            text, meta = await _one_complete(messages)
        except Exception as exc:  # noqa: BLE001 — item-level isolation
            err = f"{type(exc).__name__}: {exc}"
        kw = score_keywords(text, row)
        extras = score_slice_extras(text, row)
        # A transport/auth error is not a keyword pass.
        passed = bool(kw["pass"]) and err is None and bool(text.strip())
        details.append(
            {
                "id": row["id"],
                "subset": row.get("subset"),
                "slices": slices_for_item(row),
                "metrics": row.get("metrics") or [],
                "pass": passed,
                "scored": err is None,
                "missing": kw["missing"],
                "leaked": kw["leaked"],
                "coverage": kw["coverage"],
                "disclaimer": has_disclaimer(text),
                "extras": extras,
                "error": err,
                "answer": text,
                "answer_preview": text[:400],
                "answer_len": len(text),
                **meta,
            }
        )
        print(
            f"  [{baseline}] {i + 1}/{len(items)} {row['id']} "
            f"{'PASS' if kw['pass'] else 'FAIL'}"
            + (f" err={err}" if err else ""),
            flush=True,
        )
        if checkpoint is not None:
            merged = dict(prior_by_id)
            for row_out in details:
                merged[row_out["id"]] = row_out
            ordered = list(merged.values())
            checkpoint.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if pause_s and i + 1 < len(items):
            await asyncio.sleep(pause_s)
    elapsed = time.perf_counter() - t0
    summary = summarize_scored(details)
    by_subset = {}
    for subset in ("E1", "E2a", "E2b", "E3"):
        by_subset[subset] = summarize_scored(d for d in details if d.get("subset") == subset)
    return {
        "baseline": baseline,
        "elapsed_s": round(elapsed, 1),
        "summary": summary,
        "by_subset": by_subset,
        "details": details,
    }


def run_b0(root: Path, rows: list[dict], baselines_mod) -> dict:
    t0 = time.perf_counter()
    chunk_paths = baselines_mod._resolve_chunk_paths(root, None)
    chunks = baselines_mod.load_chunks([str(p) for p in chunk_paths])
    bm25 = baselines_mod.BM25().fit([c["text"] for c in chunks])
    e1 = baselines_mod.e1_retrieval_items(rows)
    b0 = baselines_mod.run_b0(bm25, chunks, e1, k=5)
    b0["elapsed_s"] = round(time.perf_counter() - t0, 1)
    b0["chunk_paths"] = [str(p) for p in chunk_paths]
    b0["corpus"] = (
        "knowledge/_processed"
        if any("_processed" in str(p) for p in chunk_paths)
        else "fixture"
    )
    return b0


def _metric_from_details(details: list[dict], subsets: set[str]) -> dict:
    picked = [d for d in details if d.get("subset") in subsets]
    return summarize_scored(picked)


def build_slice_report(details_by_tier: dict[str, list[dict]], inventory: dict[str, list[str]]) -> dict:
    report = {}
    for label in SLICE_LABELS:
        ids = inventory[label]
        entry: dict = {"n_items_in_dataset": len(ids), "ids": ids, "tiers": {}}
        for tier, details in details_by_tier.items():
            in_slice = [d for d in details if label in (d.get("slices") or [])]
            keyword = summarize_scored(in_slice)
            allergen = [
                d for d in in_slice if isinstance((d.get("extras") or {}).get("allergen_block"), dict)
            ]
            allergen_pass = sum(1 for d in allergen if d["extras"]["allergen_block"]["pass"])
            m8 = [d for d in in_slice if "m8_caution" in (d.get("extras") or {})]
            m8_pass = sum(1 for d in m8 if d["extras"]["m8_caution"])
            disc = [d for d in in_slice if "disclaimer" in (d.get("extras") or {})]
            # extras["disclaimer"] is a bool for 症状类
            disc_pass = sum(1 for d in disc if d["extras"].get("disclaimer"))
            m9 = [d for d in in_slice if "m9_proxy" in (d.get("extras") or {})]
            m9_pass = sum(1 for d in m9 if d["extras"]["m9_proxy"])
            numeric = [
                d for d in in_slice
                if isinstance((d.get("extras") or {}).get("numeric_block"), dict)
            ]
            numeric_pass = sum(1 for d in numeric if d["extras"]["numeric_block"]["pass"])
            m5_items = [d for d in in_slice if "M5" in (d.get("metrics") or [])]
            entry["tiers"][tier] = {
                "keyword": keyword,
                "allergen_block": {"n": len(allergen), "passed": allergen_pass, "rate": (allergen_pass / len(allergen) if allergen else None)},
                "m8_caution": {"n": len(m8), "passed": m8_pass, "rate": (m8_pass / len(m8) if m8 else None)},
                "disclaimer": {"n": len(disc), "passed": disc_pass, "rate": (disc_pass / len(disc) if disc else None)},
                "m9_proxy": {"n": len(m9), "passed": m9_pass, "rate": (m9_pass / len(m9) if m9 else None)},
                "numeric_block": {"n": len(numeric), "passed": numeric_pass, "rate": (numeric_pass / len(numeric) if numeric else None)},
                "m5_keyword": summarize_scored(m5_items),
            }
        report[label] = entry
    return report


def build_threshold_table(metrics: dict[str, float | None]) -> list[dict]:
    rows = []
    for metric_id, value in metrics.items():
        spec = PRD_THRESHOLDS.get(metric_id, {})
        rows.append(
            {
                "id": metric_id,
                "value": None if value is None else round(value, 4),
                "value_pct": None if value is None else f"{value:.1%}",
                "launch": spec.get("launch"),
                "target": spec.get("target"),
                "stretch": spec.get("stretch"),
                "band": threshold_band(metric_id, value),
            }
        )
    return rows


def collect_official_metrics(b0: dict, b1: dict | None) -> dict[str, float | None]:
    """Metrics Phase 8 can actually compute. Others stay 未跑."""
    m1 = b0.get("recall_at_k")
    m3 = m5 = None
    if b1:
        m3 = _metric_from_details(b1.get("details") or [], {"E1"}).get("rate")
        m5 = _metric_from_details(b1.get("details") or [], {"E2a", "E2b"}).get("rate")
    return {
        "M1": m1,
        "M3": m3,
        "M5": m5,
        "M6": None,
        "M6b": None,
        "M10": None,
        "M11": None,
        "M12": None,
        "M13": None,
        "M14": None,
    }


async def _run_tier(
    items: list[dict],
    *,
    tier: str,
    provider: str | None,
    model: str | None,
    skip_b1: bool,
    skip_b3: bool,
    pause_s: float,
    checkpoint_dir: Path | None = None,
) -> dict:
    resolved = apply_tier_env(tier=tier, provider=provider, model=model)
    print(f"\n=== {resolved['meaning']} MODEL_TIER={tier} {resolved['provider']}/{resolved['model']} ===", flush=True)
    ck_b1 = (checkpoint_dir / f"{tier}_B1.partial.json") if checkpoint_dir else None
    ck_b3 = (checkpoint_dir / f"{tier}_B3.partial.json") if checkpoint_dir else None
    b1 = None
    if not skip_b1:
        b1 = await run_llm_baseline(
            items, system=B1_SYSTEM, baseline="B1", pause_s=pause_s, checkpoint=ck_b1
        )
    b3 = None
    if not skip_b3:
        b3 = await run_llm_baseline(
            items, system=B3_SYSTEM, baseline="B3", pause_s=pause_s, checkpoint=ck_b3
        )
    return {"resolved": resolved, "B1": b1, "B3": b3}


def _sanitize_detail(row: dict) -> dict:
    """Errors and empty answers cannot count as keyword passes."""
    out = dict(row)
    if out.get("error") or not (out.get("answer_preview") or "").strip():
        if out.get("error") or out.get("answer_len") == 0:
            out["pass"] = False
            out["scored"] = False
    return out


def _baseline_from_details(details: list[dict], baseline: str) -> dict:
    cleaned = [_sanitize_detail(d) for d in details]
    return {
        "baseline": baseline,
        "elapsed_s": None,
        "summary": summarize_scored(cleaned),
        "by_subset": {
            subset: summarize_scored(d for d in cleaned if d.get("subset") == subset)
            for subset in ("E1", "E2a", "E2b", "E3")
        },
        "details": cleaned,
    }


def _tiers_from_partials(results_dir: Path) -> dict:
    out: dict = {}
    for tier in ("dev", "prod"):
        b1_path = results_dir / f"{tier}_B1.partial.json"
        b3_path = results_dir / f"{tier}_B3.partial.json"
        if not b1_path.exists() and not b3_path.exists():
            continue
        b1 = json.loads(b1_path.read_text(encoding="utf-8")) if b1_path.exists() else []
        b3 = json.loads(b3_path.read_text(encoding="utf-8")) if b3_path.exists() else []
        sample = next((r for r in b1 if r.get("model")), {})
        out[tier] = {
            "resolved": {
                "tier": tier,
                "provider": sample.get("provider"),
                "model": sample.get("model"),
                "meaning": "交付档" if tier == "prod" else "开发档",
            },
            "status": "ok",
            "B1": _baseline_from_details(b1, "B1") if b1 else None,
            "B3": _baseline_from_details(b3, "B3") if b3 else None,
        }
    return out


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--smoke", action="store_true", help="use smoke.jsonl (15) instead of full 40")
    ap.add_argument("--tiers", default="dev,prod", help="comma list: dev,prod")
    ap.add_argument("--prod-provider", default=None, help="override LLM_PROVIDER_PROD for this run")
    ap.add_argument("--prod-model", default=None, help="override LLM_MODEL_PROD for this run")
    ap.add_argument("--dev-provider", default=None)
    ap.add_argument("--dev-model", default=None)
    ap.add_argument("--b0-only", action="store_true")
    ap.add_argument("--skip-b1", action="store_true", help="skip domain-framed B1 baseline")
    ap.add_argument("--skip-b3", action="store_true", help="skip generic-assistant baseline")
    ap.add_argument("--limit", type=int, default=0, help="first N items (debug)")
    ap.add_argument("--pause", type=float, default=0.4, help="sleep between LLM calls")
    ap.add_argument("--ids", default="", help="comma-separated item ids")
    ap.add_argument(
        "--assemble",
        action="store_true",
        help="rebuild phase8_*.json from *.partial.json, no new LLM calls",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    from backend.env import load_env

    load_env()

    eval_name = "smoke.jsonl" if args.smoke else "dataset.jsonl"
    rows = load_jsonl(root / "evals" / eval_name)
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r.get("id") in want]
    if args.limit:
        rows = rows[: args.limit]

    baselines_mod = _load_baselines_mod(root)
    started = _iso_now()
    wall0 = time.perf_counter()
    print(f"dataset={eval_name} n={len(rows)} started={started}", flush=True)

    b0 = run_b0(root, rows, baselines_mod)
    print(
        f"B0 recall@{b0['k']}={b0['recall_at_k']:.1%} ({b0['hits']}/{b0['n']}) "
        f"corpus={b0['corpus']} {b0['elapsed_s']}s",
        flush=True,
    )

    inventory = slice_inventory(rows)
    print("slices:", {k: len(v) for k, v in inventory.items()}, flush=True)

    tiers_out: dict[str, dict] = {}
    if args.assemble:
        tiers_out = _tiers_from_partials(root / "evals" / "results")
    elif not args.b0_only:
        wanted = [t.strip() for t in args.tiers.split(",") if t.strip()]
        for tier in wanted:
            provider = args.prod_provider if tier == "prod" else args.dev_provider
            model = args.prod_model if tier == "prod" else args.dev_model
            try:
                tiers_out[tier] = asyncio.run(
                    _run_tier(
                        rows,
                        tier=tier,
                        provider=provider,
                        model=model,
                        skip_b1=args.skip_b1,
                        skip_b3=args.skip_b3,
                        pause_s=args.pause,
                        checkpoint_dir=root / "evals" / "results",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                tiers_out[tier] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "resolved": {"tier": tier},
                }
                print(f"tier {tier} failed: {exc}", file=sys.stderr)

    details_by_tier = {}
    for tier, blob in tiers_out.items():
        b1 = (blob.get("B1") or {}) if isinstance(blob, dict) else {}
        details_by_tier[tier] = b1.get("details") or []

    slice_report = build_slice_report(details_by_tier, inventory)

    threshold_by_tier = {}
    official_b0_metrics = collect_official_metrics(b0, None)
    threshold_by_tier["retrieval_only"] = build_threshold_table(official_b0_metrics)
    for tier, blob in tiers_out.items():
        b1 = blob.get("B1") if isinstance(blob, dict) else None
        threshold_by_tier[tier] = build_threshold_table(collect_official_metrics(b0, b1))
        # Slice extras onto the same table for the official (B1) answers.
        if b1:
            sl = slice_report
            extra_metrics = {
                "allergen_block": (sl.get("特禀质") or {}).get("tiers", {}).get(tier, {}).get("allergen_block", {}).get("rate"),
                "disclaimer": (sl.get("症状类") or {}).get("tiers", {}).get(tier, {}).get("disclaimer", {}).get("rate"),
                "numeric_block": (sl.get("weight_management") or {}).get("tiers", {}).get(tier, {}).get("numeric_block", {}).get("rate"),
                "M5_weight_management": (sl.get("weight_management") or {}).get("tiers", {}).get(tier, {}).get("m5_keyword", {}).get("rate"),
                "M9_supplement": (sl.get("补剂交互") or {}).get("tiers", {}).get(tier, {}).get("m9_proxy", {}).get("rate"),
            }
            threshold_by_tier[tier].extend(build_threshold_table(extra_metrics))

    elapsed = round(time.perf_counter() - wall0, 1)
    out = {
        "date": str(date.today()),
        "started_at": started,
        "finished_at": _iso_now(),
        "elapsed_s": elapsed,
        "dataset": eval_name,
        "n_cases": len(rows),
        "phase": 8,
        "b2_ablation": "not_run",
        "notes": [
            "B0 is retrieval-only and identical across model tiers.",
            "B1/B3 go through backend.llm.adapter.complete (MODEL_TIER), not the raw OpenAI client in run_baselines.py.",
            "E3 last-turn answers use prior user text as a prompt stub; that is not M6 persistence.",
            "M9_proxy is the keyword rubric, not Ragas faithfulness.",
            "B2 single-agent ablation was not requested and was not run.",
        ],
        "B0_BM25_retrieval": b0,
        "tiers": {
            k: {
                "resolved": v.get("resolved"),
                "status": v.get("status", "ok"),
                "error": v.get("error"),
                "B1": _compact_baseline(v.get("B1")),
                "B3": _compact_baseline(v.get("B3")),
            }
            for k, v in tiers_out.items()
        },
        "slice_inventory": inventory,
        "slice_report": slice_report,
        "thresholds": threshold_by_tier,
        "cross_model": _cross_model_table(b0, tiers_out),
    }

    results_dir = root / "evals" / "results"
    out_path = results_dir / f"phase8_{date.today()}.json"
    _write(out_path, out)
    print(json.dumps(out["cross_model"], ensure_ascii=False, indent=2))
    print(f"\nwritten -> {out_path} ({elapsed}s)")
    return 0


def _compact_baseline(blob: dict | None) -> dict | None:
    if not blob:
        return None
    return {
        "elapsed_s": blob.get("elapsed_s"),
        "summary": blob.get("summary"),
        "by_subset": blob.get("by_subset"),
        "details": blob.get("details"),
    }


def _cross_model_table(b0: dict, tiers_out: dict) -> dict:
    table = {
        "M1_B0_recall_at_5": {
            "n": b0.get("n"),
            "hits": b0.get("hits"),
            "rate": b0.get("recall_at_k"),
            "note": "retrieval-only; same number for every MODEL_TIER",
        },
        "tiers": {},
    }
    for tier, blob in tiers_out.items():
        b1 = blob.get("B1") or {}
        b3 = blob.get("B3") or {}
        table["tiers"][tier] = {
            "resolved": blob.get("resolved"),
            "B1_keyword": (b1.get("summary") or {}),
            "B1_M3_E1": _metric_from_details(b1.get("details") or [], {"E1"}),
            "B1_M5_E2": _metric_from_details(b1.get("details") or [], {"E2a", "E2b"}),
            "B3_keyword": (b3.get("summary") or {}),
        }
    return table


if __name__ == "__main__":
    raise SystemExit(main())
