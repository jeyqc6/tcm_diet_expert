#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8 LLM-as-judge: re-score already generated B1/B3 answers.

Does not call the chat pipeline. Reads:
    evals/results/{dev,prod}_{B1,B3}.partial.json
and writes:
    evals/results/phase8_judge_<date>.json

Default judge is the free dev-tier model (.env LLM_PROVIDER_DEV / LLM_MODEL_DEV).

Usage:
    python3 evals/run_phase8_judge.py --root .
    python3 evals/run_phase8_judge.py --root . --arms dev_B1,prod_B1
    python3 evals/run_phase8_judge.py --root . --limit 3
    python3 evals/run_phase8_judge.py --root . --assemble
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("LANGFUSE_ENABLED", "0")

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(ROOT_DEFAULT))

from evals.phase8_judge import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    build_ground_truth,
    judge_user_message,
    load_arm_answers,
    normalize_judge_scores,
    semantic_pass,
    strip_json_fences,
    summarize_arm,
    threshold_table,
)
from evals.phase8_scoring import load_jsonl, slices_for_item  # noqa: E402

DEFAULT_ARMS = ("dev_B1", "prod_B1", "dev_B3", "prod_B3")
ARM_FILES = {
    "dev_B1": "dev_B1.partial.json",
    "prod_B1": "prod_B1.partial.json",
    "dev_B3": "dev_B3.partial.json",
    "prod_B3": "prod_B3.partial.json",
}


def _parse_json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(strip_json_fences(text))
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def judge_one(complete, user_msg: str) -> dict | None:
    for attempt in range(2):
        result = await complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        parsed = _parse_json_object(result.text or "")
        if parsed is not None:
            return normalize_judge_scores(parsed)
        if attempt == 0:
            continue
    return None


def _checkpoint_path(results_dir: Path, arm: str) -> Path:
    return results_dir / f"phase8_judge_{arm}.partial.json"


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(rows, list):
        return {}
    return {r["id"]: r for r in rows if isinstance(r, dict) and r.get("id")}


async def run_arm(
    *,
    arm: str,
    dataset: list[dict],
    answers: dict[str, dict],
    constitution_rows: list[dict],
    conflict_rules: list[dict],
    complete,
    pause_s: float,
    limit: int | None,
    checkpoint: Path,
) -> list[dict]:
    prior = _load_checkpoint(checkpoint)
    out: list[dict] = []
    items = dataset[:limit] if limit else dataset
    for i, case in enumerate(items):
        item_id = case["id"]
        if item_id in prior and prior[item_id].get("scored"):
            out.append(prior[item_id])
            print(f"  [{arm}] {i + 1}/{len(items)} {item_id} resume", flush=True)
            continue
        answer = answers.get(item_id) or {
            "text": "",
            "truncated": False,
            "keyword_pass": False,
            "error": "missing_answer",
        }
        gt = build_ground_truth(
            case, constitution_rows=constitution_rows, conflict_rules=conflict_rules
        )
        user_msg = judge_user_message(gt, answer)
        scores = None
        err = answer.get("error")
        try:
            scores = await judge_one(complete, user_msg)
        except Exception as exc:  # noqa: BLE001 — isolate per item
            err = f"{type(exc).__name__}: {exc}"
        scored = scores is not None
        row = {
            "id": item_id,
            "subset": case.get("subset"),
            "slices": slices_for_item(case),
            "truncated": bool(answer.get("truncated")),
            "keyword_pass": bool(answer.get("keyword_pass")),
            "scored": scored,
            "semantic_pass": bool(scores and semantic_pass(scores)),
            "scores": scores,
            "error": err if not scored else None,
        }
        out.append(row)
        mark = "PASS" if row["semantic_pass"] else ("FAIL" if scored else "SKIP")
        print(f"  [{arm}] {i + 1}/{len(items)} {item_id} {mark}", flush=True)
        checkpoint.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        if pause_s and i + 1 < len(items):
            await asyncio.sleep(pause_s)
    return out


def assemble(results_dir: Path, arms: list[str]) -> dict:
    blob: dict[str, Any] = {
        "date": str(date.today()),
        "rubric": "semantic_pass = direction==2 and safety==1; paraphrase allowed",
        "limitations": [
            "Answers were stored as 400-char previews; truncated items are flagged.",
            "Judge is the free dev-tier model unless --judge-provider is set.",
            "This is not /api/chat. B0 retrieval is unchanged (no LLM).",
        ],
        "arms": {},
    }
    for arm in arms:
        path = _checkpoint_path(results_dir, arm)
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        summary = summarize_arm(rows)
        blob["arms"][arm] = {
            "summary": summary,
            "thresholds": threshold_table(summary),
            "n_rows": len(rows),
        }
    return blob


async def main_async(args) -> dict:
    root = Path(args.root).resolve()
    results_dir = root / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.assemble:
        return assemble(results_dir, arms)

    dataset = load_jsonl(root / "evals" / "dataset.jsonl")
    constitution_rows = load_jsonl(root / "evals" / "reference_tables" / "constitution_season.jsonl")
    conflict_rules = load_jsonl(root / "evals" / "conflict_rules.jsonl")

    os.environ.setdefault("MODEL_TIER", "dev")
    # `load_env()` only overlays `.env` onto os.environ once per process (see
    # backend/env.py); it's a no-op on every later call. Must run BEFORE the
    # CLI override below — otherwise the *first* load_env() call happens
    # inside _provider_name_for_tier()/_model_for_tier() further down, and it
    # unconditionally overlays LLM_PROVIDER_PROD/LLM_MODEL_PROD from the file,
    # silently clobbering --judge-provider/--judge-model back to whatever
    # .env's LLM_PROVIDER_PROD says (e.g. "deepseek").
    from backend.env import load_env

    load_env()
    if args.judge_provider or args.judge_model:
        if args.judge_provider:
            os.environ["LLM_PROVIDER_PROD"] = args.judge_provider
        if args.judge_model:
            os.environ["LLM_MODEL_PROD"] = args.judge_model
        os.environ["MODEL_TIER"] = "prod"
    from backend.llm import adapter as llm_adapter
    from backend.llm.adapter import ModelTier, _model_for_tier, _provider_name_for_tier

    complete = llm_adapter.complete
    mt = ModelTier.PROD if os.environ.get("MODEL_TIER") == "prod" else ModelTier.DEV
    provider = _provider_name_for_tier(mt)
    model = _model_for_tier(mt, provider)
    judge_label = f"{provider}/{model} (tier={mt.value})"

    t0 = time.perf_counter()
    for arm in arms:
        fname = ARM_FILES.get(arm)
        if fname is None:
            print(f"unknown arm {arm}", file=sys.stderr)
            continue
        path = results_dir / fname
        if not path.exists():
            print(f"missing {path}, skip {arm}", file=sys.stderr)
            continue
        answers = load_arm_answers(json.loads(path.read_text(encoding="utf-8")))
        print(f"judging {arm} ({len(answers)} stored answers) with {judge_label}", flush=True)
        await run_arm(
            arm=arm,
            dataset=dataset,
            answers=answers,
            constitution_rows=constitution_rows,
            conflict_rules=conflict_rules,
            complete=complete,
            pause_s=args.pause,
            limit=args.limit,
            checkpoint=_checkpoint_path(results_dir, arm),
        )

    blob = assemble(results_dir, arms)
    blob["judge_model"] = judge_label
    blob["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between judge calls")
    ap.add_argument("--assemble", action="store_true", help="rebuild report from judge partials")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    summary = asyncio.run(main_async(args))
    out_path = Path(args.root).resolve() / "evals" / "results" / f"phase8_judge_{date.today()}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\njudge={summary.get('judge_model', 'assembled')}")
    for arm, body in (summary.get("arms") or {}).items():
        s = body["summary"]
        m3 = s["M3_E1"]["judge_rate"]
        m5 = s["M5_E2"]["judge_rate"]
        print(
            f"{arm:10s}  all={s['judge_rate']}  "
            f"M3={m3}  M5={m5}  keyword_all={s['keyword_rate']}"
        )
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
