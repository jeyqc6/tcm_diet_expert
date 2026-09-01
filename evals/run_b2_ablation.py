#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 8 · B2 ablation(docs/DECISIONS.md D1"验证方式")。

对比:
  baseline = 当前架构:TCM SubAgent + Nutrition SubAgent(各自独立上下文)并行
             检索 → `reconcile_subagent_results()` 合并成一段结论
  B2       = `backend/agents/single_agent_baseline.py`:单 agent、两个检索
             工具同一上下文，一次调用直接产出结论

跑分对象:`evals/dataset.jsonl` 里的 E2a(冲突调和,10 条)+ E2b(表外泛化,5 条)
= 15 条——这是唯一真正需要"两侧领域证据放在一起给结论"的子集(M5 冲突调和
正确率)。E1(纯事实检索,不需要跨域)/E3(多轮记忆,跟拆不拆上下文无关)不放进
这次对比，放进去只会稀释信号。

打分:复用 `run_baselines.py` 里 B1/B3 已经用的同一套 keyword rubric——
`resolution_keywords` 全部出现且 `must_not` 一个都不出现才算 pass。

比较的是**调和前**的 baseline 文本(`reconciled.text`，未经过 `verify()`)和
B2 的 `final_text`——两条路径后面都还会经过同一个 `verify()` 核查 pass，
citation/allergen 拦截逻辑对两侧一视同仁不是这次要检验的变量，在 `verify()`
之前打分，避免核查通过率的噪声掩盖"两个上下文架构本身谁的结论内容更对"
这个真正要回答的问题。

⚠️ 诚实说明:这次跑分用的是 `.env` 里配置的开发档模型
(`LLM_PROVIDER_DEV`/`LLM_MODEL_DEV`)，不是 PRD §12.4 要求的交付档模型——
这是一次架构选择的消融实验，不是最终交付数字。免费/低算力模型可能让两种
架构的绝对分数都偏低，报告里如实标注，不假装成交付档结论。

用法:
    python3 evals/run_b2_ablation.py --root .
    python3 evals/run_b2_ablation.py --root . --limit 4        # 先冒烟
    SUBAGENT_TIMEOUT_S=90 python3 evals/run_b2_ablation.py --root .  # 免费模型慢，放宽超时
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.dispatch import _gather_dual_subagents  # noqa: E402
from backend.agents.reconciliation import reconcile_subagent_results  # noqa: E402
from backend.agents.single_agent_baseline import run_single_agent_b2  # noqa: E402
from backend.exceptions import SubAgentTimeoutError  # noqa: E402
from backend.llm import adapter as llm_adapter  # noqa: E402
from backend.mcp_server.server import DietExpertMcpServer  # noqa: E402

# 显著性判定：pass_rate 差距在这个阈值以内算"没有显著优势"(D1"验证方式"要求
# 的判断标准需要一个具体数字才能自动化，不能事后靠感觉)。15 条小样本下，
# 差 1 条就是 6.7 个百分点——这条阈值本身就是"至少要赢 2 条"的粗略换算。
SIGNIFICANCE_THRESHOLD = 0.10


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(text: str, expect: dict) -> tuple[bool, list[str], list[str]]:
    must = expect.get("resolution_keywords") or []
    must_not = expect.get("must_not") or []
    missing = [k for k in must if k not in text]
    hit_forbidden = [k for k in must_not if k in text]
    return (not missing and not hit_forbidden), missing, hit_forbidden


async def run_baseline_case(server, complete, query: str) -> tuple[str, dict]:
    t0 = time.perf_counter()
    try:
        tcm_result, nutrition_result = await _gather_dual_subagents(
            query, server, complete,
            constitution=None, allergens=(), extra_notes="", include_recipe=False,
        )
    except SubAgentTimeoutError as exc:
        return "", {"error": f"timeout: {exc}", "latency_s": round(time.perf_counter() - t0, 1)}
    if isinstance(tcm_result, BaseException) or isinstance(nutrition_result, BaseException):
        return "", {
            "error": f"subagent_failed tcm={tcm_result!r} nutrition={nutrition_result!r}",
            "latency_s": round(time.perf_counter() - t0, 1),
        }
    reconciled = await reconcile_subagent_results(tcm_result, nutrition_result, complete=complete)
    latency = time.perf_counter() - t0
    meta = {
        "latency_s": round(latency, 1),
        "tcm_tool_calls": tcm_result.tool_call_count,
        "nutrition_tool_calls": nutrition_result.tool_call_count,
        "tcm_tools_called": tcm_result.tools_called,
        "nutrition_tools_called": nutrition_result.tools_called,
        "llm_calls": tcm_result.iterations + nutrition_result.iterations + 1,  # +1 = reconcile
    }
    return reconciled.text, meta


async def run_b2_case(server, complete, query: str) -> tuple[str, dict]:
    t0 = time.perf_counter()
    try:
        result = await run_single_agent_b2(query, server, constitution=None, allergens=(), complete=complete)
    except (SubAgentTimeoutError, Exception) as exc:  # noqa: BLE001 — 消融实验，任何一侧失败都记下来继续跑
        return "", {"error": repr(exc), "latency_s": round(time.perf_counter() - t0, 1)}
    latency = time.perf_counter() - t0
    meta = {
        "latency_s": round(latency, 1),
        "tool_calls": result.tool_call_count,
        "tools_called": result.tools_called,
        "llm_calls": result.iterations,
        "used_both_domains": (
            "retrieve_tcm" in result.tools_called and "retrieve_nutrition" in result.tools_called
        ),
    }
    return result.final_text, meta


async def main_async(args) -> dict:
    root = Path(args.root).resolve()
    rows = load_jsonl(root / "evals" / "dataset.jsonl")
    cases = [r for r in rows if r.get("subset") in ("E2a", "E2b")]
    if args.limit:
        cases = cases[: args.limit]

    server = DietExpertMcpServer()
    complete = llm_adapter.complete

    results = []
    for case in cases:
        query = case["query"]
        expect = case.get("expect") or {}
        print(f"[{case['id']}] baseline...", file=sys.stderr)
        baseline_text, baseline_meta = await run_baseline_case(server, complete, query)
        print(f"[{case['id']}] b2...", file=sys.stderr)
        b2_text, b2_meta = await run_b2_case(server, complete, query)

        b_pass, b_missing, b_forbidden = score(baseline_text, expect)
        s_pass, s_missing, s_forbidden = score(b2_text, expect)

        results.append(
            {
                "id": case["id"],
                "subset": case["subset"],
                "query": query,
                "expect": expect,
                "baseline": {
                    "pass": b_pass, "missing": b_missing, "hit_forbidden": b_forbidden,
                    "text": baseline_text, **baseline_meta,
                },
                "b2": {
                    "pass": s_pass, "missing": s_missing, "hit_forbidden": s_forbidden,
                    "text": b2_text, **b2_meta,
                },
            }
        )
        print(
            f"[{case['id']}] baseline={'PASS' if b_pass else 'FAIL'} "
            f"b2={'PASS' if s_pass else 'FAIL'}",
            file=sys.stderr,
        )

    n = len(results)
    baseline_passed = sum(1 for r in results if r["baseline"]["pass"])
    b2_passed = sum(1 for r in results if r["b2"]["pass"])
    baseline_latency = sum(r["baseline"].get("latency_s", 0) for r in results)
    b2_latency = sum(r["b2"].get("latency_s", 0) for r in results)
    baseline_llm_calls = sum(r["baseline"].get("llm_calls", 0) for r in results)
    b2_llm_calls = sum(r["b2"].get("llm_calls", 0) for r in results)

    baseline_rate = baseline_passed / n if n else 0.0
    b2_rate = b2_passed / n if n else 0.0
    gap = baseline_rate - b2_rate  # 正数 = baseline(双agent)领先

    if gap > SIGNIFICANCE_THRESHOLD:
        verdict = "baseline(双 SubAgent+调和)显著优于 B2 —— 保留 D1 现状，不追加修订"
    elif gap < -SIGNIFICANCE_THRESHOLD:
        verdict = "B2(单 agent)反而显著优于 baseline —— 需要在 D1 追加修订记录并考虑回退"
    else:
        verdict = (
            f"两者差距(|{gap:+.1%}|)未超过显著性阈值({SIGNIFICANCE_THRESHOLD:.0%}) —— "
            "无显著优势，按 D1 约定在 DECISIONS.md 追加修订记录并回退至单 agent"
        )

    summary = {
        "date": str(date.today()),
        "n_cases": n,
        "model_tier": "dev (.env LLM_PROVIDER_DEV/LLM_MODEL_DEV — NOT delivery-tier, see PRD §12.4)",
        "significance_threshold": SIGNIFICANCE_THRESHOLD,
        "baseline_pass_rate": baseline_rate,
        "b2_pass_rate": b2_rate,
        "gap_baseline_minus_b2": gap,
        "baseline_total_latency_s": round(baseline_latency, 1),
        "b2_total_latency_s": round(b2_latency, 1),
        "baseline_total_llm_calls": baseline_llm_calls,
        "b2_total_llm_calls": b2_llm_calls,
        "verdict": verdict,
        "cases": results,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="项目根目录")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条(冒烟用)")
    args = ap.parse_args()

    summary = asyncio.run(main_async(args))

    results_dir = Path(args.root).resolve() / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"b2_ablation_{date.today()}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nn={summary['n_cases']} (E2a+E2b)")
    print(
        f"baseline(双 SubAgent+调和): pass_rate={summary['baseline_pass_rate']:.1%}  "
        f"latency={summary['baseline_total_latency_s']}s  llm_calls={summary['baseline_total_llm_calls']}"
    )
    print(
        f"B2(单 agent 双库同上下文):   pass_rate={summary['b2_pass_rate']:.1%}  "
        f"latency={summary['b2_total_latency_s']}s  llm_calls={summary['b2_total_llm_calls']}"
    )
    print(f"\n结论: {summary['verdict']}")
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
