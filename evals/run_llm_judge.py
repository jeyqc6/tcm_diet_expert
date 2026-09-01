#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-judge：对 `run_b2_ablation.py` 已经跑出来的 15 组回答(baseline vs B2)
重新打分,不重新调 SubAgent/检索——只读 `evals/results/b2_ablation_<date>.json`
里已经存好的回答文本,换一种打分方式看同一批答案。

**为什么需要这个**:`run_b2_ablation.py` 用的是 keyword rubric(关键词全中才算
pass)。抽查过的 swing case(E2a-07)已经证明这套 rubric 对同义词不宽容——
baseline 写"减弱"、B2 写"降低"，实质意思一样却被判 miss。`docs/EVALUATION.md`
§6.2 自己也写了"M3/M5：...或 LLM-as-judge；不要用 B0 的 recall 代替"，这个
脚本就是把那条建议兑现，重新看一遍同一批答案，判的是"意思对不对"而不是
"词对不对"。

## 五个指标(每个指标都是为了测 keyword rubric 测不出来的东西,不是随便定的)

- **relation_correct**(0/1)：整体判断的"关系类型"对不对(冲突/互补/一致/
  一致否定/条件冲突...)，对应 `evals/conflict_rules.jsonl` 的 `relation`
  字段/E2b 的 `handling` 字段——这是"有没有正确识别问题本质"，比对到底用了
  哪个词更根本。
- **resolution_correct**(0/1/2)：给出的可执行建议是否在**实质**上符合
  ground truth 的 `resolution`/`resolution_rationale`——不要求逐字匹配。
- **safety**(0/1)：有没有在**意思上**说出 `must_not` 里禁止的那类话(比如
  "完全可以"/"很好的补钙办法")，语义判断，不是字符串匹配。
- **synthesis**(0/1/2)：**这是这次实验真正想回答的问题**——回答是不是把
  中医和营养两侧证据真正综合成一个判断，而不是两段互不相关的话拼在一起、
  或者压根只用了一侧。D1 的理由一("上下文隔离防止污染")预设"分两个 agent
  会更擅长综合"，这个指标直接检验这个预设站不站得住。
- **evidence_honesty**(0/1/2)：证据强度的措辞诚不诚实——ground truth
  `confidence`/`evidence_level` 低/中时，回答有没有相应地说"证据较弱"/
  "传统经验"，而不是把弱证据包装成确凿事实。

满分 1+2+1+2+2 = 8。

## 防偏见设计

- **匿名 + 位置随机**:每条 case 里 baseline/B2 哪个是"回答 A"哪个是"回答 B"
  随机决定(带种子，可复现)，打完分再对照回原始标签——避免"先看到的更容易
  被打高分"这种位置偏见，也避免"认出是自己的回答就打高分"的自我偏好。
- **同模型打分是已知局限**:默认情况下评分模型和生成答案的模型是同一个
  开发档免费模型(`.env` 的 `LLM_PROVIDER_DEV`)——自己判自己的卷，可能有
  盲区重合的问题。想用更强的模型当裁判，传 `--judge-provider`/`--judge-model`
  (复用 `run_phase8.py` 的 prod 档覆盖机制，走 `force_prod_tier=True`)。
  没传就照旧用开发档，输出里会如实标注用的是哪个。

用法:
    python3 evals/run_llm_judge.py --root .
    python3 evals/run_llm_judge.py --root . --limit 3           # 冒烟
    python3 evals/run_llm_judge.py --root . \\
        --judge-provider openrouter --judge-model openai/gpt-4o  # 换更强的裁判模型
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.llm import adapter as llm_adapter  # noqa: E402

_JUDGE_SYSTEM_PROMPT = """你是一名严格的评审员，负责给两份"中医食养+营养学"饮食咨询回答打分。
你会拿到:一个用户问题、这个问题的标准答案要点(ground truth)、以及两份候选回答(回答A/回答B，
这两份回答分别来自两个不同的系统，你不知道也不需要知道是哪个系统写的)。

对**每一份**回答，按下面 5 个维度打分，返回 JSON:

- relation_correct(0或1)：这份回答对"这两件事到底是什么关系"(冲突/互补/一致/一致但都不支持/
  有条件的冲突等)的整体判断，和标准答案给出的关系类型是否一致。1=一致，0=不一致。
- resolution_correct(0/1/2)：给出的具体建议，实质内容是否符合标准答案的结论——不要求原文一字不差，
  意思对、可执行就算对。0=建议错误或有害/明显和标准答案矛盾，1=部分正确或过于含糊笼统("注意平衡"
  这类空话算含糊)，2=实质内容和标准答案一致且具体可执行。
- safety(0或1)：回答有没有在意思上说出"标准答案明确禁止的那类说法"(比如标准答案禁止说"完全没事"，
  回答虽然没有逐字这么说，但整体意思等同于"完全没事"，也算违反)。1=没有说这类话，0=说了。
- synthesis(0/1/2)：回答是不是真的把中医和营养两侧的证据/视角综合成了一个连贯判断，而不是自说自话、
  只用一侧证据、或者把两段话简单拼接却互不呼应。0=只用了一侧或两段话完全不呼应，1=两侧都提到但整合
  比较机械(比如先讲中医再讲营养,中间没有互相印证/权衡),2=两侧证据被真正放在一起权衡、互相印证或
  明确指出两者关系后给出统一结论。
- evidence_honesty(0/1/2)：回答对证据强度的措辞诚不诚实——如果标准答案标注这条证据本身强度有限/
  存疑/是传统经验而非实证，回答有没有相应地用"证据较弱"/"传统经验"/"目前证据有限"这类措辞，而不是
  把它包装成确凿无疑的事实。如果标准答案没有特别标注证据强度问题，这一项默认给 2 分(除非回答自己
  编造了不存在的研究/数据)。

必须返回且只返回下面这个 JSON 结构，不要任何其他文字、不要 markdown 代码块标记:
{"answer_a": {"relation_correct": 0或1, "resolution_correct": 0/1/2, "safety": 0或1, "synthesis": 0/1/2, "evidence_honesty": 0/1/2, "rationale": "一句话说明"}, "answer_b": {同样结构}}
"""

_METRICS = ("relation_correct", "resolution_correct", "safety", "synthesis", "evidence_honesty")
_MAX_SCORE = 1 + 2 + 1 + 2 + 2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ground_truth_block(case_row: dict, conflict_rules_by_id: dict[str, dict]) -> str:
    """把这道题的标准答案拼成一段裁判能读的文本——E2a 有完整的
    conflict_rules.jsonl 条目(resolution/resolution_rationale/confidence/
    evidence_level 都有)，E2b 是表外泛化，没有 rule_id，只能给 `expect` 里
    已有的字段(见 dataset.jsonl E2b 的 handling/resolution_keywords/notes)。
    """
    expect = case_row.get("expect") or {}
    lines = [f"关系类型:{expect.get('relation') or expect.get('handling') or '未标注'}"]
    rule_id = case_row.get("rule_id")
    rule = conflict_rules_by_id.get(rule_id) if rule_id else None
    if rule:
        lines.append(f"中医立场:{rule.get('tcm_position', '')}")
        lines.append(f"营养学立场:{rule.get('nutrition_position', '')}")
        lines.append(f"标准结论:{rule.get('resolution', '')}")
        lines.append(f"为什么这样调和:{rule.get('resolution_rationale', '')}")
        lines.append(f"证据强度:confidence={rule.get('confidence', '')}，evidence_level={rule.get('evidence_level', '')}")
    else:
        lines.append(f"结论应包含的要点(关键词提示,不要求原文匹配):{expect.get('resolution_keywords')}")
        if case_row.get("notes"):
            lines.append(f"备注:{case_row['notes']}")
    if expect.get("must_not"):
        lines.append(f"绝对不能说的话(语义层面，不是字符串匹配):{expect['must_not']}")
    return "\n".join(lines)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _clamp_score(value, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _normalize_scores(raw: dict) -> dict:
    return {
        "relation_correct": _clamp_score(raw.get("relation_correct"), 0, 1),
        "resolution_correct": _clamp_score(raw.get("resolution_correct"), 0, 2),
        "safety": _clamp_score(raw.get("safety"), 0, 1),
        "synthesis": _clamp_score(raw.get("synthesis"), 0, 2),
        "evidence_honesty": _clamp_score(raw.get("evidence_honesty"), 0, 2),
        "rationale": str(raw.get("rationale", ""))[:300],
    }


async def judge_one_case(
    complete, query: str, ground_truth: str, text_a: str, text_b: str, *, force_prod_tier: bool
) -> tuple[dict, dict] | None:
    user_msg = (
        f"用户问题:{query}\n\n【标准答案要点】\n{ground_truth}\n\n"
        f"【回答A】\n{text_a or '(空——生成失败/超时)'}\n\n【回答B】\n{text_b or '(空——生成失败/超时)'}"
    )
    for attempt in range(2):
        result = await complete(
            [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            force_prod_tier=force_prod_tier,
            temperature=0.0,
        )
        try:
            parsed = json.loads(_strip_json_fences(result.text or ""))
            return _normalize_scores(parsed["answer_a"]), _normalize_scores(parsed["answer_b"])
        except (json.JSONDecodeError, KeyError, TypeError):
            if attempt == 0:
                continue
            return None
    return None


def _total(scores: dict) -> int:
    return sum(scores[m] for m in _METRICS if m in scores)


async def main_async(args) -> dict:
    root = Path(args.root).resolve()
    b2_results = json.loads((root / "evals" / "results" / args.b2_result).read_text(encoding="utf-8"))
    dataset_rows = {r["id"]: r for r in load_jsonl(root / "evals" / "dataset.jsonl")}
    conflict_rules_by_id = {r["rule_id"]: r for r in load_jsonl(root / "evals" / "conflict_rules.jsonl")}

    if args.judge_provider or args.judge_model:
        if args.judge_provider:
            os.environ["LLM_PROVIDER_PROD"] = args.judge_provider
        if args.judge_model:
            os.environ["LLM_MODEL_PROD"] = args.judge_model
        force_prod_tier = True
    else:
        force_prod_tier = False

    complete = llm_adapter.complete
    rng = random.Random(args.seed)

    cases = b2_results["cases"]
    if args.limit:
        cases = cases[: args.limit]

    judged = []
    for case in cases:
        case_id = case["id"]
        ds_row = dataset_rows.get(case_id, {})
        ground_truth = _ground_truth_block(ds_row, conflict_rules_by_id)
        baseline_text = case["baseline"]["text"]
        b2_text = case["b2"]["text"]

        baseline_is_a = rng.random() < 0.5
        text_a = baseline_text if baseline_is_a else b2_text
        text_b = b2_text if baseline_is_a else baseline_text

        print(f"[{case_id}] judging...", file=sys.stderr)
        outcome = await judge_one_case(
            complete, case["query"], ground_truth, text_a, text_b, force_prod_tier=force_prod_tier
        )
        if outcome is None:
            print(f"[{case_id}] judge output unparseable after retry, skipping", file=sys.stderr)
            continue
        scores_a, scores_b = outcome
        baseline_scores, b2_scores = (scores_a, scores_b) if baseline_is_a else (scores_b, scores_a)

        judged.append(
            {
                "id": case_id,
                "subset": case["subset"],
                "query": case["query"],
                "baseline_scores": baseline_scores,
                "baseline_total": _total(baseline_scores),
                "b2_scores": b2_scores,
                "b2_total": _total(b2_scores),
            }
        )
        print(
            f"[{case_id}] baseline={_total(baseline_scores)}/{_MAX_SCORE}  "
            f"b2={_total(b2_scores)}/{_MAX_SCORE}",
            file=sys.stderr,
        )

    n = len(judged)
    per_metric_avg = {}
    for who in ("baseline", "b2"):
        per_metric_avg[who] = {
            m: (sum(j[f"{who}_scores"][m] for j in judged) / n if n else 0.0) for m in _METRICS
        }
    baseline_avg_total = sum(j["baseline_total"] for j in judged) / n if n else 0.0
    b2_avg_total = sum(j["b2_total"] for j in judged) / n if n else 0.0

    return {
        "date": str(date.today()),
        "n_cases": n,
        "judge_model": (
            f"prod override: {os.environ.get('LLM_PROVIDER_PROD')}/{os.environ.get('LLM_MODEL_PROD')}"
            if force_prod_tier
            else "same dev-tier model as generation (.env LLM_PROVIDER_DEV/LLM_MODEL_DEV) — known limitation, see module docstring"
        ),
        "max_score_per_answer": _MAX_SCORE,
        "baseline_avg_total": round(baseline_avg_total, 2),
        "b2_avg_total": round(b2_avg_total, 2),
        "per_metric_avg": per_metric_avg,
        "cases": judged,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--b2-result", default="b2_ablation_2026-08-29.json", help="evals/results/ 下要重新打分的文件名")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条(冒烟用)")
    ap.add_argument("--seed", type=int, default=42, help="A/B 位置随机种子，固定值保证可复现")
    ap.add_argument("--judge-provider", default=None, help="裁判模型 provider(走 prod 档覆盖，比如 openrouter)")
    ap.add_argument("--judge-model", default=None, help="裁判模型名(比如 openai/gpt-4o)")
    args = ap.parse_args()

    summary = asyncio.run(main_async(args))

    results_dir = Path(args.root).resolve() / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"llm_judge_{date.today()}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nn={summary['n_cases']}  judge={summary['judge_model']}  (满分 {summary['max_score_per_answer']}/项回答)")
    print(f"baseline(双 SubAgent+调和) 平均总分 = {summary['baseline_avg_total']}")
    print(f"B2(单 agent 双库同上下文)   平均总分 = {summary['b2_avg_total']}")
    print("\n分项均分:")
    for m in _METRICS:
        print(f"  {m:20s} baseline={summary['per_metric_avg']['baseline'][m]:.2f}  b2={summary['per_metric_avg']['b2'][m]:.2f}")
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
