"""M1 vector retrieval scoring.

Official metric is labeled Recall@k against
`evals/reference_tables/e1_retrieval_gold.jsonl`.

Each E1 item has one or more **groups**. A group is a set of equivalent
chunk ids (`any_of`): hitting any one counts. Constitution × season items
have two groups (体质专章 + 节气表). Derived crossover sentences are not gold.

strict / union keyword columns stay as diagnostics. They are not M1.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable

from evals.phase8_scoring import load_jsonl, threshold_band

Retriever = Callable[[dict, int], list[dict]]

_WS = re.compile(r"\s+")
GOLD_RELATIVE = Path("evals") / "reference_tables" / "e1_retrieval_gold.jsonl"


def fold_whitespace(text: str) -> str:
    """Collapse whitespace so a line-broken CJK compound still matches."""
    return _WS.sub("", text or "")


def keywords_hit(text: str, keywords: Iterable[str]) -> bool:
    """B0 rubric: one raw chunk contains every expected substring."""
    text = text or ""
    return all(kw in text for kw in keywords)


def union_keyword_coverage(chunks: list[dict], keywords: list[str]) -> dict[str, Any]:
    blob = fold_whitespace("".join((c.get("text") or "") for c in chunks))
    missing = [kw for kw in keywords if fold_whitespace(kw) not in blob]
    n = len(keywords)
    coverage = (n - len(missing)) / n if n else 1.0
    return {
        "union_pass": not missing,
        "missing": missing,
        "coverage": coverage,
    }


def e1_items(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("subset") != "E1":
            continue
        kws = (row.get("expect") or {}).get("retrieval_keywords")
        if kws:
            out.append(row)
    return out


def load_retrieval_gold(path: Path | None = None) -> dict[str, dict]:
    rows = load_jsonl(path or GOLD_RELATIVE)
    return {str(row["id"]): row for row in rows if row.get("id")}


def score_gold_groups(retrieved_ids: Iterable[str], groups: list[dict]) -> dict[str, Any]:
    """Recall over groups: any_of inside a group is an OR."""
    got = {str(x) for x in retrieved_ids if x}
    scored = []
    for group in groups or []:
        any_of = [str(x) for x in (group.get("any_of") or [])]
        found = [cid for cid in any_of if cid in got]
        scored.append(
            {
                "role": group.get("role"),
                "any_of": any_of,
                "why": group.get("why"),
                "found": found,
                "pass": bool(found),
            }
        )
    n = len(scored)
    hits = sum(1 for g in scored if g["pass"])
    recall = (hits / n) if n else 1.0
    return {
        "groups": scored,
        "group_hits": hits,
        "n_groups": n,
        "recall": recall,
        "all_hit": bool(n == 0 or hits == n),
        "missing_roles": [g["role"] for g in scored if not g["pass"]],
    }


def score_retrieval(
    items: list[dict],
    retrieve: Retriever,
    *,
    k: int = 5,
    gold_by_id: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """retrieve(item, k) -> list of dicts with at least `text`.

    When gold_by_id is set, official columns are gold Recall@k.
    """
    details = []
    strict_hits = 0
    union_hits = 0
    coverage_sum = 0.0
    gold_recall_sum = 0.0
    gold_all_hits = 0
    gold_n = 0
    for row in items:
        kws = list((row.get("expect") or {}).get("retrieval_keywords") or [])
        if not kws:
            continue
        chunks = retrieve(row, k) or []
        strict = any(keywords_hit(c.get("text") or "", kws) for c in chunks)
        union = union_keyword_coverage(chunks, kws)
        strict_hits += int(strict)
        union_hits += int(union["union_pass"])
        coverage_sum += union["coverage"]
        retrieved_ids = [c.get("source_id") or c.get("chunk_id") for c in chunks]
        gold_row = (gold_by_id or {}).get(row.get("id") or "")
        gold = None
        if gold_row:
            gold = score_gold_groups(retrieved_ids, gold_row.get("groups") or [])
            gold_recall_sum += gold["recall"]
            gold_all_hits += int(gold["all_hit"])
            gold_n += 1
        details.append(
            {
                "id": row.get("id"),
                "query": row.get("query"),
                "domain": row.get("domain"),
                "expect": kws,
                "strict_pass": strict,
                "union_pass": union["union_pass"],
                "union_missing": union["missing"],
                "union_coverage": union["coverage"],
                "gold": gold,
                "retrieval_gt": (gold_row or {}).get("retrieval_gt"),
                "context": [
                    {
                        "source_id": c.get("source_id") or c.get("chunk_id"),
                        "domain": c.get("domain"),
                        "score": round(float(c.get("score") or 0), 4),
                        "text": c.get("text") or "",
                    }
                    for c in chunks
                ],
            }
        )
    n = len(details)
    union_rate = (union_hits / n) if n else None
    strict_rate = (strict_hits / n) if n else None
    gold_recall = (gold_recall_sum / gold_n) if gold_n else None
    gold_hit_rate = (gold_all_hits / gold_n) if gold_n else None
    official = gold_recall if gold_n else union_rate
    return {
        "k": k,
        "n": n,
        "strict_hits": strict_hits,
        "strict_recall": strict_rate,
        "union_hits": union_hits,
        "union_recall": union_rate,
        "mean_union_coverage": round(coverage_sum / n, 3) if n else None,
        "gold_n": gold_n,
        "gold_all_hits": gold_all_hits,
        "gold_hit_rate": gold_hit_rate,
        "recall_at_k": gold_recall,
        "band": threshold_band("M1", official),
        "misses": [
            d
            for d in details
            if (d.get("gold") and not d["gold"]["all_hit"])
            or (not d.get("gold") and not d["union_pass"])
        ],
        "details": details,
    }


# Back-compat alias.
score_recall_at_k = score_retrieval
