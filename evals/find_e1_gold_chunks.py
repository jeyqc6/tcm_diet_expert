#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lexical corpus scan: does knowledge_chunks contain the E1 ground-truth cues?

This is NOT expert gold. It answers: for each frozen E1 item, which
chunk_ids (if any) mention the distinctive GT phrases, and whether those
ids showed up in the last vector top-5.

Usage:
    python3 evals/find_e1_gold_chunks.py --root .
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(ROOT_DEFAULT))

from evals.phase8_scoring import load_jsonl  # noqa: E402
from evals.vector_m1 import e1_items  # noqa: E402

# Distinctive phrases the derived GT / conflict rule actually needs.
# `required` missing => corpus cannot support that cue (not a retriever miss).
# `support` = related evidence that a human might still use as gold.
PROBES: dict[str, dict] = {
    "E1-01": {
        "required": [
            {"id": "少酸多甘", "terms": ["少酸多甘"]},
            {"id": "疏肝", "terms": ["疏肝"]},
        ],
        "support": [
            {"id": "平和质", "terms": ["平和质", "平和体质"]},
            {"id": "春季时令", "terms": ["早春", "春季饮食"]},
        ],
    },
    "E1-02": {
        "required": [
            {"id": "气虚", "terms": ["气虚"]},
            {"id": "山药", "terms": ["山药"]},
            {"id": "补气", "terms": ["补气", "补中益气"]},
        ],
        "support": [
            {"id": "疏肝", "terms": ["疏肝"]},
            {"id": "大枣", "terms": ["大枣"]},
        ],
    },
    "E1-03": {
        "required": [
            {"id": "阳虚", "terms": ["阳虚"]},
            {"id": "姜类", "terms": ["生姜", "干姜"]},
            {"id": "忌生冷", "terms": ["生冷"]},
        ],
        "support": [
            {"id": "倒春寒/早春", "terms": ["倒春寒", "早春", "乍暖还寒"]},
        ],
    },
    "E1-04": {
        "required": [
            {"id": "阳虚", "terms": ["阳虚"]},
            {"id": "生冷", "terms": ["生冷"]},
        ],
        "support": [
            {"id": "冰/冷饮", "terms": ["冰饮", "冷饮", "寒凉"]},
            {"id": "夏季", "terms": ["夏季", "夏天"]},
        ],
    },
    "E1-05": {
        "required": [
            {"id": "阳虚", "terms": ["阳虚"]},
            {"id": "羊肉", "terms": ["羊肉"]},
            {"id": "温补", "terms": ["温补", "温阳", "温补壮阳"]},
        ],
        "support": [
            {"id": "冬季", "terms": ["冬季食养", "冬季应增食", "冬"]},
        ],
    },
    "E1-06": {
        "required": [
            {"id": "阴虚", "terms": ["阴虚"]},
            {"id": "祛湿", "terms": ["祛湿", "利湿", "化湿"]},
        ],
        "support": [
            {"id": "长夏", "terms": ["长夏"]},
            {"id": "冬瓜", "terms": ["冬瓜"]},
            {"id": "薏米", "terms": ["薏米", "薏苡"]},
        ],
    },
    "E1-07": {
        "required": [
            {"id": "痰湿", "terms": ["痰湿"]},
            {"id": "薏米", "terms": ["薏米", "薏苡"]},
        ],
        "support": [
            {"id": "长夏/湿盛", "terms": ["长夏", "湿盛"]},
            {"id": "赤小豆", "terms": ["赤小豆"]},
        ],
    },
    "E1-08": {
        "required": [
            {"id": "湿热", "terms": ["湿热"]},
            {"id": "清热", "terms": ["清热"]},
        ],
        "support": [
            {"id": "利湿", "terms": ["利湿", "化湿"]},
            {"id": "冬瓜", "terms": ["冬瓜"]},
        ],
    },
    "E1-09": {
        "required": [
            {"id": "血瘀", "terms": ["血瘀"]},
            {"id": "活血", "terms": ["活血"]},
        ],
        "support": [
            {"id": "山楂", "terms": ["山楂"]},
            {"id": "寒凝", "terms": ["寒凝", "温经"]},
        ],
    },
    "E1-10": {
        "required": [
            {"id": "特禀/过敏", "terms": ["特禀", "过敏体质", "过敏"]},
        ],
        "support": [
            {"id": "发物", "terms": ["发物"]},
            {"id": "海鲜", "terms": ["海鲜"]},
        ],
    },
    "E1-11": {
        "required": [
            {"id": "红枣", "terms": ["红枣", "大枣"]},
            {"id": "非血红素铁", "terms": ["非血红素", "非血红素铁"]},
        ],
        "support": [
            {"id": "补血", "terms": ["补血"]},
        ],
    },
    "E1-12": {
        "required": [
            {"id": "蚝油", "terms": ["蚝油"]},
            {"id": "甲壳类", "terms": ["甲壳类", "甲壳"]},
        ],
        "support": [
            {"id": "贝", "terms": ["贝类", "软体"]},
        ],
    },
    "E1-13": {
        "required": [
            {"id": "微波", "terms": ["微波"]},
            {"id": "维生素保留", "terms": ["保留", "维生素"]},
        ],
        "support": [],
    },
    "E1-14": {
        "required": [
            {"id": "茶", "terms": ["茶"]},
            {"id": "铁", "terms": ["铁"]},
        ],
        "support": [
            {"id": "单宁/阻碍吸收", "terms": ["单宁", "阻碍", "吸收"]},
            {"id": "餐前后", "terms": ["餐前", "餐后", "1小时"]},
        ],
    },
    "E1-15": {
        "required": [
            {"id": "2025-2030", "terms": ["2025", "2030"]},
        ],
        "support": [
            {"id": "RealFood/DGA", "terms": ["RealFood", "Dietary Guidelines"]},
        ],
    },
}


def _fold_sql_expr(column: str = "text") -> str:
    # Collapse CJK line breaks so 山\\n药 still matches 山药.
    return f"replace(replace(replace({column}, E'\\n', ''), ' ', ''), E'\\t', '')"


def search_term(cur, term: str, domain: str | None, limit: int = 8) -> list[dict]:
    folded = _fold_sql_expr("text")
    needle = term.replace(" ", "").replace("\n", "")
    if domain:
        cur.execute(
            f"""
            SELECT chunk_id, domain, left(text, 180) AS snippet
            FROM knowledge_chunks
            WHERE embedding IS NOT NULL
              AND domain = %s
              AND {folded} LIKE %s
            LIMIT %s
            """,
            [domain, f"%{needle}%", limit],
        )
    else:
        cur.execute(
            f"""
            SELECT chunk_id, domain, left(text, 180) AS snippet
            FROM knowledge_chunks
            WHERE embedding IS NOT NULL
              AND {folded} LIKE %s
            LIMIT %s
            """,
            [f"%{needle}%", limit],
        )
    rows = cur.fetchall()
    return [
        {"chunk_id": r[0], "domain": r[1], "snippet": (r[2] or "").replace("\n", " ")}
        for r in rows
    ]


def _hits_for_group(cur, group: list[dict], domain: str) -> list[dict]:
    out = []
    for probe in group:
        found = []
        seen = set()
        for term in probe["terms"]:
            for hit in search_term(cur, term, domain):
                if hit["chunk_id"] in seen:
                    continue
                seen.add(hit["chunk_id"])
                found.append({**hit, "matched_term": term})
        out.append(
            {
                "id": probe["id"],
                "terms": probe["terms"],
                "n": len(found),
                "present": bool(found),
                "hits": found[:5],
            }
        )
    return out


def verdict(required: list[dict], support: list[dict]) -> str:
    req_ok = [p for p in required if p["present"]]
    if required and len(req_ok) == len(required):
        return "in_corpus"
    if req_ok or any(p["present"] for p in support):
        return "partial"
    return "absent"


def candidate_chunks(required: list[dict], support: list[dict], k: int = 5) -> list[dict]:
    """Rank chunk_ids by how many probes they satisfy."""
    scores: dict[str, dict] = {}
    for group, weight in ((required, 2), (support, 1)):
        for probe in group:
            if not probe["present"]:
                continue
            for hit in probe["hits"]:
                cid = hit["chunk_id"]
                rec = scores.setdefault(
                    cid,
                    {
                        "chunk_id": cid,
                        "domain": hit["domain"],
                        "probes": [],
                        "score": 0,
                        "snippet": hit["snippet"],
                    },
                )
                if probe["id"] not in rec["probes"]:
                    rec["probes"].append(probe["id"])
                    rec["score"] += weight
    ranked = sorted(scores.values(), key=lambda x: (-x["score"], x["chunk_id"]))
    return ranked[:k]


def load_previous_top5(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    details = ((blob.get("pools") or {}).get("domain") or {}).get("details") or []
    return {
        d["id"]: [c.get("source_id") for c in (d.get("context") or [])]
        for d in details
        if d.get("id")
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    from backend.env import get_pg_dsn, load_env
    from db.embed_bge_m3 import connect

    load_env()
    dsn = get_pg_dsn()
    if not dsn:
        print("DIET_EXPERT_PG_DSN missing", file=sys.stderr)
        sys.exit(1)

    items = e1_items(load_jsonl(root / "evals" / "dataset.jsonl"))
    prev = load_previous_top5(root / "evals" / "results" / "vector_m1_2026-08-30.json")

    conn = connect(dsn)
    cur = conn.cursor()
    report = {
        "date": str(date.today()),
        "method": "whitespace-folded ILIKE over knowledge_chunks; not expert gold",
        "items": [],
    }
    counts = {"in_corpus": 0, "partial": 0, "absent": 0}

    print(f"{'id':<7} {'verdict':<11} required_missing                  gold_candidates")
    for item in items:
        item_id = item["id"]
        domain = item.get("domain") or "tcm"
        spec = PROBES.get(item_id) or {"required": [], "support": []}
        required = _hits_for_group(cur, spec.get("required") or [], domain)
        support = _hits_for_group(cur, spec.get("support") or [], domain)
        status = verdict(required, support)
        counts[status] += 1
        gold = candidate_chunks(required, support)
        top5 = prev.get(item_id) or []
        gold_ids = [g["chunk_id"] for g in gold]
        overlap = [cid for cid in gold_ids if cid in top5]
        missing_req = [p["id"] for p in required if not p["present"]]
        rec = {
            "id": item_id,
            "query": item.get("query"),
            "domain": domain,
            "verdict": status,
            "required_missing": missing_req,
            "required": required,
            "support": support,
            "gold_candidates": gold,
            "vector_top5": top5,
            "gold_in_top5": overlap,
        }
        report["items"].append(rec)
        gold_s = ",".join(gold_ids[:3]) or "-"
        miss_s = ",".join(missing_req) or "-"
        print(f"{item_id:<7} {status:<11} {miss_s:<32} {gold_s}")

    cur.close()
    conn.close()
    report["summary"] = {"n": len(report["items"]), **counts}
    out = root / "evals" / "results" / f"e1_gold_chunks_{date.today()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary {counts}")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
