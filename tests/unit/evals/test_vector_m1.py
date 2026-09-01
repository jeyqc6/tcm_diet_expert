"""Deterministic M1 vector scoring. No BGE-M3, no Postgres."""
from pathlib import Path

from evals.phase8_scoring import load_jsonl
from evals.vector_m1 import (
    e1_items,
    fold_whitespace,
    keywords_hit,
    score_recall_at_k,
    score_retrieval,
    union_keyword_coverage,
)

ROOT = Path(__file__).resolve().parents[3]


def test_keywords_hit_requires_every_substring():
    assert keywords_hit("春季疏肝宜甘", ["春", "疏肝", "甘"]) is True
    assert keywords_hit("春季疏肝", ["春", "疏肝", "甘"]) is False


def test_e1_items_are_the_fifteen_with_retrieval_keywords():
    items = e1_items(load_jsonl(ROOT / "evals" / "dataset.jsonl"))
    assert len(items) == 15
    assert {r["domain"] for r in items} == {"tcm", "nutrition"}


def test_score_retrieval_uses_injected_retriever():
    items = [
        {
            "id": "E1-01",
            "query": "q",
            "domain": "tcm",
            "expect": {"retrieval_keywords": ["春", "甘"]},
        },
        {
            "id": "E1-02",
            "query": "q2",
            "domain": "tcm",
            "expect": {"retrieval_keywords": ["气虚", "山药"]},
        },
    ]

    def retrieve(row, k):
        if row["id"] == "E1-01":
            return [
                {
                    "text": "春天少酸多甘，疏肝即可",
                    "source_id": "a",
                    "score": 0.9,
                    "domain": "tcm",
                }
            ]
        return [{"text": "无关", "source_id": "b", "score": 0.1, "domain": "tcm"}]

    assert score_recall_at_k is score_retrieval
    out = score_recall_at_k(items, retrieve, k=5)
    assert out["n"] == 2
    assert out["recall_at_k"] is None
    assert out["gold_n"] == 0
    assert out["strict_hits"] == 1
    assert out["union_hits"] == 1
    assert out["strict_recall"] == 0.5
    assert out["union_recall"] == 0.5
    assert out["misses"][0]["id"] == "E1-02"
    assert out["band"] == "未达标"


def test_union_covers_keywords_split_across_chunks_and_line_breaks():
    assert fold_whitespace("山\n药") == "山药"
    chunks = [
        {"text": "气虚体质食养原则如下。"},
        {"text": "常用粳米、山\n药、大枣。"},
    ]
    cov = union_keyword_coverage(chunks, ["气虚", "山药"])
    assert cov["union_pass"] is True
    assert keywords_hit(chunks[0]["text"], ["气虚", "山药"]) is False


def test_score_retrieval_union_beats_strict_when_keywords_split():
    items = [
        {
            "id": "E1-02",
            "query": "q",
            "domain": "tcm",
            "expect": {"retrieval_keywords": ["气虚", "山药"]},
        }
    ]

    def retrieve(row, k):
        return [
            {"text": "气虚体质食养原则如下。", "source_id": "a", "score": 0.8},
            {"text": "常用粳米、山\n药、大枣。", "source_id": "b", "score": 0.7},
        ]

    out = score_retrieval(items, retrieve, k=5)
    assert out["strict_hits"] == 0
    assert out["union_hits"] == 1
    assert out["union_recall"] == 1.0
    assert out["misses"] == []


def test_parse_context_scores_clamps_and_strips_fences():
    from evals.run_vector_m1 import _parse_context_scores

    parsed = _parse_context_scores(
        '```json\n{"context_recall":9,"context_precision":"1","rationale":"ok"}\n```'
    )
    assert parsed == {
        "context_recall": 2,
        "context_precision": 1,
        "rationale": "ok",
    }
    assert _parse_context_scores("not json") is None


def test_score_gold_groups_any_of_and_two_roles():
    from evals.vector_m1 import score_gold_groups

    groups = [
        {"role": "constitution", "any_of": ["tcm_a", "tcm_b"]},
        {"role": "season", "any_of": ["tcm_000029"]},
    ]
    half = score_gold_groups(["tcm_a"], groups)
    assert half["recall"] == 0.5
    assert half["all_hit"] is False
    assert half["missing_roles"] == ["season"]
    full = score_gold_groups(["tcm_b", "tcm_000029", "noise"], groups)
    assert full["recall"] == 1.0
    assert full["all_hit"] is True


def test_e1_14_gold_is_tea_iron_not_time_window():
    from evals.vector_m1 import load_retrieval_gold

    gold = load_retrieval_gold(ROOT / "evals" / "reference_tables" / "e1_retrieval_gold.jsonl")
    e14 = gold["E1-14"]
    ids = {cid for g in e14["groups"] for cid in g["any_of"]}
    assert ids == {"nutrition_002839"}
    assert "不要要求" in e14["retrieval_gt"]
    assert "餐前后1小时" in e14["retrieval_gt"]


def test_constitution_season_items_require_two_groups():
    from evals.vector_m1 import load_retrieval_gold

    gold = load_retrieval_gold(ROOT / "evals" / "reference_tables" / "e1_retrieval_gold.jsonl")
    assert {g["role"] for g in gold["E1-01"]["groups"]} == {"constitution", "season"}
    assert gold["E1-01"]["groups"][1]["any_of"] == ["tcm_000029"]
    assert gold["E1-06"]["groups"][1]["any_of"] == ["tcm_000031"]
    assert "constitution" in {g["role"] for g in gold["E1-04"]["groups"]}
    assert "season" not in {g["role"] for g in gold["E1-04"]["groups"]}


def test_score_retrieval_official_band_uses_gold_recall():
    items = [
        {
            "id": "E1-01",
            "query": "q",
            "domain": "tcm",
            "expect": {"retrieval_keywords": ["春", "甘"]},
        }
    ]
    gold_by_id = {
        "E1-01": {
            "groups": [
                {"role": "constitution", "any_of": ["ping_he"]},
                {"role": "season", "any_of": ["tcm_000029"]},
            ]
        }
    }

    def retrieve(row, k):
        return [
            {"text": "春天少酸多甘", "source_id": "ping_he", "score": 0.9, "domain": "tcm"}
        ]

    out = score_retrieval(items, retrieve, k=5, gold_by_id=gold_by_id)
    assert out["recall_at_k"] == 0.5
    assert out["gold_hit_rate"] == 0.0
    assert out["misses"][0]["id"] == "E1-01"

