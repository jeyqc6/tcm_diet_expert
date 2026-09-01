"""Deterministic tests for Phase 8 LLM-as-judge helpers. No network."""
from pathlib import Path

from evals.phase8_judge import (
    build_ground_truth,
    load_arm_answers,
    normalize_judge_scores,
    parse_ground_truth_ref,
    semantic_pass,
    strip_json_fences,
    summarize_arm,
)
from evals.phase8_scoring import load_jsonl

ROOT = Path(__file__).resolve().parents[3]


def test_strip_fences_and_clamp():
    assert strip_json_fences("```json\n{\"direction\": 2}\n```") == '{"direction": 2}'
    scores = normalize_judge_scores({"direction": "9", "safety": "nope", "rationale": "x" * 500})
    assert scores["direction"] == 2
    assert scores["safety"] == 0
    assert len(scores["rationale"]) == 400


def test_semantic_pass_requires_full_direction_and_safety():
    assert semantic_pass({"direction": 2, "safety": 1}) is True
    assert semantic_pass({"direction": 1, "safety": 1}) is False
    assert semantic_pass({"direction": 2, "safety": 0}) is False


def test_ground_truth_e1_uses_constitution_table_not_just_keywords():
    dataset = {r["id"]: r for r in load_jsonl(ROOT / "evals" / "dataset.jsonl")}
    constitution = load_jsonl(ROOT / "evals" / "reference_tables" / "constitution_season.jsonl")
    rules = load_jsonl(ROOT / "evals" / "conflict_rules.jsonl")
    gt = build_ground_truth(
        dataset["E1-01"], constitution_rows=constitution, conflict_rules=rules
    )
    assert "少酸多甘" in gt
    assert "允许同义转述" in gt
    assert parse_ground_truth_ref(dataset["E1-01"]["ground_truth_ref"]) == (
        "evals/reference_tables/constitution_season.jsonl",
        "E1-01",
    )


def test_ground_truth_e2_includes_resolution_and_must_not():
    dataset = {r["id"]: r for r in load_jsonl(ROOT / "evals" / "dataset.jsonl")}
    constitution = load_jsonl(ROOT / "evals" / "reference_tables" / "constitution_season.jsonl")
    rules = load_jsonl(ROOT / "evals" / "conflict_rules.jsonl")
    gt = build_ground_truth(
        dataset["E2a-07"], constitution_rows=constitution, conflict_rules=rules
    )
    assert "华法林" in gt
    assert "语义上禁止的说法" in gt
    assert "完全可以" in gt


def test_load_arm_answers_flags_truncation():
    rows = [
        {"id": "E1-01", "answer_preview": "x" * 400, "answer_len": 553, "pass": False},
        {"id": "E1-02", "answer_preview": "short", "answer_len": 5, "pass": True},
    ]
    arms = load_arm_answers(rows)
    assert arms["E1-01"]["truncated"] is True
    assert arms["E1-02"]["truncated"] is False
    assert arms["E1-01"]["keyword_pass"] is False


def test_summarize_arm_splits_m3_m5_and_slices():
    rows = [
        {
            "id": "E1-01",
            "subset": "E1",
            "slices": [],
            "scored": True,
            "semantic_pass": True,
            "keyword_pass": False,
            "scores": {
                "direction": 2,
                "safety": 1,
                "allergen_safe": 1,
                "disclaimer": 1,
                "numeric_restraint": 1,
                "supplement_caution": 1,
            },
        },
        {
            "id": "E2a-01",
            "subset": "E2a",
            "slices": ["症状类"],
            "scored": True,
            "semantic_pass": False,
            "keyword_pass": True,
            "scores": {
                "direction": 1,
                "safety": 1,
                "allergen_safe": 1,
                "disclaimer": 1,
                "numeric_restraint": 1,
                "supplement_caution": 1,
            },
        },
    ]
    summary = summarize_arm(rows)
    assert summary["n"] == 2
    assert summary["judge_passed"] == 1
    assert summary["keyword_passed"] == 1
    assert summary["M3_E1"]["n"] == 1
    assert summary["M3_E1"]["judge_passed"] == 1
    assert summary["M5_E2"]["judge_passed"] == 0
    assert summary["slices"]["症状类"]["extra_passed"] == 1
