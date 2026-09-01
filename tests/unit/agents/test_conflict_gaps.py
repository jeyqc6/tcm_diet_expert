"""conflict_gaps.jsonl append must not fail the request path."""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.conflict_gaps import record_conflict_gap


def test_record_conflict_gap_appends_trace_id(tmp_path: Path):
    path = tmp_path / "conflict_gaps.jsonl"
    assert record_conflict_gap(
        trace_id="tr-1",
        constitutions=["qi_xu"],
        goal_tags=["weight_management"],
        path=path,
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_id"] == "tr-1"
    assert payload["constitutions"] == ["qi_xu"]


def test_record_conflict_gap_swallows_write_errors(tmp_path: Path, monkeypatch):
    path = tmp_path / "conflict_gaps.jsonl"

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", boom)
    assert record_conflict_gap(trace_id="tr-2", path=path) is False
