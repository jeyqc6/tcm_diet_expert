#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append a line to evals/conflict_gaps.jsonl when dual-side reconcile runs
with an empty matched_rules set (PRD §11 / ENGINEERING §6.3).

Never raise into the request path: a disk/permission failure is logged
and dropped. This file is review fodder for the next conflict_rules
revision, not a query table.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("diet_expert.agents.conflict_gaps")

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "evals" / "conflict_gaps.jsonl"


def record_conflict_gap(
    *,
    trace_id: str,
    constitutions: list[str] | None = None,
    goal_tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> bool:
    """Return True when a line was written. False on any failure."""
    target = path or _DEFAULT_PATH
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "constitutions": list(constitutions or []),
        "goal_tags": list(goal_tags or []),
    }
    if extra:
        payload.update(extra)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception:
        logger.warning("conflict_gaps write failed · path=%s", target, exc_info=True)
        return False
