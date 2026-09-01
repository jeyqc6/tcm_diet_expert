#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Skills · Layer 1 目录 + Layer 2 按需加载（ARCHITECTURE.md §6.2）。

关键约束（D22 / D20）:
  - 加载是**确定性**的：代码在到达 pipeline 步骤时必然 load，不是模型读
    description 后自己决定要不要加载。
  - Skill **不**常驻中枢 system prompt；只拼进「那一次」调用的 prompt
    （调和层 / 核查 pass / 菜谱组装）。
  - Skill 不获得独立 LLM 调用，只是装进已有调用的上下文（D14/D15）。

Layer 1：本模块的 SKILL_CATALOG（给人看的索引，不进任何 LLM prompt）。
Layer 2：load_skill_body() 读 .md 正文并进程内缓存。
Layer 3：conflict_rules 命中行 —— 不在本模块，由调和层查表后自行拼入。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent

# Pipeline step ids used in load_at_step (ARCHITECTURE §5.2 / §6.1)
STEP_RECONCILIATION = "reconciliation"  # §5.2 步骤 5
STEP_VERIFICATION = "verification"  # §5.2 步骤 6
STEP_RECIPE_ASSEMBLY = "recipe_assembly"  # §5.2 步骤 7
STEP_ONBOARDING_QUESTIONNAIRE = "onboarding_questionnaire"  # §11.2
STEP_ED_RISK_RESPONSE = "ed_risk_response"  # §5.4 / PRD §16

_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class SkillMeta:
    """Layer 1 catalog entry — human/docs index only; never stuffed into LLM prompts."""

    id: str
    path: str  # filename relative to backend/skills/
    description: str
    load_at_step: str
    version: str  # must match YAML front matter in the .md file


# Layer 1 catalog (ARCHITECTURE §6.2). Task #10 要求三份基础 Skill 内容齐备；
# ccmq / ed_risk 已登记元数据，正文可仍是短占位，等引导/安全层任务再补。
SKILL_CATALOG: tuple[SkillMeta, ...] = (
    SkillMeta(
        id="reconciliation_rubric",
        path="reconciliation_rubric.md",
        description="调和层仲裁准则（含 D25 harm-reduction、2026-08-31 反延伸规则+去重）",
        load_at_step=STEP_RECONCILIATION,
        version="1.2.0",
    ),
    SkillMeta(
        id="verification_checklist",
        path="verification_checklist.md",
        description="核查 pass 检查清单（PRD §10.1 七条 + D25 候选评估规则 + 证据修复边界）",
        load_at_step=STEP_VERIFICATION,
        version="1.2.0",
    ),
    SkillMeta(
        id="recipe_and_shopping_list",
        path="recipe_and_shopping_list.md",
        description="菜谱与购物清单输出模板",
        load_at_step=STEP_RECIPE_ASSEMBLY,
        version="1.0.0",
    ),
    SkillMeta(
        id="ccmq_questionnaire",
        path="ccmq_questionnaire.md",
        description="CCMQ 简版问卷题库与计分口径（D22 补充）",
        load_at_step=STEP_ONBOARDING_QUESTIONNAIRE,
        version="0.1.0",
    ),
    SkillMeta(
        id="ed_risk_response",
        path="ed_risk_response.md",
        description="ED 风险响应话术模板（D22 补充；改动需人工审校）",
        load_at_step=STEP_ED_RISK_RESPONSE,
        version="0.0.1",
    ),
)

_CATALOG_BY_ID: dict[str, SkillMeta] = {s.id: s for s in SKILL_CATALOG}


def get_skill_meta(skill_id: str) -> SkillMeta:
    try:
        return _CATALOG_BY_ID[skill_id]
    except KeyError as exc:
        raise KeyError(f"unknown skill id {skill_id!r}; known: {sorted(_CATALOG_BY_ID)}") from exc


def skills_for_step(step: str) -> list[SkillMeta]:
    """Return catalog entries declared for a pipeline step (usually 0 or 1)."""
    return [s for s in SKILL_CATALOG if s.load_at_step == step]


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = _FRONT_MATTER_RE.match(raw)
    if not m:
        raise ValueError("skill file must start with YAML front matter (--- ... ---)")
    meta: dict[str, str] = {}
    for line in m.group("meta").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, m.group("body").strip()


@lru_cache(maxsize=32)
def load_skill_body(skill_id: str) -> str:
    """Layer 2: read skill markdown body (no front matter). Cached in-process."""
    meta = get_skill_meta(skill_id)
    path = _SKILLS_DIR / meta.path
    raw = path.read_text(encoding="utf-8")
    file_meta, body = _parse_front_matter(raw)
    file_version = file_meta.get("version")
    if file_version != meta.version:
        raise ValueError(
            f"skill {skill_id!r}: catalog version {meta.version!r} "
            f"!= file front matter version {file_version!r}"
        )
    file_id = file_meta.get("id")
    if file_id and file_id != skill_id:
        raise ValueError(f"skill path id mismatch: catalog={skill_id!r} file={file_id!r}")
    return body


def clear_skill_cache() -> None:
    """Test helper / explicit invalidation (V1 has no hot-reload)."""
    load_skill_body.cache_clear()


def compose_prompt_with_skills(base_system: str, skill_ids: list[str]) -> str:
    """Append Layer 2 skill bodies to a call-specific system prompt.

    This is how Skills enter an LLM call: only when the caller (reconciliation /
    verification / recipe assembly) explicitly asks. The hub/router system prompt
    must NOT call this with these skill_ids (D22 budget constraint).

    Each skill body already opens with its own descriptive H1 (e.g. "# 调和层仲裁
    准则"), which is far more useful to the model than an internal `id`/`version`
    slug — so we only insert a `---` separator, not a synthetic header on top of
    that title. `get_skill_meta()` is still called (via `load_skill_body`) for the
    version-consistency check; it's just no longer echoed into the prompt itself.
    """
    parts = [base_system.rstrip()]
    for skill_id in skill_ids:
        body = load_skill_body(skill_id)
        parts.append(f"\n\n---\n\n{body}")
    return "".join(parts).strip() + "\n"


# Convenience loaders for pipeline steps (deterministic — always load when step runs)
def load_reconciliation_skill() -> str:
    return load_skill_body("reconciliation_rubric")


def load_verification_skill() -> str:
    return load_skill_body("verification_checklist")


def load_recipe_skill() -> str:
    return load_skill_body("recipe_and_shopping_list")
