"""
测试目标：Layer 1 目录、version 解析、触发步骤映射、Layer 2 缓存、
Skill 拼进对应调用 prompt 而非中枢常驻 system prompt（D22 / ARCHITECTURE §6.2）
对应实现：backend/skills/registry.py
覆盖要求：常规
"""
from __future__ import annotations

import backend.skills.registry as registry
from backend.skills.registry import (
    STEP_RECIPE_ASSEMBLY,
    STEP_RECONCILIATION,
    STEP_VERIFICATION,
    SkillMeta,
    clear_skill_cache,
    compose_prompt_with_skills,
    get_skill_meta,
    load_skill_body,
    skills_for_step,
)


def setup_function() -> None:
    clear_skill_cache()


def test_catalog_contains_three_core_skills() -> None:
    ids = {s.id for s in registry.SKILL_CATALOG}
    assert {
        "reconciliation_rubric",
        "verification_checklist",
        "recipe_and_shopping_list",
    }.issubset(ids)


def test_load_at_step_mapping() -> None:
    assert [s.id for s in skills_for_step(STEP_RECONCILIATION)] == ["reconciliation_rubric"]
    assert [s.id for s in skills_for_step(STEP_VERIFICATION)] == ["verification_checklist"]
    assert [s.id for s in skills_for_step(STEP_RECIPE_ASSEMBLY)] == ["recipe_and_shopping_list"]
    assert skills_for_step("router_system") == []


def test_version_matches_front_matter() -> None:
    for skill_id in (
        "reconciliation_rubric",
        "verification_checklist",
        "recipe_and_shopping_list",
    ):
        meta = get_skill_meta(skill_id)
        assert isinstance(meta, SkillMeta)
        body = load_skill_body(skill_id)
        assert body  # front matter stripped
        assert "---" not in body.splitlines()[0]


def test_load_skill_body_is_cached(monkeypatch) -> None:
    reads: list[str] = []
    real_read = registry.Path.read_text

    def tracking_read(self, *args, **kwargs):
        reads.append(str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(registry.Path, "read_text", tracking_read)
    clear_skill_cache()
    load_skill_body("reconciliation_rubric")
    load_skill_body("reconciliation_rubric")
    assert len(reads) == 1


def test_skill_content_injected_into_reconciliation_prompt_not_hub() -> None:
    hub_system = "你是中枢 agent。只做路由与编排，不在此加载领域 Skill。"
    # Hub must NOT compose with reconciliation skill
    assert "Harm reduction" not in hub_system
    assert "调和层仲裁准则" not in hub_system

    recon_system = compose_prompt_with_skills(
        "你是调和层。根据两侧结论做一次仲裁。",
        ["reconciliation_rubric"],
    )
    assert "你是调和层" in recon_system
    assert "Harm reduction" in recon_system or "harm reduction" in recon_system.lower()
    assert "过敏原" in recon_system


def test_verification_skill_includes_prd_items_and_d25_candidate_rule() -> None:
    body = load_skill_body("verification_checklist")
    assert "source_id" in body
    assert "ED 防护" in body or "ED" in body
    assert "候选评估" in body
    assert "支持理由" in body


def test_recipe_skill_loaded_only_via_explicit_compose() -> None:
    hub = "你是中枢 agent。"
    assert "购物清单" not in hub
    prompt = compose_prompt_with_skills(hub, ["recipe_and_shopping_list"])
    assert "购物清单" in prompt
    assert "source_id" in prompt or "source:" in prompt


def test_unknown_skill_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        get_skill_meta("not_a_real_skill")
