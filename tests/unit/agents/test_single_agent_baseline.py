"""
测试目标：B2 ablation baseline(docs/DECISIONS.md D1)的接线本身是对的——
role 的工具白名单是 TCM+Nutrition 两个 SubAgent 工具的并集去掉 write_memory，
system prompt 确实要求两个检索工具都用、确实带引用/追问/体质降级措辞。
不测真实检索/真实 LLM 输出质量——那是 evals/run_b2_ablation.py 的职责。
对应实现：backend/agents/single_agent_baseline.py
覆盖要求：常规
"""
from __future__ import annotations

from backend.agents.single_agent_baseline import build_single_agent_system_prompt
from backend.mcp_server.roles import (
    ROLE_TOOL_WHITELIST,
    TOOL_QUERY_DIET_LOG,
    TOOL_QUERY_RECIPES,
    TOOL_QUERY_WEATHER,
    TOOL_RETRIEVE_NUTRITION,
    TOOL_RETRIEVE_TCM,
    TOOL_WRITE_MEMORY,
    CallerRole,
)


def test_single_agent_b2_whitelist_has_both_retrieval_tools():
    whitelist = ROLE_TOOL_WHITELIST[CallerRole.SINGLE_AGENT_B2]
    assert TOOL_RETRIEVE_TCM in whitelist
    assert TOOL_RETRIEVE_NUTRITION in whitelist


def test_single_agent_b2_whitelist_is_union_of_both_subagents_minus_write():
    tcm = ROLE_TOOL_WHITELIST[CallerRole.TCM_SUBAGENT]
    nutrition = ROLE_TOOL_WHITELIST[CallerRole.NUTRITION_SUBAGENT]
    whitelist = ROLE_TOOL_WHITELIST[CallerRole.SINGLE_AGENT_B2]
    assert whitelist == tcm | nutrition
    assert TOOL_WRITE_MEMORY not in whitelist


def test_single_agent_b2_whitelist_includes_weather_and_recipes():
    whitelist = ROLE_TOOL_WHITELIST[CallerRole.SINGLE_AGENT_B2]
    assert TOOL_QUERY_WEATHER in whitelist
    assert TOOL_QUERY_RECIPES in whitelist
    assert TOOL_QUERY_DIET_LOG in whitelist


def test_prompt_requires_using_both_retrieval_tools():
    prompt = build_single_agent_system_prompt(constitution="qi_xu", allergens=())
    assert "retrieve_tcm" in prompt
    assert "retrieve_nutrition" in prompt


def test_prompt_includes_citation_requirement():
    prompt = build_single_agent_system_prompt(constitution="qi_xu", allergens=())
    assert "[source:" in prompt


def test_prompt_degrades_when_constitution_unknown():
    prompt = build_single_agent_system_prompt(constitution=None, allergens=())
    assert "体质未知" in prompt


def test_prompt_includes_allergen_instruction_when_present():
    prompt = build_single_agent_system_prompt(constitution="qi_xu", allergens=("花生",))
    assert "花生" in prompt
