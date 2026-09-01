"""Recipe Skill is loaded on the recipe path, not left as a dead catalog entry."""
from backend.agents.nutrition_subagent import (
    build_nutrition_system_prompt,
    is_recipe_assembly_request,
)
from backend.skills.registry import load_recipe_skill


def test_recipe_hints_detect_shopping_and_how_to():
    assert is_recipe_assembly_request("帮我列一份购物清单") is True
    assert is_recipe_assembly_request("这道菜怎么做") is True
    assert is_recipe_assembly_request("今天该吃什么") is False


def test_nutrition_prompt_loads_recipe_skill_when_requested():
    body = load_recipe_skill()
    prompt = build_nutrition_system_prompt(include_recipe_skill=True)
    assert "购物清单" in prompt
    assert body.splitlines()[0] in prompt


def test_nutrition_prompt_omits_recipe_skill_by_default():
    prompt = build_nutrition_system_prompt()
    assert "购物清单" not in prompt
    assert "query_recipes_by_ingredients" in prompt
