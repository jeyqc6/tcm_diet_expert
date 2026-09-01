"""
测试目标：backend/agents/user_context.py
——`_row_to_context`/`UserProfileContext` 的纯函数部分用构造好的行数据测；
`fetch_user_profile` 只测失败即静默降级(不连真实数据库,同 query_diet_log.py
`_fetch_user_timezone` 的既有测试风格)。真实数据库端到端验证见对话记录，
不在 CI 里重复打真实网络/DB。
对应实现：backend/agents/user_context.py
"""
from __future__ import annotations

from backend.agents.user_context import (
    DEFAULT_USER_ID,
    UserProfileContext,
    _row_to_context,
    create_user,
    ensure_user_profile,
    fetch_user_profile,
    list_users,
)


def test_row_to_context_maps_all_fields():
    row = {
        "constitution": "qi_xu",
        "constitution_secondary": ["yang_xu"],
        "constitution_source": "ccmq_computed",
        "allergens": ["花生", "虾"],
        "supplements": [{"name": "鱼油", "dose": "1000mg"}],
        "goal_tags": ["weight_management"],
        "preferences": {"忌口": ["香菜"]},
        "city": "上海",
        "timezone": "Asia/Shanghai",
    }
    ctx = _row_to_context("default_user", row)
    assert ctx.constitution == "qi_xu"
    assert ctx.constitution_secondary == ("yang_xu",)
    assert ctx.allergens == ("花生", "虾")
    assert ctx.supplements == ({"name": "鱼油", "dose": "1000mg"},)
    assert ctx.goal_tags == ("weight_management",)
    assert ctx.preferences == {"忌口": ["香菜"]}
    assert ctx.city == "上海"
    assert ctx.timezone == "Asia/Shanghai"
    assert ctx.onboarding_done is False
    assert ctx.locale == "zh"


def test_row_to_context_handles_missing_optional_fields():
    ctx = _row_to_context("default_user", {"constitution": None})
    assert ctx.constitution is None
    assert ctx.constitution_secondary == ()
    assert ctx.allergens == ()
    assert ctx.supplements == ()
    assert ctx.preferences == {}
    assert ctx.onboarding_done is False
    assert ctx.locale == "zh"


def test_row_to_context_parses_json_string_supplements_as_fallback():
    row = {"supplements": '[{"name": "鱼油", "dose": null}]'}
    ctx = _row_to_context("default_user", row)
    assert ctx.supplements == ({"name": "鱼油", "dose": None},)


def test_row_to_context_tolerates_unparseable_supplements_string():
    row = {"supplements": "not json"}
    ctx = _row_to_context("default_user", row)
    assert ctx.supplements == ()


def test_row_to_context_parses_json_string_preferences_as_fallback():
    row = {"preferences": '{"忌口": ["香菜"]}'}
    ctx = _row_to_context("default_user", row)
    assert ctx.preferences == {"忌口": ["香菜"]}


def test_row_to_context_tolerates_unparseable_preferences_string():
    row = {"preferences": "not json"}
    ctx = _row_to_context("default_user", row)
    assert ctx.preferences == {}


def test_constitutions_merges_primary_and_secondary():
    ctx = UserProfileContext(
        user_id=DEFAULT_USER_ID, constitution="qi_xu", constitution_secondary=("yang_xu", "tan_shi")
    )
    assert ctx.constitutions() == ("qi_xu", "yang_xu", "tan_shi")


def test_constitutions_falls_back_to_secondary_when_no_primary():
    ctx = UserProfileContext(user_id=DEFAULT_USER_ID, constitution_secondary=("yang_xu",))
    assert ctx.constitutions() == ("yang_xu",)


def test_constitutions_empty_when_nothing_known():
    ctx = UserProfileContext(user_id=DEFAULT_USER_ID)
    assert ctx.constitutions() == ()


def test_to_reconciliation_dict_shape():
    ctx = UserProfileContext(
        user_id=DEFAULT_USER_ID,
        constitution="qi_xu",
        constitution_secondary=("yang_xu",),
        allergens=("花生",),
        supplements=({"name": "鱼油", "dose": "1000mg"},),
        goal_tags=("weight_management",),
        preferences={"忌口": ["香菜"]},
    )
    d = ctx.to_reconciliation_dict()
    assert d == {
        "constitution": "qi_xu",
        "constitution_secondary": ["yang_xu"],
        "allergens": ["花生"],
        "supplements": ["鱼油(1000mg)"],
        "goal_tags": ["weight_management"],
        "preferences": {"忌口": ["香菜"]},
    }


def test_to_verification_summary_includes_constitution_and_allergens():
    ctx = UserProfileContext(user_id=DEFAULT_USER_ID, constitution="qi_xu", allergens=("花生", "虾"))
    assert ctx.to_verification_summary() == "体质:qi_xu；过敏原:花生,虾"


def test_to_verification_summary_includes_supplements_and_preferences():
    ctx = UserProfileContext(
        user_id=DEFAULT_USER_ID,
        supplements=({"name": "鱼油", "dose": None},),
        preferences={"忌口": ["香菜"]},
    )
    summary = ctx.to_verification_summary()
    assert "在服补剂:鱼油" in summary
    assert "香菜" in summary


def test_profile_prompt_notes_uses_e8_disclaimer_when_supplements_present():
    ctx = UserProfileContext(
        user_id=DEFAULT_USER_ID,
        supplements=({"name": "鱼油", "dose": None},),
        preferences={"忌口": ["香菜"]},
    )
    notes = ctx.profile_prompt_notes()
    assert "鱼油" in notes
    assert "不确定" in notes
    assert "香菜" in notes


def test_profile_prompt_notes_includes_city_for_query_weather():
    """回归测试：`city` 之前没写进 profile_prompt_notes()，TCM SubAgent 调
    query_weather 时不知道真实城市，会编一个占位字符串当参数传，Open-Meteo
    地理编码查不到，工具静默降级成节气兜底——看起来像"天气接口没连上"，
    其实是画像里的 city 从没喂给过模型。"""
    ctx = UserProfileContext(user_id=DEFAULT_USER_ID, city="广州")
    notes = ctx.profile_prompt_notes()
    assert "广州" in notes
    assert "query_weather" in notes


def test_profile_prompt_notes_no_city_line_when_city_unset():
    ctx = UserProfileContext(user_id=DEFAULT_USER_ID)
    assert "query_weather" not in ctx.profile_prompt_notes()


def test_to_verification_summary_empty_when_nothing_known():
    ctx = UserProfileContext(user_id=DEFAULT_USER_ID)
    assert ctx.to_verification_summary() == ""


def test_fetch_user_profile_returns_none_without_dsn(monkeypatch):
    # Patch the module's own `get_pg_dsn` reference rather than the env var directly:
    # `backend.env.load_env()` caches "already loaded" at module-import time (and the
    # repo's real .env does have a DSN set), so deleting the env var here would not
    # reliably force a "no DSN configured" scenario — this way it's deterministic
    # regardless of what other tests/imports already triggered.
    monkeypatch.setattr("backend.agents.user_context.get_pg_dsn", lambda explicit=None: None)
    assert fetch_user_profile() is None


def test_fetch_user_profile_returns_none_on_connection_failure():
    assert fetch_user_profile(dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist") is None


def test_list_users_returns_empty_without_dsn(monkeypatch):
    monkeypatch.setattr("backend.agents.user_context.get_pg_dsn", lambda explicit=None: None)
    assert list_users() == []


def test_create_user_returns_none_without_dsn(monkeypatch):
    monkeypatch.setattr("backend.agents.user_context.get_pg_dsn", lambda explicit=None: None)
    assert create_user("新用户") is None


def test_ensure_user_profile_returns_false_without_dsn(monkeypatch):
    monkeypatch.setattr("backend.agents.user_context.get_pg_dsn", lambda explicit=None: None)
    assert ensure_user_profile() is False


def test_ensure_user_profile_returns_false_on_connection_failure():
    assert ensure_user_profile(dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist") is False
