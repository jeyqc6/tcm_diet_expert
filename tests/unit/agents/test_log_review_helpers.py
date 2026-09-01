"""
测试目标：backend/agents/log_review.py 里 log_review 分支用到的纯函数辅助逻辑——
按查询用户自己的时区格式化 logged_at(2026-08-31 新增，之前直接展示数据库驱动
带回来的原始 tzinfo，跟用户所在地对不上)、回顾摘要优先展示 dishes 而不是
raw_input。集成级别的完整 SSE 流程验证见 test_api_chat_sse.py。
对应实现：backend/agents/log_review.py
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

from backend.agents.log_review import (
    _format_diet_log_summary,
    _format_entry_food_summary,
    _format_logged_at,
    _resolve_review_tz,
)
from backend.agents.user_context import UserProfileContext


def test_resolve_review_tz_prefers_profile_timezone():
    profile = UserProfileContext(user_id="default_user", timezone="America/Los_Angeles")
    assert _resolve_review_tz(profile) == ZoneInfo("America/Los_Angeles")


def test_resolve_review_tz_falls_back_to_default_without_profile():
    assert _resolve_review_tz(None) == ZoneInfo("Asia/Shanghai")


def test_resolve_review_tz_ignores_invalid_timezone_name():
    profile = UserProfileContext(user_id="default_user", timezone="Not/A_Real_Zone")
    assert _resolve_review_tz(profile) == ZoneInfo("Asia/Shanghai")


def test_format_logged_at_converts_to_target_timezone():
    """数据库驱动带回来的 tzinfo(这里模拟成会话时区 -04:00)不是用户自己的
    时区——展示前必须转换成用户自己的 timezone，而不是原样透传驱动的 offset。"""
    # 06:13 -04:00 == 18:13 +08:00(同一个绝对时刻)
    result = _format_logged_at("2026-08-31T06:13:02.348934-04:00", ZoneInfo("Asia/Shanghai"))
    assert result == "2026-08-31 18:13"


def test_format_logged_at_handles_unparseable_input_gracefully():
    assert _format_logged_at("", ZoneInfo("Asia/Shanghai")) == ""
    assert _format_logged_at("not-a-datetime", ZoneInfo("Asia/Shanghai")) == "not-a-datetime"


def test_format_entry_food_summary_prefers_dishes_over_raw_input():
    entry = {
        "raw_input": "help me record my breakfast, it's two eggs with lettuce and pork dumplings",
        "dishes": [{"dish": "Eggs with lettuce"}, {"dish": "Pork dumplings"}],
    }
    assert _format_entry_food_summary(entry) == "Eggs with lettuce、Pork dumplings"


def test_format_entry_food_summary_falls_back_to_raw_input_without_dishes():
    """历史上更早、dishes 字段还不存在时写入的行——没有 dishes 时退回
    raw_input，好过展示空字符串。"""
    entry = {"raw_input": "红烧肉", "dishes": []}
    assert _format_entry_food_summary(entry) == "红烧肉"


def test_format_diet_log_summary_uses_target_timezone_for_every_entry():
    raw = {
        "time_range": "今天",
        "entries": [
            {
                "logged_at": "2026-08-31T06:13:02-04:00",
                "meal_type": "早餐",
                "dishes": [{"dish": "Eggs with lettuce"}],
            }
        ],
    }
    text = _format_diet_log_summary(raw, ZoneInfo("Asia/Shanghai"))
    assert "2026-08-31 18:13" in text
    assert "Eggs with lettuce" in text
    assert "-04:00" not in text
