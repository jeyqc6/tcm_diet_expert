"""
测试目标：backend/agents/log_write.py 里 log_write 分支用到的纯函数辅助逻辑——
meal_type 推断(关键词优先/时段兜底)、logged_at 相对日期解析、幂等键计算
(分钟级截断)、确认文本格式化。集成级别的完整 SSE 流程验证见 test_api_chat_sse.py。
对应实现：backend/agents/log_write.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from backend.agents.log_write import (
    _compute_idempotency_key,
    _format_log_write_confirmation,
    _infer_meal_type,
    _resolve_log_tz,
    _resolve_logged_at,
)
from backend.agents.user_context import UserProfileContext
from backend.memory.dish_decomposition import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    SOURCE_GLOBAL_TABLE,
    SOURCE_LLM_FALLBACK,
    DishMatch,
    MealDecomposition,
)

TZ = ZoneInfo("Asia/Shanghai")


def test_infer_meal_type_matches_keyword_even_when_not_adjacent_to_verb():
    """真实发现的问题：最初版本要求"早上吃"连在一起才命中，"早上刚喝了..."
    这种中间插了别的字的说法会漏检，错误退到按当前时刻推断。"""
    now = datetime(2026, 8, 27, 15, 0, tzinfo=TZ)  # 下午
    assert _infer_meal_type("早上刚喝了一杯燕麦牛奶", now) == "早餐"


def test_infer_meal_type_keyword_priority_over_time_of_day():
    now = datetime(2026, 8, 27, 8, 0, tzinfo=TZ)  # 早上
    assert _infer_meal_type("昨天夜宵吃了泡面", now) == "夜宵"


def test_infer_meal_type_falls_back_to_time_of_day_without_keyword():
    now = datetime(2026, 8, 27, 8, 0, tzinfo=TZ)
    assert _infer_meal_type("喝了一杯咖啡", now) == "早餐"
    assert _infer_meal_type("喝了一杯咖啡", now.replace(hour=12)) == "午餐"
    assert _infer_meal_type("喝了一杯咖啡", now.replace(hour=15)) == "下午茶"
    assert _infer_meal_type("喝了一杯咖啡", now.replace(hour=19)) == "晚餐"
    assert _infer_meal_type("喝了一杯咖啡", now.replace(hour=23)) == "夜宵"
    assert _infer_meal_type("喝了一杯咖啡", now.replace(hour=3)) == "未知"


def test_resolve_logged_at_defaults_to_now():
    now_before = datetime.now(TZ)
    result = _resolve_logged_at("刚吃了番茄炒蛋", TZ)
    assert abs((result - now_before).total_seconds()) < 5


def test_resolve_logged_at_recognizes_yesterday():
    result = _resolve_logged_at("昨天吃了番茄炒蛋", TZ)
    now = datetime.now(TZ)
    assert (now.date() - result.date()).days == 1


def test_resolve_logged_at_recognizes_the_day_before_yesterday():
    result = _resolve_logged_at("前天吃了番茄炒蛋", TZ)
    now = datetime.now(TZ)
    assert (now.date() - result.date()).days == 2


def test_resolve_log_tz_prefers_profile_timezone():
    profile = UserProfileContext(user_id="default_user", timezone="America/New_York")
    assert _resolve_log_tz(profile) == ZoneInfo("America/New_York")


def test_resolve_log_tz_falls_back_to_default_without_profile():
    assert _resolve_log_tz(None) == ZoneInfo("Asia/Shanghai")


def test_resolve_log_tz_ignores_invalid_timezone_name():
    profile = UserProfileContext(user_id="default_user", timezone="Not/A_Real_Zone")
    assert _resolve_log_tz(profile) == ZoneInfo("Asia/Shanghai")


def test_compute_idempotency_key_stable_for_same_minute():
    t1 = datetime(2026, 8, 27, 12, 0, 5, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 27, 12, 0, 55, tzinfo=timezone.utc)
    key1 = _compute_idempotency_key("default_user", t1, "麻婆豆腐")
    key2 = _compute_idempotency_key("default_user", t2, "麻婆豆腐")
    assert key1 == key2


def test_compute_idempotency_key_differs_across_minutes():
    t1 = datetime(2026, 8, 27, 12, 0, 5, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 27, 12, 1, 5, tzinfo=timezone.utc)
    key1 = _compute_idempotency_key("default_user", t1, "麻婆豆腐")
    key2 = _compute_idempotency_key("default_user", t2, "麻婆豆腐")
    assert key1 != key2


def test_compute_idempotency_key_differs_for_different_raw_input():
    t = datetime(2026, 8, 27, 12, 0, 5, tzinfo=timezone.utc)
    key1 = _compute_idempotency_key("default_user", t, "麻婆豆腐")
    key2 = _compute_idempotency_key("default_user", t, "宫保鸡丁")
    assert key1 != key2


def test_compute_idempotency_key_differs_for_different_user():
    t = datetime(2026, 8, 27, 12, 0, 5, tzinfo=timezone.utc)
    key1 = _compute_idempotency_key("user_a", t, "麻婆豆腐")
    key2 = _compute_idempotency_key("user_b", t, "麻婆豆腐")
    assert key1 != key2


def test_format_log_write_confirmation_shows_dishes_and_ingredients():
    match = DishMatch(
        dish="番茄炒蛋", ingredients=("鸡蛋", "番茄"), tcm_nature="平", allergens=("蛋",),
        confidence=CONFIDENCE_HIGH, source_tier=SOURCE_GLOBAL_TABLE,
    )
    decomp = MealDecomposition(matches=(match,))
    text = _format_log_write_confirmation(decomp, "午餐", datetime(2026, 8, 27, 12, 0, tzinfo=TZ), duplicate=False)
    assert "已记录" in text
    assert "番茄炒蛋" in text
    assert "鸡蛋" in text


def test_format_log_write_confirmation_flags_llm_fallback_matches():
    match = DishMatch(
        dish="燕麦牛奶", ingredients=("燕麦", "牛奶"), tcm_nature="平", allergens=(),
        confidence=CONFIDENCE_LOW, source_tier=SOURCE_LLM_FALLBACK,
    )
    decomp = MealDecomposition(matches=(match,))
    text = _format_log_write_confirmation(decomp, "下午茶", datetime(2026, 8, 27, 15, 0, tzinfo=TZ), duplicate=False)
    assert "模型推测" in text


def test_format_log_write_confirmation_marks_duplicate():
    match = DishMatch(
        dish="番茄炒蛋", ingredients=("鸡蛋",), tcm_nature="平", allergens=(),
        confidence=CONFIDENCE_HIGH, source_tier=SOURCE_GLOBAL_TABLE,
    )
    decomp = MealDecomposition(matches=(match,))
    text = _format_log_write_confirmation(decomp, "午餐", datetime(2026, 8, 27, 12, 0, tzinfo=TZ), duplicate=True)
    assert "之前已经记过" in text
