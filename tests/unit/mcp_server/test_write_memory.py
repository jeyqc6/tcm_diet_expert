"""
测试目标：backend/mcp_server/tools/write_memory.py
——category="critical"(写 user_profile)、category="daily_log"(写 diet_log)的
校验/短路逻辑不连数据库能测；真正的 INSERT ... ON CONFLICT 落库路径(含幂等键
去重)见对话记录里对真实本地 Postgres 的验证(读回来核对字段，不只是"没报错")。
对应实现：backend/mcp_server/tools/write_memory.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.mcp_server.tools.write_memory import write_memory


def test_unknown_category_raises_value_error():
    with pytest.raises(ValueError, match="unknown write_memory category"):
        write_memory("not_a_real_category", {})


def test_critical_rejects_unknown_field_without_touching_db():
    with pytest.raises(ValueError, match="unknown user_profile field"):
        write_memory("critical", {"not_a_real_column": 1})


def test_critical_empty_payload_is_a_noop_without_touching_db():
    # 空 payload 在函数体内会在解析 dsn/连库之前就直接返回——不传 dsn 也不会
    # 因为"连不上库"而报错，这个测试本身就是在验证这条早退路径。
    result = write_memory("critical", {})
    assert result.ok is False
    assert result.fields_written == ()


def test_critical_all_none_values_is_a_noop_without_touching_db():
    """None 表示"这一步引导没收集到值"，不是"要把这一列清空"——不该发起写请求。"""
    result = write_memory("critical", {"constitution": None, "city": None})
    assert result.ok is False


def test_critical_accepts_supplements_field():
    """`supplements` 曾经不在 `_CRITICAL_COLUMNS` 里，`write_memory("critical",
    {"supplements": [...]})` 会直接被当成未知字段拒绝(`ValueError`)——
    backend/memory/critical_fact_scanner.py 需要写这一列，这里验证字段校验
    这一步不再挡住它：失败的原因必须是连不上这个假 DSN(`Exception` 泛化，
    同 `test_critical_write_failure_raises_not_silently_swallowed` 的既有
    写法)，不能是"unknown user_profile field"。"""
    with pytest.raises(Exception) as exc_info:
        write_memory(
            "critical",
            {"supplements": [{"name": "鱼油", "dose": None}]},
            dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist",
        )
    assert "unknown user_profile field" not in str(exc_info.value)


def test_critical_write_failure_raises_not_silently_swallowed():
    """和 fetch_user_profile()/fetch_matched_conflict_rules() 的"读失败静默降级
    为 None/[]"刻意不对称：写失败必须让调用方知道，不能让用户以为保存成功了
    实际上没有落库。"""
    with pytest.raises(Exception):
        write_memory(
            "critical",
            {"constitution": "qi_xu"},
            dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist",
        )


_VALID_DAILY_LOG_PAYLOAD = {
    "raw_input": "晚上吃了番茄炒蛋",
    "dishes": [{"dish": "番茄炒蛋", "confidence": "high"}],
    "ingredients": ["鸡蛋", "番茄"],
    "food_properties": ["平"],
    "meal_type": "晚餐",
    "logged_at": datetime(2026, 8, 26, 19, 0, tzinfo=timezone.utc),
}


def test_daily_log_requires_idempotency_key():
    with pytest.raises(ValueError, match="idempotency_key is required"):
        write_memory("daily_log", _VALID_DAILY_LOG_PAYLOAD, idempotency_key=None)


def test_daily_log_rejects_missing_field_without_touching_db():
    payload = dict(_VALID_DAILY_LOG_PAYLOAD)
    del payload["meal_type"]
    with pytest.raises(ValueError, match="missing field"):
        write_memory("daily_log", payload, idempotency_key="k1")


def test_daily_log_rejects_unknown_field_without_touching_db():
    payload = {**_VALID_DAILY_LOG_PAYLOAD, "extra_field": 1}
    with pytest.raises(ValueError, match="unknown field"):
        write_memory("daily_log", payload, idempotency_key="k1")


def test_daily_log_rejects_invalid_meal_type_without_touching_db():
    payload = {**_VALID_DAILY_LOG_PAYLOAD, "meal_type": "不是一个合法值"}
    with pytest.raises(ValueError, match="invalid meal_type"):
        write_memory("daily_log", payload, idempotency_key="k1")


def test_daily_log_rejects_non_datetime_logged_at_without_touching_db():
    payload = {**_VALID_DAILY_LOG_PAYLOAD, "logged_at": "2026-08-26"}
    with pytest.raises(ValueError, match="logged_at must be a datetime"):
        write_memory("daily_log", payload, idempotency_key="k1")


def test_daily_log_write_failure_raises_not_silently_swallowed():
    with pytest.raises(Exception):
        write_memory(
            "daily_log",
            _VALID_DAILY_LOG_PAYLOAD,
            idempotency_key="k1",
            dsn="postgresql://nouser:nopass@127.0.0.1:1/doesnotexist",
        )
