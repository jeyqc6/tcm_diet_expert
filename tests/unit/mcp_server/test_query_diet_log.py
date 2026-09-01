"""
测试目标：backend/mcp_server/tools/query_diet_log.py
——相对日期解析(parse_time_range)是纯函数，聚合逻辑(query_diet_log)用注入的
entries 测，都不需要真实数据库。真实数据库端到端验证见本文件末尾的说明
（手工在真实 diet_expert 库上跑过，见对话记录，不在 CI 里重复打真实网络/DB）。
对应实现：backend/mcp_server/tools/query_diet_log.py
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import backend.mcp_server.tools.query_diet_log as qdl
from backend.mcp_server.tools.query_diet_log import _get_tz, parse_time_range, query_diet_log

TZ = ZoneInfo("Asia/Shanghai")
# 固定"现在"为一个周三，方便断言"本周"/"上周"的边界，不依赖测试运行的真实日期
NOW = datetime(2026, 8, 26, 20, 0, tzinfo=TZ)  # 2026-08-26 是周三


def test_today():
    span = parse_time_range("今天", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 26, 0, 0, tzinfo=TZ)
    assert span.end == datetime(2026, 8, 27, 0, 0, tzinfo=TZ)


def test_yesterday():
    span = parse_time_range("昨天", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 25, 0, 0, tzinfo=TZ)
    assert span.end == datetime(2026, 8, 26, 0, 0, tzinfo=TZ)


def test_the_day_before_yesterday():
    span = parse_time_range("前天", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 24, 0, 0, tzinfo=TZ)


def test_recent_n_days_includes_today():
    span = parse_time_range("最近3天", now=NOW, tz=TZ)
    # 最近3天 = 24日/25日/26日，含今天
    assert span.start == datetime(2026, 8, 24, 0, 0, tzinfo=TZ)
    assert span.end == datetime(2026, 8, 27, 0, 0, tzinfo=TZ)


def test_recent_n_days_rejects_non_positive():
    with pytest.raises(ValueError, match="正整数"):
        parse_time_range("最近0天", now=NOW, tz=TZ)


def test_this_week_starts_monday():
    span = parse_time_range("本周", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 24, 0, 0, tzinfo=TZ)  # 2026-08-24 是周一
    assert span.end == datetime(2026, 8, 27, 0, 0, tzinfo=TZ)  # 到"明天"（今天24点）


def test_last_week_is_full_seven_days_before_this_monday():
    span = parse_time_range("上周", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 17, 0, 0, tzinfo=TZ)
    assert span.end == datetime(2026, 8, 24, 0, 0, tzinfo=TZ)  # 不含本周一


def test_explicit_iso_date():
    span = parse_time_range("2026-08-01", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 1, 0, 0, tzinfo=TZ)
    assert span.end == datetime(2026, 8, 2, 0, 0, tzinfo=TZ)


def test_unparseable_expression_raises_not_silently_guesses():
    with pytest.raises(ValueError, match="无法解析"):
        parse_time_range("上上个月", now=NOW, tz=TZ)


def test_english_today_yesterday_last_week():
    today = parse_time_range("today", now=NOW, tz=TZ)
    assert today.start == datetime(2026, 8, 26, 0, 0, tzinfo=TZ)
    yesterday = parse_time_range("Yesterday", now=NOW, tz=TZ)
    assert yesterday.start == datetime(2026, 8, 25, 0, 0, tzinfo=TZ)
    last_week = parse_time_range("last week", now=NOW, tz=TZ)
    assert last_week.start == datetime(2026, 8, 17, 0, 0, tzinfo=TZ)
    this_week = parse_time_range("this week", now=NOW, tz=TZ)
    assert this_week.start == datetime(2026, 8, 24, 0, 0, tzinfo=TZ)


def test_english_last_n_days():
    span = parse_time_range("last 3 days", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 24, 0, 0, tzinfo=TZ)
    assert span.end == datetime(2026, 8, 27, 0, 0, tzinfo=TZ)


def test_the_day_before_yesterday_english():
    span = parse_time_range("the day before yesterday", now=NOW, tz=TZ)
    assert span.start == datetime(2026, 8, 24, 0, 0, tzinfo=TZ)


# ---------- 聚合逻辑（注入 entries，不连数据库） ----------

_ENTRIES = [
    {
        "logged_at": "2026-08-26T08:00:00+08:00",
        "meal_type": "早餐",
        "raw_input": "豆浆油条",
        "dishes": [],
        "ingredients": ["豆浆", "油条"],
        "food_properties": ["温"],
    },
    {
        "logged_at": "2026-08-26T12:30:00+08:00",
        "meal_type": "午餐",
        "raw_input": "番茄炒蛋加饭",
        "dishes": [],
        "ingredients": ["番茄", "鸡蛋", "米饭"],
        "food_properties": ["平"],
    },
    {
        "logged_at": "2026-08-25T22:00:00+08:00",
        "meal_type": "夜宵",
        "raw_input": "麻辣烫",
        "dishes": [],
        "ingredients": ["米饭", "各种蔬菜"],
        "food_properties": ["热"],
    },
]


def test_raw_returns_entries_as_is():
    result = query_diet_log("最近3天", aggregation="raw", now=NOW, tz=TZ, entries=_ENTRIES)
    assert result["count"] == 3
    assert result["entries"] == _ENTRIES


def test_raw_respects_limit():
    result = query_diet_log("最近3天", aggregation="raw", limit=1, now=NOW, tz=TZ, entries=_ENTRIES)
    assert len(result["entries"]) == 1
    assert result["count"] == 3  # count 是真实总数，不是截断后的数量


def test_raw_limit_zero_returns_no_entries_not_all():
    # 回归测试：`if limit else entries` 曾经把 limit=0 误判成"没传 limit"
    # （0 是 falsy），导致 limit=0 反而返回全部条目——应该和 by_ingredient 等
    # 聚合用 `Counter.most_common(0)` 的行为一致，返回空。
    result = query_diet_log("最近3天", aggregation="raw", limit=0, now=NOW, tz=TZ, entries=_ENTRIES)
    assert result["entries"] == []
    assert result["count"] == 3  # count 仍是真实总数，不受 limit 影响


def test_by_ingredient_counts_across_entries():
    result = query_diet_log("最近3天", aggregation="by_ingredient", now=NOW, tz=TZ, entries=_ENTRIES)
    breakdown = {row["value"]: row["count"] for row in result["breakdown"]}
    assert breakdown["米饭"] == 2  # 出现在两条记录里
    assert breakdown["豆浆"] == 1


def test_by_property_counts_across_entries():
    result = query_diet_log("最近3天", aggregation="by_property", now=NOW, tz=TZ, entries=_ENTRIES)
    breakdown = {row["value"]: row["count"] for row in result["breakdown"]}
    assert breakdown == {"温": 1, "平": 1, "热": 1}


def test_by_nutrient_not_implemented_not_faked():
    # 明确报错，不是返回一个假的空结果——见模块文档
    with pytest.raises(NotImplementedError, match="营养素"):
        query_diet_log("今天", aggregation="by_nutrient", now=NOW, tz=TZ, entries=_ENTRIES)


def test_by_meal_type_counts_across_entries():
    result = query_diet_log("最近3天", aggregation="by_meal_type", now=NOW, tz=TZ, entries=_ENTRIES)
    breakdown = {row["value"]: row["count"] for row in result["breakdown"]}
    assert breakdown == {"早餐": 1, "午餐": 1, "夜宵": 1}


def test_unknown_aggregation_rejected():
    with pytest.raises(ValueError, match="不在支持范围内"):
        query_diet_log("今天", aggregation="by_calorie", now=NOW, tz=TZ, entries=_ENTRIES)


def test_resolved_range_is_included_for_debuggability():
    result = query_diet_log("今天", now=NOW, tz=TZ, entries=[])
    assert result["resolved_range"]["start"] == "2026-08-26T00:00:00+08:00"
    assert result["resolved_range"]["end"] == "2026-08-27T00:00:00+08:00"


# ---------- 时区优先级链路：user_profile.timezone > DIET_EXPERT_TZ > 默认值(D30) ----------
# 用 monkeypatch 直接换掉 _fetch_user_timezone，不连真实数据库——这是纯逻辑（优先级/
# 容错），跟"能不能连上 Postgres"无关，混在一起测反而两件事都测不干净。

def test_get_tz_prefers_user_profile(monkeypatch):
    monkeypatch.setattr(qdl, "_fetch_user_timezone", lambda user_id, dsn: "Europe/London")
    monkeypatch.setenv("DIET_EXPERT_TZ", "America/New_York")  # 就算环境变量也配了，优先级更低
    tz = _get_tz("someone", None)
    assert tz.key == "Europe/London"


def test_get_tz_falls_back_to_env_when_user_profile_empty(monkeypatch):
    monkeypatch.setattr(qdl, "_fetch_user_timezone", lambda user_id, dsn: None)
    monkeypatch.setenv("DIET_EXPERT_TZ", "America/New_York")
    tz = _get_tz("someone", None)
    assert tz.key == "America/New_York"


def test_get_tz_falls_back_to_default_when_nothing_set(monkeypatch):
    monkeypatch.setattr(qdl, "_fetch_user_timezone", lambda user_id, dsn: None)
    monkeypatch.delenv("DIET_EXPERT_TZ", raising=False)
    tz = _get_tz("someone", None)
    assert tz.key == qdl.DEFAULT_TZ_NAME


def test_get_tz_skips_invalid_timezone_name_and_falls_through(monkeypatch):
    # user_profile 里存了个不是合法 IANA 时区名的值（比如手滑存错）——跳过它，
    # 退到下一层，不整个报错
    monkeypatch.setattr(qdl, "_fetch_user_timezone", lambda user_id, dsn: "Not/ARealZone")
    monkeypatch.setenv("DIET_EXPERT_TZ", "America/New_York")
    tz = _get_tz("someone", None)
    assert tz.key == "America/New_York"


def test_fetch_user_timezone_returns_none_without_dsn(monkeypatch):
    # get_pg_dsn 解析不出连接串时，_fetch_user_timezone 应该安静地返回 None
    # （交给调用方继续往下退），不是抛异常炸掉整个时区解析链路
    monkeypatch.setattr(qdl, "get_pg_dsn", lambda dsn: None)
    assert qdl._fetch_user_timezone("someone", None) is None
