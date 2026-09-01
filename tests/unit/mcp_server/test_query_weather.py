"""query_weather: cache, circuit, solar-term fallback. HTTP is mocked."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.mcp_server.tools.query_weather import (
    CACHE_TTL_S,
    query_weather,
    reset_weather_state_for_tests,
    solar_term_for,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_weather_state_for_tests()
    yield
    reset_weather_state_for_tests()


def _forecast_ok(*_args, **_kwargs):
    def fetcher(url: str) -> dict:
        if "geocoding-api" in url:
            return {"results": [{"latitude": 31.23, "longitude": 121.47}]}
        return {
            "timezone": "Asia/Shanghai",
            "daily": {
                "time": ["2026-08-29"],
                "temperature_2m_max": [32.0],
                "temperature_2m_min": [24.0],
                "precipitation_sum": [0.0],
                "weather_code": [1],
            },
        }

    return fetcher


def test_solar_term_around_lichun():
    from datetime import date

    term = solar_term_for(date(2026, 2, 4))
    assert term["solar_term"] == "立春"
    assert term["season"] == "春"


def test_open_meteo_success_uses_builtin_city_coords():
    result = query_weather(
        "上海",
        date="2026-08-29",
        http_get_json=_forecast_ok(),
        now=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )
    assert result["source"] == "open_meteo"
    assert result["city"] == "上海"
    assert result["daily"][0]["temp_max_c"] == 32.0
    assert result["cached"] is False
    assert result["circuit_open"] is False


def test_english_weather_uses_english_geocoding_and_calendar_labels():
    calls: list[str] = []

    def fetcher(url: str) -> dict:
        calls.append(url)
        if "geocoding-api" in url:
            return {"results": [{"latitude": 31.23, "longitude": 121.47}]}
        return {
            "timezone": "Asia/Shanghai",
            "daily": {
                "time": ["2026-02-04"],
                "temperature_2m_max": [10.0],
                "temperature_2m_min": [2.0],
                "precipitation_sum": [0.0],
                "weather_code": [1],
            },
        }

    result = query_weather(
        "An unknown city",
        date="2026-02-04",
        language="en",
        http_get_json=fetcher,
    )

    assert "language=en" in calls[0]
    assert result["solar_term"] == "Start of Spring"
    assert result["season"] == "spring"


def test_three_hour_cache_returns_same_payload_without_second_http():
    calls = {"n": 0}

    def fetcher(url: str) -> dict:
        calls["n"] += 1
        return _forecast_ok()(url)

    first = query_weather("北京", date="2026-08-29", http_get_json=fetcher)
    second = query_weather("北京", date="2026-08-29", http_get_json=fetcher)
    assert first["source"] == "open_meteo"
    assert second["cached"] is True
    assert calls["n"] == 1
    assert CACHE_TTL_S == 3 * 60 * 60


def test_cache_disabled_env_bypasses_cache(monkeypatch):
    monkeypatch.setenv("CACHE_DISABLED", "1")
    calls = {"n": 0}

    def fetcher(url: str) -> dict:
        calls["n"] += 1
        return _forecast_ok()(url)

    query_weather("北京", date="2026-08-29", http_get_json=fetcher)
    query_weather("北京", date="2026-08-29", http_get_json=fetcher)
    assert calls["n"] == 2


def test_three_consecutive_failures_open_circuit_and_use_solar_term():
    def boom(_url: str) -> dict:
        raise TimeoutError("open-meteo down")

    first = query_weather("未知城", date="2026-08-29", http_get_json=boom)
    assert first["source"] == "solar_term_fallback"
    assert first["fallback_reason"] == "fetch_failed"
    query_weather("未知城", date="2026-08-29", http_get_json=boom)
    third = query_weather("未知城", date="2026-08-29", http_get_json=boom)
    assert third["circuit_open"] is True
    assert third["solar_term"]
    assert third["season"]

    # Fourth call must not hit HTTP once the circuit is open.
    def must_not_run(_url: str) -> dict:
        raise AssertionError("circuit should skip HTTP")

    fourth = query_weather("未知城", date="2026-08-29", http_get_json=must_not_run)
    assert fourth["source"] == "solar_term_fallback"
    assert fourth["circuit_open"] is True


def test_success_resets_circuit():
    def boom(_url: str) -> dict:
        raise TimeoutError("down")

    query_weather("未知城", date="2026-08-29", http_get_json=boom)
    query_weather("上海", date="2026-08-29", http_get_json=_forecast_ok())
    # One failure + one success: circuit stays closed; next failure is not trip 3.
    again = query_weather("未知城", date="2026-08-30", http_get_json=boom)
    assert again["circuit_open"] is False


def test_rejects_empty_city():
    with pytest.raises(ValueError):
        query_weather("  ", http_get_json=_forecast_ok())


def test_rejects_bad_date():
    with pytest.raises(ValueError):
        query_weather("上海", date="08/29", http_get_json=_forecast_ok())


def test_accepts_relative_today_aliases():
    clock = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    for expr in ("today", "Today", "今天"):
        result = query_weather(
            "上海",
            date=expr,
            http_get_json=_forecast_ok(),
            now=clock,
        )
        assert result["date"] == "2026-08-29"


def test_accepts_relative_yesterday_and_tomorrow():
    clock = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    yesterday = query_weather(
        "上海",
        date="yesterday",
        http_get_json=_forecast_ok(),
        now=clock,
    )
    tomorrow = query_weather(
        "上海",
        date="明天",
        http_get_json=_forecast_ok(),
        now=clock,
    )
    assert yesterday["date"] == "2026-08-28"
    assert tomorrow["date"] == "2026-08-30"


def test_relative_today_uses_user_timezone(monkeypatch):
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(
        "backend.mcp_server.tools.query_weather._get_tz",
        lambda user_id, dsn=None: ZoneInfo("America/Los_Angeles"),
    )
    # 2026-08-29 02:00 UTC is still 2026-08-28 in Los Angeles.
    clock = datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc)
    result = query_weather(
        "上海",
        date="today",
        http_get_json=_forecast_ok(),
        now=clock,
    )
    assert result["date"] == "2026-08-28"


def test_accepts_slash_and_chinese_date_formats():
    clock = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    slash = query_weather(
        "上海",
        date="2026/08/29",
        http_get_json=_forecast_ok(),
        now=clock,
    )
    chinese = query_weather(
        "上海",
        date="2026年8月29日",
        http_get_json=_forecast_ok(),
        now=clock,
    )
    assert slash["date"] == "2026-08-29"
    assert chinese["date"] == "2026-08-29"
