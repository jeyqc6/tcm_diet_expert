#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-Meteo weather lookup with a 3h cache and solar-term fallback.

ARCHITECTURE §2.2 · ENGINEERING §1.3 / §3 · PRD §11 (weather API fail → calendar).

Bounded by design: stdlib urllib, no weather SDK. HTTP is injectable so unit
tests never hit the network. Circuit opens after 3 consecutive failures and
returns the solar-term table silently until a later successful fetch.

Relative dates (today/yesterday/明天) resolve in the same timezone chain as
`query_diet_log`: user_profile.timezone > DIET_EXPERT_TZ > Asia/Shanghai.
`user_id` is injected by the MCP session (not part of the public tool schema).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date as date_cls
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from backend.i18n import current_locale, normalize_locale
from backend.mcp_server.tools.query_diet_log import _get_tz

JsonFetcher = Callable[[str], dict[str, Any]]

CACHE_TTL_S = 3 * 60 * 60
CIRCUIT_FAILURE_THRESHOLD = 3
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Common cities so a missing geocode hop still works offline / after circuit.
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "成都": (30.5728, 104.0668),
    "杭州": (30.2741, 120.1551),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "南京": (32.0603, 118.7969),
    "重庆": (29.4316, 106.9123),
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
}

# Approximate Gregorian day-of-year midpoints for the 24 solar terms.
# Fallback only — not an astronomical ephemeris.
_SOLAR_TERMS: tuple[tuple[int, str, str], ...] = (
    (5, "小寒", "冬"),
    (20, "大寒", "冬"),
    (34, "立春", "春"),
    (49, "雨水", "春"),
    (64, "惊蛰", "春"),
    (80, "春分", "春"),
    (95, "清明", "春"),
    (110, "谷雨", "春"),
    (125, "立夏", "夏"),
    (141, "小满", "夏"),
    (156, "芒种", "夏"),
    (172, "夏至", "夏"),
    (188, "小暑", "夏"),
    (204, "大暑", "夏"),
    (219, "立秋", "秋"),
    (235, "处暑", "秋"),
    (250, "白露", "秋"),
    (266, "秋分", "秋"),
    (281, "寒露", "秋"),
    (296, "霜降", "秋"),
    (311, "立冬", "冬"),
    (326, "小雪", "冬"),
    (341, "大雪", "冬"),
    (356, "冬至", "冬"),
)

_lock = threading.Lock()
_cache: dict[tuple[str, str, int, str], tuple[float, dict[str, Any]]] = {}
_consecutive_failures = 0


def reset_weather_state_for_tests() -> None:
    """Unit-test helper: drop cache + circuit so cases do not leak."""
    global _consecutive_failures
    with _lock:
        _cache.clear()
        _consecutive_failures = 0


def _cache_disabled() -> bool:
    return os.environ.get("CACHE_DISABLED", "").strip() in {"1", "true", "TRUE", "yes"}


def _normalize_city(city: str) -> str:
    return city.strip()


_RELATIVE_DAY_OFFSETS = {
    "今天": 0,
    "今日": 0,
    "today": 0,
    "昨天": 1,
    "昨日": 1,
    "yesterday": 1,
    "前天": 2,
    "the day before yesterday": 2,
    "明天": -1,
    "明日": -1,
    "tomorrow": -1,
    "后天": -2,
    "the day after tomorrow": -2,
}
_ISO_DATE_PATTERN = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$")
_CN_DATE_PATTERN = re.compile(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$")


def _clock_in_weather_tz(now: datetime | None, tz: ZoneInfo) -> datetime:
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _parse_date(value: str | None, *, now: datetime) -> date_cls:
    if value is None or not str(value).strip():
        return now.date()

    expr = str(value).strip()
    key = expr.casefold()
    if key in _RELATIVE_DAY_OFFSETS:
        return now.date() - timedelta(days=_RELATIVE_DAY_OFFSETS[key])

    try:
        return date_cls.fromisoformat(expr)
    except ValueError:
        pass

    for pattern in (_ISO_DATE_PATTERN, _CN_DATE_PATTERN):
        match = pattern.fullmatch(expr)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return date_cls(year, month, day)

    raise ValueError(
        "date must be YYYY-MM-DD or a relative day "
        "(today/yesterday/tomorrow/今天/昨天/明天)"
    )


def solar_term_for(day: date_cls) -> dict[str, str]:
    doy = day.timetuple().tm_yday
    chosen = _SOLAR_TERMS[0]
    for start, name, season in _SOLAR_TERMS:
        if doy >= start:
            chosen = (start, name, season)
        else:
            break
    return {"solar_term": chosen[1], "season": chosen[2]}


def _solar_term_payload(
    city: str,
    day: date_cls,
    include_recent_days: int,
    *,
    reason: str,
    language: str = "zh",
) -> dict[str, Any]:
    term = solar_term_for(day)
    payload = {
        "city": city,
        "date": day.isoformat(),
        "source": "solar_term_fallback",
        "fallback_reason": reason,
        "cached": False,
        "circuit_open": reason == "circuit_open",
        "include_recent_days": include_recent_days,
        **term,
        "daily": [],
    }
    return _localize_weather_payload(payload, language)


_EN_SOLAR_TERMS = {
    "小寒": "Minor Cold",
    "大寒": "Major Cold",
    "立春": "Start of Spring",
    "雨水": "Rain Water",
    "惊蛰": "Awakening of Insects",
    "春分": "Spring Equinox",
    "清明": "Clear and Bright",
    "谷雨": "Grain Rain",
    "立夏": "Start of Summer",
    "小满": "Grain Full",
    "芒种": "Grain in Ear",
    "夏至": "Summer Solstice",
    "小暑": "Minor Heat",
    "大暑": "Major Heat",
    "立秋": "Start of Autumn",
    "处暑": "End of Heat",
    "白露": "White Dew",
    "秋分": "Autumn Equinox",
    "寒露": "Cold Dew",
    "霜降": "Frost's Descent",
    "立冬": "Start of Winter",
    "小雪": "Minor Snow",
    "大雪": "Major Snow",
    "冬至": "Winter Solstice",
}
_EN_SEASONS = {"春": "spring", "夏": "summer", "秋": "autumn", "冬": "winter"}


def _localize_weather_payload(payload: dict[str, Any], language: str) -> dict[str, Any]:
    """Translate calendar labels while preserving the machine-readable fields."""
    if language != "en":
        return payload
    localized = dict(payload)
    if localized.get("solar_term") in _EN_SOLAR_TERMS:
        localized["solar_term"] = _EN_SOLAR_TERMS[localized["solar_term"]]
    if localized.get("season") in _EN_SEASONS:
        localized["season"] = _EN_SEASONS[localized["season"]]
    return localized


def _default_http_get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "diet_expert/query_weather"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _geocode(
    city: str, fetcher: JsonFetcher, language: str = "zh"
) -> tuple[float, float] | None:
    key = city.casefold()
    if city in _CITY_COORDS:
        return _CITY_COORDS[city]
    if key in _CITY_COORDS:
        return _CITY_COORDS[key]
    params = urllib.parse.urlencode({"name": city, "count": 1, "language": language})
    data = fetcher(f"{GEOCODE_URL}?{params}")
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    return float(first["latitude"]), float(first["longitude"])


def _fetch_forecast(
    lat: float,
    lon: float,
    day: date_cls,
    include_recent_days: int,
    fetcher: JsonFetcher,
) -> dict[str, Any]:
    past = max(0, min(int(include_recent_days), 7))
    start = day - timedelta(days=past)
    params = urllib.parse.urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start.isoformat(),
            "end_date": day.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "auto",
        }
    )
    data = fetcher(f"{FORECAST_URL}?{params}")
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    rows = []
    for i, stamp in enumerate(times):
        rows.append(
            {
                "date": stamp,
                "temp_max_c": _maybe_float((daily.get("temperature_2m_max") or [None])[i : i + 1]),
                "temp_min_c": _maybe_float((daily.get("temperature_2m_min") or [None])[i : i + 1]),
                "precipitation_mm": _maybe_float(
                    (daily.get("precipitation_sum") or [None])[i : i + 1]
                ),
                "weather_code": (daily.get("weather_code") or [None])[i] if i < len(times) else None,
            }
        )
    return {
        "latitude": lat,
        "longitude": lon,
        "timezone": data.get("timezone"),
        "daily": rows,
    }


def _maybe_float(values: list[Any]) -> float | None:
    if not values or values[0] is None:
        return None
    return float(values[0])


def _record_success() -> None:
    global _consecutive_failures
    with _lock:
        _consecutive_failures = 0


def _record_failure() -> int:
    global _consecutive_failures
    with _lock:
        _consecutive_failures += 1
        return _consecutive_failures


def _circuit_open() -> bool:
    with _lock:
        return _consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD


def query_weather(
    city: str,
    date: str | None = None,
    include_recent_days: int = 3,
    *,
    user_id: str = "default_user",
    dsn: str | None = None,
    http_get_json: JsonFetcher | None = None,
    now: datetime | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")
    if not isinstance(include_recent_days, int) or isinstance(include_recent_days, bool):
        raise ValueError("include_recent_days must be an integer")
    if include_recent_days < 0:
        raise ValueError("include_recent_days must be >= 0")

    resolved_city = _normalize_city(city)
    tz = _get_tz(user_id, dsn)
    clock = _clock_in_weather_tz(now, tz)
    day = _parse_date(date, now=clock)
    resolved_language = normalize_locale(language if language is not None else current_locale())
    cache_key = (resolved_city.casefold(), day.isoformat(), include_recent_days, resolved_language)
    fetcher = http_get_json or _default_http_get_json

    if not _cache_disabled():
        with _lock:
            hit = _cache.get(cache_key)
        if hit is not None:
            expires_at, payload = hit
            if expires_at > time.time():
                cached = dict(payload)
                cached["cached"] = True
                return cached

    if _circuit_open():
        return _solar_term_payload(
            resolved_city,
            day,
            include_recent_days,
            reason="circuit_open",
            language=resolved_language,
        )

    try:
        coords = _geocode(resolved_city, fetcher, language=resolved_language)
        if coords is None:
            raise RuntimeError("geocode_miss")
        forecast = _fetch_forecast(
            coords[0], coords[1], day, include_recent_days, fetcher
        )
    except Exception:
        failures = _record_failure()
        reason = "circuit_open" if failures >= CIRCUIT_FAILURE_THRESHOLD else "fetch_failed"
        return _solar_term_payload(
            resolved_city,
            day,
            include_recent_days,
            reason=reason,
            language=resolved_language,
        )

    _record_success()
    term = solar_term_for(day)
    payload = _localize_weather_payload({
        "city": resolved_city,
        "date": day.isoformat(),
        "source": "open_meteo",
        "fallback_reason": None,
        "cached": False,
        "circuit_open": False,
        "include_recent_days": include_recent_days,
        **term,
        **forecast,
    }, resolved_language)
    if not _cache_disabled():
        with _lock:
            _cache[cache_key] = (time.time() + CACHE_TTL_S, dict(payload))
    return payload
