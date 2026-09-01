"""
测试目标：计数正确性、要点抽取不携带原始检索文本、边界值（恰好15次）
对应实现：backend/memory/status_prompt.py
覆盖要求：建议按 Critical 档同等严格——D27 点名"状态栏投毒"风险的那段代码，
这里穷举边界而不是抽样。
"""
from backend.memory.status_prompt import (
    ToolCallSummary,
    build_status_message,
    extract_tool_call_summaries,
    format_status_prompt,
    summarize_tool_call,
)

_HUGE_RETRIEVED_TEXT = "阳虚质忌生冷" * 500  # 模拟一段真实检索原文，篇幅很大


def _assistant_tool_call_msg(calls):
    return {"role": "assistant", "content": None, "tool_calls": calls}


def _tool_result_msg(call_id, name, content):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content, "ok": True}


# ---------- summarize_tool_call：按工具挑关键参数 ----------

def test_summarize_known_tool_picks_only_key_fields():
    summary = summarize_tool_call("retrieve_tcm", {"query": "气虚质吃什么", "top_k": 5, "filters": {"a": 1}})
    assert summary.tool_name == "retrieve_tcm"
    assert summary.key_args == {"query": "气虚质吃什么"}  # top_k/filters 被过滤掉


def test_summarize_query_weather_keeps_city_and_date():
    summary = summarize_tool_call("query_weather", {"city": "北京", "date": "2026-08-27", "include_recent_days": 3})
    assert summary.key_args == {"city": "北京", "date": "2026-08-27"}


def test_summarize_unknown_tool_falls_back_to_all_args():
    summary = summarize_tool_call("some_future_tool", {"foo": "bar"})
    assert summary.key_args == {"foo": "bar"}


def test_summarize_missing_arguments_is_empty_not_error():
    summary = summarize_tool_call("retrieve_tcm", None)
    assert summary.key_args == {}


# ---------- extract_tool_call_summaries：只看 assistant.tool_calls，不看 tool 结果 ----------

def test_extract_ignores_tool_result_content():
    messages = [
        {"role": "user", "content": "气虚质吃什么"},
        _assistant_tool_call_msg([{"id": "c1", "name": "retrieve_tcm", "arguments": {"query": "气虚质"}}]),
        _tool_result_msg("c1", "retrieve_tcm", _HUGE_RETRIEVED_TEXT),
    ]
    summaries = extract_tool_call_summaries(messages)
    assert len(summaries) == 1
    assert summaries[0] == ToolCallSummary(tool_name="retrieve_tcm", key_args={"query": "气虚质"})


def test_extract_preserves_order_across_multiple_rounds():
    messages = [
        _assistant_tool_call_msg([{"id": "c1", "name": "retrieve_tcm", "arguments": {"query": "第一次"}}]),
        _tool_result_msg("c1", "retrieve_tcm", "..."),
        _assistant_tool_call_msg([{"id": "c2", "name": "query_weather", "arguments": {"city": "北京"}}]),
        _tool_result_msg("c2", "query_weather", "..."),
    ]
    summaries = extract_tool_call_summaries(messages)
    assert [s.tool_name for s in summaries] == ["retrieve_tcm", "query_weather"]
    assert summaries[0].key_args == {"query": "第一次"}


def test_extract_handles_multiple_calls_in_one_round():
    messages = [
        _assistant_tool_call_msg(
            [
                {"id": "c1", "name": "query_weather", "arguments": {"city": "北京"}},
                {"id": "c2", "name": "query_weather", "arguments": {"city": "上海"}},
            ]
        ),
    ]
    summaries = extract_tool_call_summaries(messages)
    assert len(summaries) == 2
    assert [s.key_args["city"] for s in summaries] == ["北京", "上海"]


def test_extract_empty_messages_returns_empty():
    assert extract_tool_call_summaries([]) == []


def test_extract_ignores_assistant_messages_without_tool_calls():
    messages = [{"role": "assistant", "content": "最终回答，没有工具调用"}]
    assert extract_tool_call_summaries(messages) == []


# ---------- format_status_prompt：格式 + 边界值 ----------

def test_format_no_calls_yet():
    text = format_status_prompt(0, 15, [])
    assert text == "[状态] 已用工具调用:0/15(资源限额,§5.4)· 已检索到的候选信息要点:（还没有）"


def test_format_with_summaries_joins_with_dun_hao():
    summaries = [
        ToolCallSummary("retrieve_tcm", {"query": "气虚质"}),
        ToolCallSummary("query_weather", {"city": "北京"}),
    ]
    text = format_status_prompt(2, 15, summaries)
    assert "已用工具调用:2/15" in text
    assert "retrieve_tcm(query=气虚质)、query_weather(city=北京)" in text


def test_format_boundary_exactly_at_cap():
    # 恰好 15 次——不是 14 也不是 16，边界值本身要精确
    text = format_status_prompt(15, 15, [])
    assert "已用工具调用:15/15" in text


def test_format_boundary_one_below_cap():
    text = format_status_prompt(14, 15, [])
    assert "已用工具调用:14/15" in text


def test_format_tool_with_no_key_args_shows_bare_name():
    summary = ToolCallSummary("write_memory", {})
    text = format_status_prompt(1, 15, [summary])
    assert "write_memory" in text
    assert "write_memory(" not in text  # 没有参数就不加括号


# ---------- build_status_message：可直接 append 的消息 ----------

def test_build_status_message_uses_user_role_not_system():
    # role="system" 会被 anthropic_provider._translate_messages 抽到最前面合并，
    # 破坏"必须在最新位置"这条硬约束——见模块文档。
    msg = build_status_message([], tool_call_count=0, max_tool_calls=15)
    assert msg["role"] == "user"


def test_build_status_message_never_contains_raw_retrieved_text():
    messages = [
        _assistant_tool_call_msg([{"id": "c1", "name": "retrieve_tcm", "arguments": {"query": "气虚质"}}]),
        _tool_result_msg("c1", "retrieve_tcm", _HUGE_RETRIEVED_TEXT),
    ]
    msg = build_status_message(messages, tool_call_count=1, max_tool_calls=15)
    assert _HUGE_RETRIEVED_TEXT not in msg["content"]
    assert "气虚质" in msg["content"]  # 查询参数本身(问了什么)可以出现，那不是检索原文
