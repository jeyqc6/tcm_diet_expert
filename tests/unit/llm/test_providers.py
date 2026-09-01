"""
测试目标：backend/llm/providers/{openai_compatible,anthropic_provider}.py
——不同服务商各自的请求/响应格式转换 + 错误分类。构造 provider 实例本身不
打网络（SDK client 构造是惰性的，真正发请求才连网络），只在 .call() 这里
直接替换掉内部 client 的方法，避免真的调用外部服务。
对应实现：backend/llm/providers/openai_compatible.py、anthropic_provider.py
"""
import asyncio
from types import SimpleNamespace

from backend.llm.providers.anthropic_provider import DEFAULT_MAX_TOKENS, AnthropicProvider
from backend.llm.providers.base import TokenUsage, classify_http_error
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider


def _run(coro):
    return asyncio.run(coro)


# ---------- OpenAICompatibleProvider ----------

def test_openai_compatible_classify_error_by_status_code():
    provider = OpenAICompatibleProvider(api_key="x", base_url="http://example.invalid/v1", timeout_s=1)

    class _Err(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert provider.classify_error(_Err(429)) == "retryable"
    assert provider.classify_error(_Err(500)) == "retryable"
    assert provider.classify_error(_Err(400)) == "non_retryable"
    assert provider.classify_error(_Err(401)) == "non_retryable"
    assert provider.classify_error(Exception("no status code at all")) == "non_retryable"


def test_classify_http_error_is_the_shared_table():
    class _Err(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert classify_http_error(_Err(429)) == "retryable"
    assert classify_http_error(_Err(503)) == "retryable"
    assert classify_http_error(_Err(401)) == "non_retryable"
    assert classify_http_error(TimeoutError("network")) == "retryable"
    assert classify_http_error(Exception("other")) == "non_retryable"

    class _Conn(Exception):
        pass

    assert classify_http_error(_Conn(), extra_retryable=(_Conn,)) == "retryable"


def test_openai_compatible_call_maps_finish_reason(monkeypatch):
    provider = OpenAICompatibleProvider(api_key="x", base_url="http://example.invalid/v1", timeout_s=1)

    def _fake_response(finish_reason):
        message = SimpleNamespace(content="内容")
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])

    async def _fake_create(**kwargs):
        return _fake_response(kwargs["_finish_reason"])

    # 直接换掉内部 client 的 create 方法，不打真实网络
    provider._client.chat.completions.create = _fake_create

    for finish_reason, expected in [("stop", "stop"), ("length", "max_tokens"), ("content_filter", "content_filter")]:
        resp = _run(
            provider.call([{"role": "user", "content": "hi"}], model="m", _finish_reason=finish_reason)
        )
        assert resp.stop_reason == expected
        assert resp.text == "内容"


def test_openai_compatible_extracts_usage():
    provider = OpenAICompatibleProvider(api_key="x", base_url="http://example.invalid/v1", timeout_s=1)

    async def _fake_create(**kwargs):
        message = SimpleNamespace(content="内容", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
        return SimpleNamespace(choices=[choice], usage=usage)

    provider._client.chat.completions.create = _fake_create
    resp = _run(provider.call([{"role": "user", "content": "hi"}], model="m"))
    assert resp.usage == TokenUsage(input_tokens=11, output_tokens=7, total_tokens=18)


# ---------- AnthropicProvider ----------

def test_anthropic_extracts_system_message_and_sets_default_max_tokens(monkeypatch):
    provider = AnthropicProvider(api_key="fake-key", timeout_s=1)

    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        block = SimpleNamespace(type="text", text="你好")
        return SimpleNamespace(content=[block], stop_reason="end_turn")

    provider._client.messages.create = _fake_create

    messages = [
        {"role": "system", "content": "你是一个中医助手"},
        {"role": "user", "content": "阳虚质吃什么"},
    ]
    resp = _run(provider.call(messages, model="claude-x"))

    assert captured["system"] == "你是一个中医助手"
    assert captured["messages"] == [{"role": "user", "content": "阳虚质吃什么"}]
    assert captured["max_tokens"] == DEFAULT_MAX_TOKENS
    assert resp.text == "你好"
    assert resp.stop_reason == "stop"


def test_anthropic_extracts_usage():
    provider = AnthropicProvider(api_key="fake-key", timeout_s=1)

    async def _fake_create(**kwargs):
        block = SimpleNamespace(type="text", text="你好")
        usage = SimpleNamespace(input_tokens=9, output_tokens=4)
        return SimpleNamespace(content=[block], stop_reason="end_turn", usage=usage)

    provider._client.messages.create = _fake_create
    resp = _run(provider.call([{"role": "user", "content": "hi"}], model="claude-x"))
    assert resp.usage == TokenUsage(input_tokens=9, output_tokens=4, total_tokens=13)


def test_anthropic_no_system_message_omits_system_kwarg(monkeypatch):
    provider = AnthropicProvider(api_key="fake-key", timeout_s=1)
    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[], stop_reason="end_turn")

    provider._client.messages.create = _fake_create
    _run(provider.call([{"role": "user", "content": "hi"}], model="claude-x"))
    assert "system" not in captured


def test_anthropic_explicit_max_tokens_not_overridden(monkeypatch):
    provider = AnthropicProvider(api_key="fake-key", timeout_s=1)
    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[], stop_reason="max_tokens")

    provider._client.messages.create = _fake_create
    resp = _run(provider.call([{"role": "user", "content": "hi"}], model="claude-x", max_tokens=64))
    assert captured["max_tokens"] == 64
    assert resp.stop_reason == "max_tokens"


def test_anthropic_classify_error_by_status_code():
    provider = AnthropicProvider(api_key="fake-key", timeout_s=1)

    class _Err(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert provider.classify_error(_Err(429)) == "retryable"
    assert provider.classify_error(_Err(529)) == "retryable"  # Anthropic 的"过载"状态码
    assert provider.classify_error(_Err(400)) == "non_retryable"


# ---------- 工具调用(tool_use)——backend/agents/router.py Agent Loop 依赖这一段 ----------

_TOOLS = [
    {
        "name": "query_weather",
        "description": "weather lookup",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def test_openai_compatible_translates_tools_and_parses_tool_calls():
    provider = OpenAICompatibleProvider(api_key="x", base_url="http://example.invalid/v1", timeout_s=1)
    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="query_weather", arguments='{"city": "北京"}'),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        return SimpleNamespace(choices=[choice])

    provider._client.chat.completions.create = _fake_create
    resp = _run(provider.call([{"role": "user", "content": "北京天气"}], model="m", tools=_TOOLS))

    # tools 被翻译成 OpenAI 的 function-calling 包装形状
    assert captured["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "query_weather",
                "description": "weather lookup",
                "parameters": _TOOLS[0]["input_schema"],
            },
        }
    ]
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_1"
    assert resp.tool_calls[0].name == "query_weather"
    assert resp.tool_calls[0].arguments == {"city": "北京"}


def test_openai_compatible_translates_assistant_tool_calls_and_tool_result_messages():
    provider = OpenAICompatibleProvider(api_key="x", base_url="http://example.invalid/v1", timeout_s=1)
    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="收到", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])

    provider._client.chat.completions.create = _fake_create
    messages = [
        {"role": "user", "content": "北京天气"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "name": "query_weather", "arguments": {"city": "北京"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "query_weather", "content": '{"temp": 20}'},
    ]
    resp = _run(provider.call(messages, model="m"))

    assert captured["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "query_weather", "arguments": '{"city": "北京"}'},
            }
        ],
    }
    # role="tool" 归一化消息里多余的 "name" 字段被丢弃——OpenAI 的 tool 消息不认这个字段
    assert captured["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"temp": 20}',
    }
    assert resp.tool_calls is None  # 这一轮没有新的 tool_use


def test_anthropic_translates_tools_and_parses_tool_use_blocks():
    provider = AnthropicProvider(api_key="fake-key", timeout_s=1)
    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        block = SimpleNamespace(type="tool_use", id="toolu_1", name="query_weather", input={"city": "北京"})
        return SimpleNamespace(content=[block], stop_reason="tool_use")

    provider._client.messages.create = _fake_create
    resp = _run(provider.call([{"role": "user", "content": "北京天气"}], model="claude-x", tools=_TOOLS))

    # Anthropic 原生 tool 格式和归一化格式一致，原样传入不用翻译
    assert captured["tools"] == _TOOLS
    assert resp.stop_reason == "tool_use"
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "toolu_1"
    assert resp.tool_calls[0].name == "query_weather"
    assert resp.tool_calls[0].arguments == {"city": "北京"}


def test_anthropic_translates_assistant_tool_calls_and_merges_tool_results_into_one_user_message():
    provider = AnthropicProvider(api_key="fake-key", timeout_s=1)
    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        block = SimpleNamespace(type="text", text="收到")
        return SimpleNamespace(content=[block], stop_reason="end_turn")

    provider._client.messages.create = _fake_create
    messages = [
        {"role": "user", "content": "查两个城市天气"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "name": "query_weather", "arguments": {"city": "北京"}},
                {"id": "call_2", "name": "query_weather", "arguments": {"city": "上海"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "query_weather", "content": '{"temp": 20}'},
        {"role": "tool", "tool_call_id": "call_2", "name": "query_weather", "content": '{"temp": 25}'},
    ]
    _run(provider.call(messages, model="claude-x"))

    assert captured["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "query_weather", "input": {"city": "北京"}},
            {"type": "tool_use", "id": "call_2", "name": "query_weather", "input": {"city": "上海"}},
        ],
    }
    # 两条归一化 role="tool" 消息合并进同一条 user 消息(Anthropic API 的硬性要求)
    assert captured["messages"][2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": '{"temp": 20}'},
            {"type": "tool_result", "tool_use_id": "call_2", "content": '{"temp": 25}'},
        ],
    }
