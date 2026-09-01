"""
测试目标：资源限额（≤15次工具调用）触发终止、循环防护（连续3轮无新增信息）、
两侧 SubAgent 的领域隔离——打日志确认 TCM 上下文里没有营养学检索内容,反之亦然。
对应实现：backend/agents/{tcm_subagent,nutrition_subagent,_subagent_common}.py，
复用 backend/agents/agent_loop.py 的 run_agent_loop()。
覆盖要求：集成测试，注入假 complete()（同 tests/unit/agents/test_agent_loop.py 的模式），
不打真实网络/LLM；MCP server 用真实 DietExpertMcpServer + stub 工具 handler。
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from backend.agents.agent_loop import AgentLoopResourceLimitError
from backend.agents.nutrition_subagent import run_nutrition_subagent
from backend.agents.tcm_subagent import run_tcm_subagent
from backend.llm.adapter import LLMResult, ModelTier
from backend.llm.providers.base import ToolCall
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.server import DietExpertMcpServer


def _run(coro):
    return asyncio.run(coro)


def _server() -> DietExpertMcpServer:
    """Handler 按工具名回一个可辨认的领域标记，方便断言"没有跨领域内容泄漏"。"""
    base = default_tool_definitions()
    stubs: dict[str, ToolDefinition] = {}
    for name, tool in base.items():
        stubs[name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=lambda _n=name, **kwargs: {"tool": _n, "domain_marker": _n, "args": kwargs},
        )
    return DietExpertMcpServer(tools=stubs)


def _result(text="", tool_calls=None) -> LLMResult:
    return LLMResult(text=text, model="m", tier=ModelTier.DEV, provider="fake", tool_calls=tool_calls)


class _ScriptedComplete:
    def __init__(self, script: list[LLMResult]):
        self._script = list(script)
        self.calls = 0

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.calls += 1
        return self._script.pop(0)


# ---------------------------------------------------------------------------
# 领域隔离：TCM 上下文里没有营养学检索内容，反之亦然
# ---------------------------------------------------------------------------

def test_tcm_subagent_only_ever_calls_tcm_tools(caplog):
    complete = _ScriptedComplete(
        [
            _result(tool_calls=[ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "阳虚质"})]),
            _result(text="阳虚质应少食生冷。"),
        ]
    )
    with caplog.at_level(logging.INFO, logger="diet_expert.agents.subagent"):
        result = _run(run_tcm_subagent("阳虚质吃什么", _server(), complete=complete))

    assert result.tools_called == ["retrieve_tcm"]
    assert "retrieve_nutrition" not in result.tools_called
    tool_result_payloads = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert all("retrieve_nutrition" not in payload for payload in tool_result_payloads)

    # BUILD_PLAN 完成判据要求"打日志确认"——日志里应该能看到这一侧调用过的工具列表
    done_logs = [r.message for r in caplog.records if "loop done" in r.message]
    assert done_logs, "缺少循环结束日志"
    assert "retrieve_tcm" in done_logs[0]
    assert "retrieve_nutrition" not in done_logs[0]


def test_nutrition_subagent_only_ever_calls_nutrition_tools(caplog):
    complete = _ScriptedComplete(
        [
            _result(
                tool_calls=[ToolCall(id="c1", name="retrieve_nutrition", arguments={"query": "补铁"})]
            ),
            _result(text="补铁可以多吃红肉。"),
        ]
    )
    with caplog.at_level(logging.INFO, logger="diet_expert.agents.subagent"):
        result = _run(run_nutrition_subagent("怎么补铁", _server(), complete=complete))

    assert result.tools_called == ["retrieve_nutrition"]
    assert "retrieve_tcm" not in result.tools_called
    tool_result_payloads = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert all("retrieve_tcm" not in payload for payload in tool_result_payloads)

    done_logs = [r.message for r in caplog.records if "loop done" in r.message]
    assert done_logs
    assert "retrieve_nutrition" in done_logs[0]
    assert "retrieve_tcm" not in done_logs[0]


def test_tcm_subagent_cannot_reach_nutrition_tool_even_if_model_tries():
    """协议层隔离（§2.3）是真正的边界：即便模型"想"调用对方领域的工具，
    MCP session 里根本看不到那个工具，越权调用会被拒绝而不是执行。"""
    complete = _ScriptedComplete(
        [
            _result(
                tool_calls=[
                    ToolCall(id="c1", name="retrieve_nutrition", arguments={"query": "维生素C"})
                ]
            ),
            _result(text="我没有权限查营养学知识库。"),
        ]
    )
    server = _server()
    result = _run(run_tcm_subagent("维生素C怎么补", server, complete=complete))

    assert result.tools_called == []  # 越权调用没有真正执行到 handler
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "error" in tool_messages[0]["content"]
    assert len(server.unauthorized_attempts) == 1


# ---------------------------------------------------------------------------
# 资源限额（≤15 次工具调用）触发终止
# ---------------------------------------------------------------------------

def test_resource_limit_terminates_after_fifteen_tool_calls():
    # 每次查询不同的 city，确保每一轮都算"新信息"，不会先被循环防护拦下——
    # 这样才能确认真正触发终止的是资源限额，不是别的路径。
    script = [
        _result(tool_calls=[ToolCall(id=f"c{i}", name="query_weather", arguments={"city": f"城市{i}"})])
        for i in range(20)
    ]
    complete = _ScriptedComplete(script)

    with pytest.raises(AgentLoopResourceLimitError, match="15"):
        _run(run_tcm_subagent("连续查天气", _server(), complete=complete))


def test_resource_limit_is_fifteen_not_router_default():
    """SubAgent 的限额是 §5.4 明确的 15 次，不是 router.py 中枢自己的 50 次兜底。"""
    from backend.agents._subagent_common import SUBAGENT_MAX_TOOL_CALLS

    assert SUBAGENT_MAX_TOOL_CALLS == 15


# ---------------------------------------------------------------------------
# 循环防护：连续 3 轮无新增信息
# ---------------------------------------------------------------------------

def test_stall_guard_terminates_after_three_repeated_rounds(caplog):
    # 同一个工具、同样的参数，重复调用 5 次——第 3 次重复(累计第 4 轮)应该
    # 触发循环防护，而不是耗到资源限额。
    repeated_call = ToolCall(id="c", name="retrieve_tcm", arguments={"query": "阳虚质"})
    script = [_result(tool_calls=[repeated_call]) for _ in range(5)]
    complete = _ScriptedComplete(script)

    with caplog.at_level(logging.WARNING, logger="diet_expert.agents.subagent"):
        with caplog.at_level(logging.WARNING, logger="diet_expert.agents.agent_loop"):
            result = _run(run_tcm_subagent("阳虚质吃什么", _server(), complete=complete))

    assert result.terminated_reason == "stall_guard"
    assert result.tool_call_count == 4  # 1 次产生新信息 + 3 次重复触发防护
    assert complete.calls == 4  # 没有为了收尾再多问模型一次
    stall_logs = [r.message for r in caplog.records if "stall guard triggered" in r.message]
    assert stall_logs


# ---------------------------------------------------------------------------
# SubAgent 循环状态提示（ARCHITECTURE §4.5，backend/memory/status_prompt.py）
# 只在这一处（SubAgent 的开放式循环）挂载，中枢自己的 run_agent_loop 调用不传
# 这个 hook——见 backend/agents/router.py `before_next_call` 的模块文档。
# ---------------------------------------------------------------------------

class _RecordingComplete:
    """记录每次调用时收到的完整 messages，用来断言状态提示确实被追加、且位置
    在最新（喂给下一次 complete() 的那一轮），不是随便某个位置。"""

    def __init__(self, script: list[LLMResult]):
        self._script = list(script)
        self.calls: list[list[dict]] = []

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.calls.append(list(messages))
        return self._script.pop(0)


def test_subagent_loop_injects_status_prompt_with_correct_count():
    complete = _RecordingComplete(
        [
            _result(tool_calls=[ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "阳虚质"})]),
            _result(text="阳虚质建议温阳祛寒。"),
        ]
    )

    _run(run_tcm_subagent("阳虚质吃什么", _server(), complete=complete))

    assert len(complete.calls) == 2

    # 第一次调用前：还没发生任何工具调用，0/15
    first_call_last_msg = complete.calls[0][-1]
    assert first_call_last_msg["role"] == "user"
    assert "已用工具调用:0/15" in first_call_last_msg["content"]
    assert "（还没有）" in first_call_last_msg["content"]

    # 第二次调用前：已经真实执行了一次 retrieve_tcm，计数和要点都要对
    second_call_last_msg = complete.calls[1][-1]
    assert second_call_last_msg["role"] == "user"
    assert "已用工具调用:1/15" in second_call_last_msg["content"]
    assert "retrieve_tcm(query=阳虚质)" in second_call_last_msg["content"]

    # 不携带检索原文——真正的工具执行结果不应该出现在状态提示文本里
    tool_result_content = complete.calls[1][-2]["content"]
    assert tool_result_content not in second_call_last_msg["content"]


def test_central_router_loop_does_not_get_status_prompt():
    """同一个 run_agent_loop，中枢自己调用时（不经过 run_subagent）不传
    before_next_call，行为必须和加这个功能之前完全一样——状态提示只属于
    SubAgent 这一处开放式循环（D20 五处 agent 行为之一），不是全链路通用组件。
    """
    from backend.mcp_server.roles import CallerRole

    server = _server()
    session = server.open_session(CallerRole.ROUTER)
    complete = _RecordingComplete([_result(text="不需要工具")])

    from backend.agents.agent_loop import run_agent_loop

    _run(run_agent_loop([{"role": "user", "content": "阳虚质是什么"}], session, complete=complete))

    assert len(complete.calls[0]) == 1  # 只有最初那条 user 消息，没有被状态提示插队
    assert not any("已用工具调用" in str(m.get("content", "")) for m in complete.calls[0])


# ---------------------------------------------------------------------------
# 过敏原避让指令：生成阶段就提醒模型避开，不是只靠核查 pass 事后拦截
# ---------------------------------------------------------------------------


def test_tcm_subagent_system_prompt_includes_allergen_instruction_when_present():
    from backend.agents.tcm_subagent import build_tcm_system_prompt

    prompt = build_tcm_system_prompt(constitution="qi_xu", allergens=["甲壳类"])
    assert "甲壳类" in prompt
    assert "蚝油" in prompt  # 隐藏来源反查也拼进去了
    assert "整道放弃推荐" in prompt


def test_tcm_subagent_system_prompt_omits_allergen_section_when_absent():
    from backend.agents.tcm_subagent import build_tcm_system_prompt

    prompt = build_tcm_system_prompt(constitution="qi_xu", allergens=None)
    assert "过敏原" not in prompt
    assert "蚝油" not in prompt


def test_nutrition_subagent_system_prompt_includes_allergen_instruction():
    from backend.agents.nutrition_subagent import build_nutrition_system_prompt

    prompt = build_nutrition_system_prompt(allergens=["芝麻"])
    assert "芝麻" in prompt
    assert "麻酱" in prompt


def test_run_tcm_subagent_threads_allergens_into_first_llm_call():
    """不只是 build_tcm_system_prompt() 本身支持 allergens——`run_tcm_subagent()`
    真的把它传下去了，是拿真实 complete() 调用收到的第一条 system 消息来验证的，
    不是只测参数签名存在。"""
    complete = _RecordingComplete([_result(text="阳虚质应少食生冷，已避开甲壳类食材。")])
    server = _server()

    result = _run(
        run_tcm_subagent(
            "阳虚质吃什么", server, constitution="yang_xu", allergens=["甲壳类"], complete=complete
        )
    )

    assert result.final_text
    system_content = complete.calls[0][0]["content"]
    assert "甲壳类" in system_content
    assert "蚝油" in system_content


def test_run_nutrition_subagent_threads_allergens_into_first_llm_call():
    complete = _RecordingComplete([_result(text="建议多摄入优质蛋白，已避开芝麻类食材。")])
    server = _server()

    result = _run(
        run_nutrition_subagent("怎么补蛋白质", server, allergens=["芝麻"], complete=complete)
    )

    assert result.final_text
    system_content = complete.calls[0][0]["content"]
    assert "芝麻" in system_content
    assert "麻酱" in system_content


def test_subagent_prompts_include_confirmed_supplements():
    from backend.agents.nutrition_subagent import build_nutrition_system_prompt
    from backend.agents.tcm_subagent import build_tcm_system_prompt

    tcm = build_tcm_system_prompt(
        constitution="qi_xu",
        extra_profile_notes="用户在服补剂:鱼油。检索不到药食/补剂交互依据时，不得编造交互；必须声明不确定，并提示咨询医生（E8 / PRD §9 Critical，不是医嘱）。",
    )
    assert "鱼油" in tcm
    assert "不确定" in tcm

    nutrition = build_nutrition_system_prompt(
        extra_profile_notes="用户在服补剂:鱼油。检索不到药食/补剂交互依据时，不得编造交互；必须声明不确定，并提示咨询医生（E8 / PRD §9 Critical，不是医嘱）。",
        include_recipe_skill=True,
    )
    assert "鱼油" in nutrition
    assert "购物清单" in nutrition
