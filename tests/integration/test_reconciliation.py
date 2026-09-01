"""
测试目标：调和层只收两侧结论,不收原始检索内容(D14)；Skill 内容确实被拼入
prompt 而非常驻中枢 system prompt；调和层调用跟随 MODEL_TIER,不强制
force_prod_tier(D19 已于 2026-08-27 撤销这条例外,见决策修订记录)。
对应实现：backend/agents/reconciliation.py
覆盖要求：集成测试，注入假 complete()（同 tests/unit/agents/test_agent_loop.py 的模式），
不打真实网络/LLM。
"""
from __future__ import annotations

import asyncio
import logging

from backend.agents._subagent_common import SubAgentResult
from backend.agents.reconciliation import reconcile, reconcile_subagent_results
from backend.llm.adapter import LLMResult, ModelTier

# 模拟"原始检索 chunk 原文"的标记——真实 SubAgent 的 messages 里，工具结果会
# 长这样；如果这段字符串出现在喂给调和层 LLM 调用的 prompt 里，就说明 D14 被
# 破坏了(原始检索内容泄漏进了调和层)。
_RAW_CHUNK_MARKER = "RAW_CHUNK_TEXT_MARKER：阳虚质忌生冷这段来自 knowledge_chunks 的原文……"


def _run(coro):
    return asyncio.run(coro)


class _RecordingComplete:
    def __init__(self, text: str = "冲突摘要：……\n立场：……"):
        self._text = text
        self.calls: list[dict] = []

    async def __call__(self, messages, *, force_prod_tier=False, **kwargs):
        self.calls.append({"messages": list(messages), "force_prod_tier": force_prod_tier, **kwargs})
        return LLMResult(text=self._text, model="m", tier=ModelTier.PROD, provider="fake")


def _subagent_result(domain: str, final_text: str) -> SubAgentResult:
    return SubAgentResult(
        domain=domain,
        final_text=final_text,
        tool_call_count=1,
        iterations=2,
        terminated_reason="no_tool_use",
        messages=[
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "name": f"retrieve_{domain}", "arguments": {}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": f"retrieve_{domain}",
                "content": _RAW_CHUNK_MARKER,
                "ok": True,
            },
        ],
        tools_called=[f"retrieve_{domain}"],
    )


# ---------------------------------------------------------------------------
# D14：只收两侧结论，不收原始检索内容
# ---------------------------------------------------------------------------

def test_reconcile_never_forwards_raw_chunk_content_via_subagent_results(caplog):
    tcm_result = _subagent_result("tcm", "阳虚质应少食生冷 [source: tcm_000123]")
    nutrition_result = _subagent_result("nutrition", "缺铁建议多摄入红肉 [source: nutri_000456]")
    complete = _RecordingComplete()

    with caplog.at_level(logging.INFO, logger="diet_expert.agents.reconciliation"):
        _run(reconcile_subagent_results(tcm_result, nutrition_result, complete=complete))

    assert len(complete.calls) == 1
    sent_messages = complete.calls[0]["messages"]
    full_prompt_text = "\n".join(m["content"] for m in sent_messages)
    assert _RAW_CHUNK_MARKER not in full_prompt_text
    assert "阳虚质应少食生冷" in full_prompt_text
    assert "缺铁建议多摄入红肉" in full_prompt_text

    # 打日志确认：知道两侧各有多少条 messages 被丢弃，而不是悄悄忽略
    discard_logs = [r.message for r in caplog.records if "discarded" in r.message]
    assert discard_logs
    assert "messages=3" in discard_logs[0]  # 每侧 mock 数据固定 3 条 messages


def test_reconcile_signature_cannot_accept_subagent_result_directly():
    """D14 的边界靠函数签名强制：reconcile() 只接受纯文本，类型层面就传不进
    SubAgentResult(想传整个对象会在拼 prompt 时得到一个 repr 字符串，而不是
    真的泄漏 .messages 内容)。"""
    tcm_result = _subagent_result("tcm", "结论A")
    complete = _RecordingComplete()

    _run(reconcile(str(tcm_result.final_text), "结论B", complete=complete))
    sent_messages = complete.calls[0]["messages"]
    full_prompt_text = "\n".join(m["content"] for m in sent_messages)
    assert _RAW_CHUNK_MARKER not in full_prompt_text


# ---------------------------------------------------------------------------
# Skill 内容确实被拼入 prompt（而非常驻中枢 system prompt）
# ---------------------------------------------------------------------------

def test_reconciliation_rubric_skill_is_injected_into_system_prompt():
    complete = _RecordingComplete()
    _run(reconcile("TCM结论", "营养结论", complete=complete))

    system_message = complete.calls[0]["messages"][0]
    assert system_message["role"] == "system"
    assert "调和层仲裁准则" in system_message["content"]
    assert "harm reduction" in system_message["content"].lower()
    assert "过敏原" in system_message["content"]


def test_matched_conflict_rules_are_rendered_into_user_message():
    complete = _RecordingComplete()
    matched_rules = [
        {
            "rule_id": "W01",
            "topic": "抗凝药与维生素K",
            "relation": "conditional_conflict",
            "resolution": "服用抗凝药期间需保持维生素K摄入稳定，不建议骤增骤减",
        }
    ]
    _run(reconcile("TCM结论", "营养结论", matched_rules=matched_rules, complete=complete))

    user_message = complete.calls[0]["messages"][1]
    assert "抗凝药与维生素K" in user_message["content"]
    assert "conditional_conflict" in user_message["content"]
    assert "服用抗凝药期间需保持维生素K摄入稳定" in user_message["content"]


def test_matched_rule_id_never_appears_in_the_prompt_to_avoid_citation_collision():
    """真实踩到的坑，而且踩了两次：第一次规则编号写成 "- [W01] ..."，方括号
    包一个短 id 和 `[source: chunk_id]` 引用格式长得几乎一样，模型把规则编号
    当成可引用的 id、套进 `[source: W01]`。改成不带方括号的"规则W01："之后，
    模型换了个方式复现同一个错误——直接把"规则W01"这串文本原样当成 id 塞进
    `[source: 规则W01]`。两次教训是同一件事：只要 prompt 里出现任何"看起来
    像可以指着说的短码"，模型就有概率把它套进引用模板，光靠 rubric 里加一句
    说明性指令堵不住。真正的修法是 rule_id 完全不出现在喂给模型的文本里——
    这个测试锁住"规则的原始 id 不该以任何形式出现在 prompt 里"这条约束，
    不只是"不该被方括号包着"这一种具体形态。"""
    complete = _RecordingComplete()
    matched_rules = [
        {"rule_id": "K01", "topic": "主食搭配", "relation": "conflict", "resolution": "不建议断主食"}
    ]
    _run(reconcile("TCM结论", "营养结论", matched_rules=matched_rules, complete=complete))

    user_message = complete.calls[0]["messages"][1]["content"]
    assert "K01" not in user_message
    assert "主食搭配" in user_message
    assert "不建议断主食" in user_message


def test_no_matched_rules_is_stated_explicitly_not_silently_empty():
    complete = _RecordingComplete()
    _run(reconcile("TCM结论", "营养结论", complete=complete))
    user_message = complete.calls[0]["messages"][1]
    assert "无命中" in user_message["content"]


# ---------------------------------------------------------------------------
# D19 修订(2026-08-27)：调和层不再强制 force_prod_tier，跟随 MODEL_TIER
# ---------------------------------------------------------------------------

def test_reconcile_does_not_force_prod_tier():
    """原 D19 要求调和层永远 force_prod_tier=True——本地开发环境因此需要
    单独配一套 prod 档凭据，不然调和层直接鉴权失败；已撤销这条例外(见
    DECISIONS.md D19 决策修订记录)，调和层现在和其余调用一样跟随
    MODEL_TIER，不显式传 force_prod_tier。"""
    complete = _RecordingComplete()
    _run(reconcile("TCM结论", "营养结论", complete=complete))
    assert complete.calls[0]["force_prod_tier"] is False


def test_reconcile_result_reports_actual_tier_used():
    result = _run(reconcile("TCM结论", "营养结论", complete=_RecordingComplete()))
    assert result.tier == ModelTier.PROD
    assert result.text


# ---------------------------------------------------------------------------
# avoid_note：过敏原重试场景专用小节（api/main.py 检测到命中后带反馈重新调用）
# ---------------------------------------------------------------------------


def test_avoid_note_is_appended_as_a_dedicated_section():
    complete = _RecordingComplete()
    _run(
        reconcile(
            "TCM结论", "营养结论",
            avoid_note="上一版提到了蚝油，用户对甲壳类过敏，请重新生成避开它。",
            complete=complete,
        )
    )
    user_message = complete.calls[0]["messages"][1]["content"]
    assert "重新生成要求" in user_message
    assert "蚝油" in user_message


def test_no_avoid_note_omits_the_section_entirely():
    complete = _RecordingComplete()
    _run(reconcile("TCM结论", "营养结论", complete=complete))
    user_message = complete.calls[0]["messages"][1]["content"]
    assert "重新生成要求" not in user_message


def test_reconcile_subagent_results_threads_avoid_note_through():
    complete = _RecordingComplete()
    tcm_result = _subagent_result("tcm", "结论A")
    nutrition_result = _subagent_result("nutrition", "结论B")
    _run(
        reconcile_subagent_results(
            tcm_result, nutrition_result,
            avoid_note="请避开花生。",
            complete=complete,
        )
    )
    user_message = complete.calls[0]["messages"][1]["content"]
    assert "请避开花生" in user_message
