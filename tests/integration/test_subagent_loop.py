"""
测试目标：资源限额（≤15次工具调用）触发终止、状态提示确实被追加、循环防护
对应实现：backend/agents/tcm_subagent.py / nutrition_subagent.py
覆盖要求：集成测试，mock LLM / record-replay
"""
import pytest

pytestmark = pytest.mark.skip(reason="待实现：backend/agents/tcm_subagent.py / nutrition_subagent.py 尚未编写")


def test_placeholder():
    pass
