"""
测试目标：SSE事件顺序（核查必须在第一条token事件前完成）、trace_id贯穿
对应实现：api/main.py
覆盖要求：集成测试，mock LLM / record-replay
"""
import pytest

pytestmark = pytest.mark.skip(reason="待实现：api/main.py 尚未编写")


def test_placeholder():
    pass
