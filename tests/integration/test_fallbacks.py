"""
测试目标：PRD §11 fallback表逐行覆盖，含Tier1回落失败时从Tier3丢弃（D27修订二）
对应实现：全链路
覆盖要求：集成测试，mock LLM / record-replay
"""
import pytest

pytestmark = pytest.mark.skip(reason="待实现：全链路 尚未编写")


def test_placeholder():
    pass
