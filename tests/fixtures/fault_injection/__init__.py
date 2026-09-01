"""
docs/ENGINEERING.md §7.2"故障注入"、docs/BUILD_PLAN.md 阶段6"故障注入 fixture"
的实现——可复用的假 Provider + 一组预制故障，用来测 §1.2(重试与退避)/§1.3
(熔断与降级)的逻辑，不需要每个测试文件各自手搓一份假 provider/假异常。

三类故障，对应 `backend.llm.providers.base.classify_http_error`（真实
provider 和本 fixture 共用同一份判定，不是各抄一份）：
  - 有 `status_code` 属性，429 或 5xx → retryable，其余状态码 → non_retryable
  - `TimeoutError`（含本模块的 `TimeoutFault`）以及 SDK 的
    `APITimeoutError`/`APIConnectionError` → retryable
  - 其余(包括 400/401 鉴权错误、SDK 内部解析失败等)→ non_retryable

"格式错乱的 JSON"专门指第三种、性质不同的故障：不是 provider 调用抛异常，是
调用**成功**了，但模型返回的文本内容不是调用方期望的合法 JSON——`router.py`
的路由分类(`_parse_route_llm_json`/`_parse_turn_llm_json`)、
`dish_decomposition.py` 的菜品拆解、`verification.py` 的核查软判定，这几处
都要求模型输出严格 JSON 并各自处理解析失败。`MALFORMED_JSON_RESPONSES` 提供
几个真实观察到过的变体，配合 `FaultInjectingProvider` 的响应脚本使用，不用
每个测试文件重新构造一遍。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.llm.providers.base import ProviderResponse, classify_http_error

__all__ = [
    "RateLimitFault",
    "ServerErrorFault",
    "AuthFault",
    "ContentFilterFault",
    "TimeoutFault",
    "MALFORMED_JSON_RESPONSES",
    "malformed_json_response",
    "FaultInjectingProvider",
]


class RateLimitFault(Exception):
    """429 限流——ENGINEERING §1.2 表格"429 限流/5xx → 退避重试"。"""

    def __init__(self, status_code: int = 429):
        self.status_code = status_code
        super().__init__(f"rate limited (status={status_code})")


class ServerErrorFault(Exception):
    """5xx 服务端错误，同上一条重试规则。默认 503，可传别的 5xx 状态码。"""

    def __init__(self, status_code: int = 503):
        self.status_code = status_code
        super().__init__(f"server error (status={status_code})")


class AuthFault(Exception):
    """401 鉴权失败——ENGINEERING §1.2 表格"400 参数错误/401 鉴权 → 立即失败，
    重试无意义"。默认 401，可传 400 复用同一条不重试路径。"""

    def __init__(self, status_code: int = 401):
        self.status_code = status_code
        super().__init__(f"auth/param error (status={status_code})")


class ContentFilterFault(Exception):
    """内容策略拒绝——ENGINEERING §1.2 表格"内容策略拒绝 → 走 fallback，不重试"。
    这条不是靠 `classify_error()` 分类，是 `complete()` 直接检查
    `resp.stop_reason == "content_filter"`(见 backend/llm/adapter.py)，所以
    这里不是异常，是要塞进 `FaultInjectingProvider.script` 的一个
    `ProviderResponse`，不是 raise 的对象——用 `ContentFilterFault.response()`
    取。"""

    @staticmethod
    def response() -> ProviderResponse:
        return ProviderResponse(text="", stop_reason="content_filter")


class TimeoutFault(TimeoutError):
    """网络超时——ENGINEERING §1.2 表格"网络超时 → 退避重试"。

    `classify_http_error` 把 builtin `TimeoutError` 归为 retryable，所以这个
    子类不用仿造 SDK 的 `APITimeoutError`（构造那两个类需要真实 request
    对象），效果一致。
    """


# 三种真实会遇到的"模型该输出JSON却没规规矩矩输出"变体，`backend/agents/
# routing.py` 的 `_strip_json_fences()` 对前两种都做了处理(```json 代码块/
# JSON 后面跟着解释性文字，各自独立处理，见该函数文档串)；第三种(压根不是
# JSON)预期解析失败返回 None，调用方应该退回规则结果/默认分支，不该直接抛
# 异常崩掉整个请求——`tests/unit/llm/test_fault_injection.py` 用这三个变体
# 对真实解析函数做了验证，不是只测这个 fixture 模块自己。
MALFORMED_JSON_RESPONSES: dict[str, str] = {
    "wrapped_in_code_fence": '```json\n{"branch": "full_recommend", "domain_hint": null}\n```',
    "trailing_prose": '{"branch": "log_write", "domain_hint": null}\n\n希望这对你有帮助！',
    "not_json_at_all": "抱歉，我不太确定应该怎么分类这个问题。",
}


def malformed_json_response(variant: str) -> ProviderResponse:
    """包一层 `ProviderResponse`，方便直接塞进 `FaultInjectingProvider.script`。
    `variant` 必须是 `MALFORMED_JSON_RESPONSES` 的一个 key。"""
    return ProviderResponse(text=MALFORMED_JSON_RESPONSES[variant], stop_reason="stop")


@dataclass
class FaultInjectingProvider:
    """`backend/llm/providers/base.py` Provider 协议的假实现——按脚本顺序弹出
    `ProviderResponse` 或抛异常，通过 `complete()` 已有的 `provider=` 注入点
    使用(同 `backend/llm/providers/replay.py` `ReplayProvider` 的接入方式，
    业务代码不用改)。

    `classify_error` 委托给 `classify_http_error`（和两家真实 provider 同一份），
    保证注入的故障在 `adapter.py` 里触发的重试/熔断行为和真实 provider 一致，
    不会出现"测试里过了，真实 provider 分类结果不一样"这种偏差。
    """

    script: list[Any]
    calls: list[str] = field(default_factory=list)

    async def call(self, messages: list[dict], *, model: str, tools: list[dict] | None = None, **kwargs: Any) -> ProviderResponse:
        self.calls.append(model)
        if not self.script:
            raise AssertionError(f"FaultInjectingProvider 脚本已用完，但又被调用了第 {len(self.calls)} 次")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def classify_error(self, exc: Exception) -> str:
        return classify_http_error(exc)

    async def aclose(self) -> None:
        return None
