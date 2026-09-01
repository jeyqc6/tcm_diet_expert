存放 429/超时/格式错乱响应，用于测 docs/ENGINEERING.md §1.2(重试与退避)/§1.3
(熔断与降级)的逻辑。

实现：`__init__.py`(`FaultInjectingProvider` + 一组预制故障类/样本)，是可导入的
Python 包（不是纯数据目录），用法见 `tests/unit/llm/test_fault_injection.py`。

## 怎么用

```python
from backend.llm.adapter import complete
from tests.fixtures.fault_injection import FaultInjectingProvider, RateLimitFault

provider = FaultInjectingProvider(script=[RateLimitFault(), ProviderResponse(text="ok", stop_reason="stop")])
result = await complete(messages, provider=provider, sleep=fake_sleep, circuit=circuit)
```

提供的故障：

| 名字 | 模拟什么 | `classify_error` 结果 |
|---|---|---|
| `RateLimitFault` | 429 限流 | retryable |
| `ServerErrorFault` | 5xx 服务端错误 | retryable |
| `TimeoutFault` | 网络超时 | retryable |
| `AuthFault` | 400/401 | non_retryable |
| `ContentFilterFault.response()` | 内容策略拒绝 | 不经过 classify_error，`complete()` 直接检查 `stop_reason` |
| `MALFORMED_JSON_RESPONSES` / `malformed_json_response(variant)` | 模型正常返回，但内容不是合法 JSON(代码块包裹/JSON后跟解释文字/完全不是JSON) | 不是异常，是一个正常的 `ProviderResponse` |

`FaultInjectingProvider.classify_error` 调用 `backend.llm.providers.base.classify_http_error`（和两家真实 provider 同一份），不是简化版——保证测出来的重试/熔断行为和真实 provider 一致。
