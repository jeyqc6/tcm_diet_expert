存放 docs/ENGINEERING.md §7.2 的录制响应，按 `{caller}__{fingerprint}.json` 命名。
真实跑一次录制，CI 回放时零成本、完全离线。

实现：`backend/llm/providers/replay.py`(`ReplayProvider`/`replay_provider_for`)。

## 怎么用

测试代码只需要一行注入，不改业务代码：

```python
from backend.llm.adapter import complete
from backend.llm.providers.replay import replay_provider_for

provider = replay_provider_for("caller_label")  # caller_label 只影响文件名，随便起一个能看出是哪个调用点的名字
result = await complete(messages, provider=provider)
```

- **默认(不设置任何环境变量)= replay 模式**：不打网络，按 `(model, messages, tools)` 算出的指纹在这个目录里查文件；查不到就报错——这是设计好的行为(指纹对不上=prompt/messages/model 变了，需要重新录制)，不是 bug。CI 跑这个模式，不需要任何 LLM API key。
- **`LLM_REPLAY_MODE=record`(+ 真实凭据)**：真的打一次网络请求，把 `(请求, 响应)` 存成这个目录下的 fixture 文件。跑完把新文件提交进仓库。

重新录制某个 fixture 时，本地设：

```bash
LLM_REPLAY_MODE=record ANTHROPIC_API_KEY=sk-... python3 -m pytest tests/integration/test_llm_replay.py -q
```

参考示例：`tests/integration/test_llm_replay.py` + `demo_router_call__*.json`(真实调用 Anthropic Haiku 录制)。
