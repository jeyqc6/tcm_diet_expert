# 后端工程化要求

**最后更新** 2026-08-31 · 关联:`PRD.md` §12 非功能性需求 · `DECISIONS.md` · `THREAT_MODEL.md` · `ASYNC_DESIGN.md`

> PRD 说的是"失败时降级到什么",本文说的是"**怎么实现那个降级**"。
> 这些不是加分项——PRD §11 的 fallback 表如果没有下面这些机制,就只是一张纸。

---

## 1. LLM 调用的可靠性层

所有模型调用经统一 adapter(`backend/llm/adapter.py`,`DECISIONS.md` D19)。可靠性逻辑全部收敛在这一层,业务代码不重复实现。

### 1.1 超时分层

**必须分三层**,只设一个全局超时会导致"整条链路一起死"。

| 层级 | 超时 | 超时后 | 实现 |
|---|---|---|---|
| 单次 LLM 调用 | 20s | 重试(见 1.2) | ✅ adapter / provider client timeout |
| 单个 SubAgent(含内部多轮) | 45s | 放弃该侧,走单边输出 | ✅ `run_subagent` + `asyncio.wait_for` |
| 整条请求链路 | 90s | 强制收口,输出当前最佳结论 | ✅ `_stream_chat` + `aiter_with_timeout`；已开始吐 token 的继续保留，未完成的取消后吐 `chain_timeout` |

与 M10(首字节 < 4s)、M11(端到端 p95 < 35s)对齐:**超时阈值必须大于指标目标,否则正常请求会被误杀**。

### 1.2 重试与退避

```
指数退避 + jitter:1s → 2s → 4s,最多 3 次
```

| 错误 | 重试? |
|---|---|
| 429 限流 / 5xx | ✅ 退避重试 |
| 网络超时 | ✅ |
| 400 参数错误 / 401 鉴权 | ❌ 立即失败,重试无意义 |
| 内容策略拒绝 | ❌ 走 fallback,不重试 |

**jitter 不可省。** 两个 SubAgent 并行时若同时被限流,固定退避会导致它们永远同步重试、同步再被限流。

⚠️ **写操作不重试。** 记录落库(M6)如果重试,会产生重复记录。写路径用**幂等键**:`(user_id, logged_at, raw_input_hash)` 唯一约束,重复插入直接忽略。

### 1.3 熔断与降级

对每个外部依赖维护熔断器:

| 依赖 | 熔断阈值 | 打开后行为 | 对应 PRD |
|---|---|---|---|
| 主力模型 | 连续 5 次失败 | 切备用模型档 | §13.2 |
| Open-Meteo | 连续 3 次失败 | 回退节气表,静默 | §11 |
| 向量检索 | 连续 3 次失败 | 回退静态兜底表 | §11 |

熔断打开后**半开探测**:30s 后放一个请求试探,成功则关闭。

**熔断状态必须进 trace**,否则你会看到"分数突然掉了"却不知道是因为整段时间在跑降级路径。

---

## 2. 并发与取消

并发发生在两处:**双 SubAgent 并行**(`dispatch.py` `asyncio.gather`)与**单轮多 tool 并行**(`agent_loop.py` `gather + to_thread`)。完整分层、sync/async 边界、已知不完美点与改进工时见 **`ASYNC_DESIGN.md`**。本节只记可靠性坑与已踩过的 bug。

下面四个坑里,前三个是编排层老问题;坑四是 2026-08-31 工具并发上线时真实撞过的 event-loop/缓存问题。

```python
# 关键:return_exceptions=True，一侧失败不能拖垮另一侧
results = await asyncio.gather(
    run_subagent("tcm", ctx),
    run_subagent("nutrition", ctx),
    return_exceptions=True,
)
```

**坑一:部分失败要能单边输出。** 对应 PRD §11"单个 SubAgent 失败 → 单边输出并标注"。用 `return_exceptions=True` 而不是让异常冒泡。✅ `backend/agents/dispatch.py` `_stream_dual_dispatch`。

**坑二:取消要能传播。** 整链超时后,还在跑的 SubAgent 必须被真正取消,否则它会继续烧 token。用 `asyncio.wait_for` 包裹并确保 `CancelledError` 一路传到 HTTP 客户端。✅ 已实现(2026-08-29):

- 45s: `run_subagent()` 用 `asyncio.wait_for` 包住 `run_agent_loop()`，超时取消该侧并抛 `SubAgentTimeoutError`（双派发走单边；单领域吐 `subagent_timeout`）
- 90s: `_stream_chat()` 用 `aiter_with_timeout` 包住整条生成器；超时 `aclose` 生成器，从而取消还在 `gather` 里的一侧，再给 HTTP 客户端吐 `chain_timeout` + `done`
- `gather(..., return_exceptions=True)` 会把 `CancelledError` 收成结果，`reraise_if_cancelled()` 再抛出去
- 阈值可配:`SUBAGENT_TIMEOUT_S` / `CHAIN_TIMEOUT_S`

**坑三:并发不等于免费。** 两侧并行只省墙钟时间,**token 成本照算**。✅ 已实现:每次 `complete()` 经 `_metered_complete` 累加进请求级 `RequestCost`；根 span `chat.output` 带 `tokens` / `cost_est` / `llm_calls`（加总，不是 `max(tcm, nutrition)`）；双派发另打 `dual_dispatch` 日志，`cost_is_sum_not_wall=true`。

**坑四:惰性单例/缓存一旦跟 event loop 的生命周期绑定，顺序执行时不会暴露，改成真并发就会稳定复现。** 2026-08-31 真实撞过:把 `backend/agents/agent_loop.py` 一轮里的多个工具调用从顺序 `for` 循环改成 `asyncio.gather` + `asyncio.to_thread` 并发执行后,检索类请求整体挂死、撞 45s SubAgent 超时——不报错,单纯卡住,日志里什么异常都看不到。

根因:`backend/mcp_server/tools/_retrieval_common.py` 的 MQE(查询改写)通过 `_run_coroutine_sync()` 在同步代码里调用异步的 `complete()`，**每次调用都现造一个用完即销毁的 event loop**；而 `backend/llm/adapter.py` 的 provider 客户端(如 `AsyncAnthropic`，内部持有 httpx 连接池)当时是**按进程生命周期全局缓存**的。多个并发 worker 各自起新 loop 时，会把同一个缓存客户端在互不相同的 event loop 之间反复复用——这是已知会挂死的 asyncio 反模式：客户端内部对象(连接池/内部同步原语)绑定在创建它那一刻正在跑的 loop 上，那个 loop 被销毁之后不会自动"解绑"，只是变成一个停摆但仍被引用着的对象；换一个新 loop 复用它，请求发出去之后没人负责通知"收到响应了"，于是永远卡住。

✅ 已修复(两处，都是同一类"惰性单例没加锁/没考虑跨 loop"问题的不同表现):
- `backend/llm/adapter.py` `_get_provider()`：缓存键从"仅 provider 名字"改成"`(provider 名字, 当前 event loop)`，用 `weakref.WeakKeyDictionary` 以 loop 对象本身(不是 `id()`)做 key——`id()` 不安全，CPython 里被回收对象的 id 之后可能被全新对象复用，会把"早就销毁的旧 loop 的客户端"错误地当成"当前新 loop 的客户端"发回去，等于把 bug 原样绕了个圈子重新引入。
- `backend/mcp_server/tools/_retrieval_common.py` `_get_embedder()`(BGE-M3 惰性单例)：原来没加锁，两个线程同时看到"还没初始化"会各自跑一遍模型加载，补了 `threading.Lock` 双重检查锁定。

**教训，不只是这一处**:任何"建一次、全局复用"的缓存/单例，只要背后的资源(HTTP 客户端、模型对象、任何持有 `asyncio.Lock`/`Future`/socket 的东西)生命周期实际上跟"当前 event loop"或"当前线程"绑定，在代码从顺序执行改成并发执行的那一刻都可能暴露——**顺序执行时这类 bug 会长期潜伏、单测也测不出来**(单测通常不会真的触发跨 loop 复用这个条件)，必须用真实请求跑几轮才能验证；这次就是先看单测全绿、以为改完了，接入真实请求后才发现挂死，回退过一次，定位根因后才重新上线。完整推导过程(含可复现的隔离验证脚本结果)见 `backend/llm/adapter.py`/`backend/agents/agent_loop.py` 对应函数的注释。

---

## 3. 缓存分层

| 缓存 | 键 | TTL | 收益 |
|---|---|---|---|
| **Prompt caching** | system prompt + 工具定义 + L0 画像(约 3k tokens) | 供应商侧 | 成本降约 90%(这部分) |
| 天气数据 | `(城市, 日期)` | **3h** | 直接来自 PRD §12.2 的新鲜度要求 |
| 向量检索结果 | `hash(query + 预筛条件)` | 24h | 开发期反复跑 eval 时省大钱 |
| 冲突规则表 | 全量 40 条 | 进程内,启动加载 | 表太小,不值得每次查库 |

⚠️ **eval 跑分时必须能关掉检索缓存**,否则改了 chunking 却读到旧缓存,分数不变会让你以为改动无效。加一个 `CACHE_DISABLED=1` 开关。

---

## 4. 数据层

### 4.1 迁移与 ingest

- **迁移用 Alembic**,不要手写 SQL 改表。表结构会随 eval 迭代改好几次
- **ingest 幂等**:JSONL → DB 单向(`DECISIONS.md` D18 的延伸)。用 `INSERT ... ON CONFLICT (id) DO UPDATE`,重跑不产生重复
- **ingest 有校验**:字段完整性、`source_status` 合法值、`allergens` 数组元素在 9 类之内。**校验失败就中止,不要灌半张表进去**

### 4.2 索引

数组字段全部建 GIN,否则 `= ANY()` 会全表扫:

```sql
CREATE INDEX ON condiment_allergens USING GIN (allergens);
CREATE INDEX ON condiment_allergens USING GIN (aliases);
CREATE INDEX ON conflict_rules USING GIN (applicable_constitutions);
CREATE INDEX ON diet_log (user_id, logged_at DESC);   -- query_diet_log 的主查询路径
```

向量索引在**数据灌完之后**再建(HNSW 建索引比增量插入快得多)。

### 4.3 连接池

单用户场景池子设小(5-10)。但 **ingest 脚本和 API 用不同的池**,否则一次全量 ingest 会把 API 的连接吃光。

---

## 5. 配置与密钥

| 项 | 做法 |
|---|---|
| 密钥 | 只从环境变量读,`.env` 进 `.gitignore`,repo 里放 `.env.example` |
| 模型档切换 | `MODEL_TIER=dev|prod` 一个变量切换双档(D19),**不改代码** |
| 阈值 | 超时、重试次数、缓存 TTL 全部可配,默认值写在代码里 |
| 提交前检查 | pre-commit hook 扫 API key 模式,防止误提交 |

⚠️ **健康数据不进日志明文**(PRD §13.4)。脱敏在 adapter 层做,不要指望每个调用点都记得。

---

## 6. 可观测性

**状态（2026-08-26）**：Langfuse 全链路已接入，`backend/observability/`。没配密钥时 spans 是 no-op，`trace_id` 仍然生成并写进 HTTP 头 / SSE `done` / 结构化日志。

**不会用 Langfuse、想对照代码看一条请求**：`docs/LANGFUSE.md`（概念 + 注册配密钥 + 网页上怎么读瀑布图 + 本仓库埋点怎么接的）。

### 6.1 trace_id 贯穿

每个请求生成一个 `trace_id`,**贯穿到每一层**:HTTP 响应头 → 结构化日志 → Langfuse span → 数据库写入记录。

出问题时能用一个 id 把整条链路捞出来。没有这个,五段管线出错只能靠猜。

### 6.2 结构化日志

用 JSON 行,不用 f-string 拼接。每条至少带:`trace_id` / `stage`(router|tcm|nutrition|reconcile|verify)/ `latency_ms` / `tokens` / `cost_est` / `fallback_triggered`。

`backend/logging_config.py` `configure_logging()` 在 FastAPI lifespan 里显式调用（不要在 import 时 `basicConfig`）。`LOG_FORMAT=json|text`，`LOG_LEVEL` 可配。`trace_id` 从 `tracing.py` 的 ContextVar 经 logging.Filter 自动挂上，业务代码不用手传。健康字段走同一份 `redact.py`，不要在每个调用点自己脱敏。

### 6.3 必须埋的点

| 埋点 | 为什么 |
|---|---|
| 路由决策及其分支 | M13 的数据来源 |
| 每段的耗时与 token | M10/M11/M12 |
| 熔断/降级触发 | 否则分数波动无法解释 |
| 核查 pass 的每次拦截及原因 | M14 + 安全审计 |
| 冲突规则命中的 rule_id | 用来分析规则表覆盖率 |
| `conflict_gaps` 写入 | 自我改进闭环的输入 |

实现落点：

| 埋点 | 代码 |
|---|---|
| 根 trace + HTTP `X-Trace-Id` | `api/main.py` `chat()` / `_stream_chat()` |
| 路由决策 | `backend/agents/routing.py` `classify_route_async` |
| LLM generation（tokens / cost_est / 熔断降级） | `backend/llm/adapter.py` `complete()` |
| 请求级合计（两侧加总，不是墙钟 max） | `api/main.py` `_stream_chat` → `chat.output.tokens` / `cost_est` / `llm_calls` |
| 双派发墙钟 vs 成本 | `backend/agents/dispatch.py` `dual_dispatch` `stage_log`（`cost_is_sum_not_wall`）
| SubAgent | `backend/agents/_subagent_common.py` |
| 调和层 + 命中的 `rule_id` | `backend/agents/reconciliation.py` |
| 核查拦截原因 | `backend/agents/verification.py` |
| MCP 工具 | `backend/mcp_server/server.py` `call_tool` |

`conflict_gaps` 由 `backend/agents/conflict_gaps.py` 在双侧调和且 `matched_rules` 为空时追加一行；写失败不阻断请求。`tests/integration/test_fallbacks.py` 覆盖单侧失败 / 双侧失败（guardrail，无 naive RAG）/ 链超时 / 空检索（核查拒绝，无静态兜底表）。

---

## 7. ⭐ 测试策略:LLM 应用怎么测

**这一节是本文最有价值的部分。** 大部分人做 AI 项目只有 eval,没有测试——两者不是一回事。

### 7.1 三层测试金字塔

| 层 | 测什么 | 特点 | 覆盖率要求 |
|---|---|---|---|
| **单元测试** | 确定性逻辑:过敏原查询、菜品拆解、GIN 数组匹配、ED 正则拦截、幂等键 | 快、无 LLM、可 100% 确定 | **过敏原与 ED 相关路径要求 100%** |
| **集成测试** | 管线编排:超时、重试、熔断、部分失败、取消传播 | 用 **mock LLM**,不烧 token | 每条 fallback 路径至少一条 |
| **Eval** | 输出质量:M1-M14 | 慢、贵、有波动 | 见 `EVALUATION.md` |

**关键认知**:PRD §11 那张 fallback 表的每一行,都应该有一条集成测试。**没测过的降级路径 = 不存在的降级路径**——它只会在演示时第一次被触发。

### 7.2 怎么在不烧 token 的情况下测管线

**Record / Replay fixtures**(✅ 已实现,2026-08-27:`backend/llm/providers/replay.py` `ReplayProvider`/`replay_provider_for`,`tests/fixtures/llm_replay/`):

1. 录制模式:真实跑一次,把每次 LLM 调用的 `(请求指纹, 响应)` 存成 fixture 文件
2. 回放模式:CI 里 adapter 层拦截调用,按指纹返回录制的响应
3. 指纹对不上就报错——**这本身就是"prompt 被意外改动"的检测器**

这样集成测试完全离线、零成本、可重复。`ReplayProvider` 实现的是 `backend/llm/providers/base.py` 的 Provider 协议,走 `complete()` 已有的 `provider=` 测试注入点(§7.1 表格里"注入假实现"的既有模式),业务代码不用改一行;`LLM_REPLAY_MODE` 环境变量在 record/replay 两种模式之间切换,默认 replay,不需要任何 LLM API key。用法与示例见 `tests/fixtures/llm_replay/README.md`、`tests/integration/test_llm_replay.py`。

**故障注入**(✅ 已实现,2026-08-28:`tests/fixtures/fault_injection/` `FaultInjectingProvider` + 预制故障类):另有一组 fixture 专门返回 429、超时、格式错乱的 JSON,用来测 1.2 和 1.3 的逻辑。`classify_error` 走 `backend/llm/providers/base.py` 的 `classify_http_error`（真实 provider 和 fixture 共用同一份），保证测出来的重试/熔断行为和真实 provider 一致;用法见 `tests/fixtures/fault_injection/README.md`、`tests/unit/llm/test_fault_injection.py`。

### 7.3 确定性优先原则

**能不经过模型的判断,就不要经过模型。** 过敏原命中、药食同源白名单、ED 数值拦截——这三样全部是确定性代码,因此可以被 100% 单测覆盖。

这不只是工程洁癖:**Critical 档的安全边界必须可穷举验证**,而 LLM 的输出无法穷举。

---

## 8. CI

```
lint (ruff) → 单元测试 → 集成测试(replay,离线) → smoke eval(15 条)
```

✅ 已实现(2026-08-28)：`.github/workflows/ci.yml`(三个 job 串行，`lint`→`test`→`smoke-eval`)+ `ruff.toml`(只选 `E`/`F`，忽略 `E501`，见该文件注释)+ `evals/run_baselines.py` 新增 `--check-launch-threshold`(M1 recall@5 跌破 Launch 阈值 exit(1))。

| Gate | 规则 |
|---|---|
| 单元测试 | 全绿才能合并 |
| 集成测试 | 全绿 |
| **Smoke eval** | 任一指标跌破 **Launch 阈值** → CI 失败 |
| 成本 | 单次 smoke 跑分成本记录在 CI 输出里,超预算告警 |

**smoke eval 进 CI 是本项目区别于普通 demo 的一个具体标志。** 它把"eval 作为回归测试"(PRD §14.3)从一句话变成一个会拦住你的门。

⚠️ smoke eval 用**开发档模型**跑(D19),否则每次 push 都烧钱。

⚠️ **诚实限制**：CI smoke-eval 用的是 `evals/fixtures/bm25_smoke_chunks.jsonl` + `--check-ci-floor 0.6`，**每次都跑**。这不是真实 5837 条语料上的 Launch 70%（当前全量 B0=53.3%）。`--check-launch-threshold` 留给本机对着 `knowledge/_processed` 对照。M3/M5 仍无自动化 eval。

---

## 9. 容器化与部署

| 项 | 做法 |
|---|---|
| 镜像 | 多阶段构建,运行镜像不含构建工具链 |
| 健康检查 | `/healthz` 检查 DB 连接 + 向量索引就绪;`docker compose` 用 `depends_on: condition: service_healthy` |
| 启动顺序 | Postgres → 迁移 → ingest(可跳过)→ API → 前端 |
| 冷启动 | 90s 内可用(PRD §12.1)。**ingest 不放在启动路径上**,否则每次起容器都要跑 10 分钟 |
| 数据卷 | 向量库数据持久化,重启不丢 |
| 一键起 | `docker compose up` 后 `/healthz` 返回 200 即视为成功 |

**D14 前必须验证一次**:在干净机器上(或删掉本地所有卷)重来一遍。**"在我机器上能跑"是交付物的头号杀手。**

---

## 10. 优先级(时间不够时)

| 优先级 | 项 | 能砍吗 |
|---|---|---|
| 1 | adapter 层 + 超时/重试 | ❌ 没有它 fallback 表是假的 |
| 2 | trace_id 贯穿 + 结构化日志 | ❌ 没有它无法归因 |
| 3 | 确定性逻辑的单元测试(过敏原/ED) | ❌ Critical 档安全边界 |
| 4 | ingest 幂等 + 校验 | ❌ 数据脏了后面全脏 |
| 5 | smoke eval 进 CI | ❌ 这是差异化标志 |
| 6 | Record/Replay 集成测试 | 🟡 可降级为手工跑一遍 fallback 路径 |
| 7 | 熔断器 | 🟡 可降级为简单重试 + 降级 |
| 8 | 缓存分层 | 🟡 只保留天气缓存 |
| 9 | 健康检查与启动编排 | 🟡 可降级为 README 说明启动顺序 |
| 10 | 多阶段构建 | ✅ 单阶段也能跑 |
