# 实施清单:按步骤对应文件与测试

**最后更新** 2026-08-26 · 关联:`ARCHITECTURE.md`(长什么样)· `planning/roadmap.md`(按什么顺序学+做)· `DECISIONS.md`(为什么这样选)· `ENGINEERING.md`(可靠性怎么做)

> **这份文档回答的问题**:roadmap.md 按"学习+制作"排了 9 个阶段,ARCHITECTURE.md §8.1/§8.2 分别给了目录树和逐文件测试映射——但没有一份文档把"这一步该写哪个文件、写完用哪个测试验证、验证过了才能进下一步"串成一条线。本文档做这件事,是喂给编码 agent 的执行清单,不是新的设计决策来源。
>
> **怎么用**(照抄 `planning/roadmap.md` 第七节的纪律):每次只把"当前步骤"这一行 + 它指向的 ARCHITECTURE.md 章节 + DECISIONS.md 对应条目喂给编码 agent,不要把整份文档倒进去;明确要求"照设计实现,不要自己发明架构";一步做完、对应测试跑过,再进下一步。
>
> **骨架已经建好**(2026-08-26):`backend/`、`api/`、`tests/`、`db/migrations/`、`frontend/` 的目录结构和占位文件已按 ARCHITECTURE.md §8.1 建出来了——每个占位文件只有 docstring(指向对应设计章节),没有实现逻辑;每个测试占位文件用 `pytest.mark.skip` 标注"待实现",跑 `pytest --collect-only tests/` 能确认全部可收集、不报错。下面清单里"文件"这一列指的就是这些占位文件,任务是把 `⏳` 变成真代码,不是新建文件。

---

## 0. 现状快照(诚实版,和 ARCHITECTURE.md §9 保持一致)

| 已完成 | 未完成(按优先级) |
|---|---|
| RAG 摄入管线(`ingest.py`,含 token 感知二次切分) | git 仓库尚未初始化(`git init` + 第一个 commit,roadmap 阶段 1 产出物) |
| `knowledge_chunks`/`recipes` 建表并在真实 Postgres 跑通 | `evals/dataset.jsonl`(E1/E2a/E2b/E3 ≥40 条)+ `smoke.jsonl` 尚未组装 |
| `user_profile`/`diet_log`/`conversation_sessions`/`messages`/`conflict_rules`/`user_dish_aliases` 建表 | `conflict_rules.jsonl` → `conflict_rules` 表的 ingest 脚本(`db/load_conflict_rules.py`)未写 |
| BGE-M3 → pgvector embedding 全量跑通(5837 条真实向量) | `dish_ingredient_map` 数据资产(目标 100-200 条,现有 44 条,见 `knowledge/food/dish-decomposition.jsonl`) |
| `conflict_rules.jsonl` 已有 40 条(18 verified/22 needs_source),**已达 roadmap 阶段 1 的 ≥30 条判据** | `backend/`/`api/`/`tests/` 下全部占位文件(本次新建的骨架) |
| BM25 baseline 脚本(`build_and_eval_bm25.py`) | LLM adapter 层(`backend/llm/adapter.py`)——roadmap 点名"必须第一个建" |

**下一步该做什么,看 §1 的"当前建议起点"标记。**

---

## 1. 分阶段清单

### 阶段 0 · 地基验证(不写代码)

| 任务 | 产出 | 状态 |
|---|---|---|
| A1 手机备忘录连记 5 天饮食 | `planning/worknotes.md` 记录 | 需人工完成,AI 不能代做 |
| A4 三个 prompt 测双边推理链稳定性 | 同上 | 需人工完成 |
| A2 徒手列 ≥10 条冲突场景 | 同上 | **已被 §1.2 的 40 条规则表间接覆盖,可视为已达标** |

这一阶段是纯人工判断,不进入"文件/测试"体系,跳过不写。

---

### 阶段 1 · 判断力 + 冲突规则表

| 任务 | 文件 | 完成判据 | 状态 |
|---|---|---|---|
| 冲突规则表扩到 ≥30 条,每条带 source | `evals/conflict_rules.jsonl` | ≥30 条且 source 非空;三类关系(一致/互补/冲突)都有覆盖 | ✅ **已完成**(40 条,18 verified/22 needs_source) |
| 建 repo + 第一个 commit | `.git/` | `git log` 能看到至少一条记录 | ⏳ **待办,建议现在就做**——后面每一步产出物都依赖有版本历史可回溯,拖得越久补 commit 越困难 |

---

### 阶段 2 · RAG + 检索层 ★

| 任务 | 文件 | 测试/验证方式 | 状态 |
|---|---|---|---|
| Step 1 最烂版 RAG | `planning/step1-naive-rag/naive_rag.py` | 手算 recall@5 | ✅ 已完成 |
| Step 2 归因 + Step 3 改动对照表 | `docs/DECISIONS.md` D2/D3/D23 | 归因表写进决策文档 | ✅ 已完成(体现在 D2/D3/D23 的论证里) |
| 结构优先切分 + token 感知二次切分 | `planning/step1-naive-rag/ingest.py` | 跑通全部真实知识源,产出 `knowledge/_processed/*.jsonl` | ✅ 已完成 |
| Embedding + 入库 | `db/embed_bge_m3.py`、`db/build_knowledge_base.py` | `SELECT count(*) FROM knowledge_chunks` 对得上 | ✅ 已完成(5837 条,tcm 4447 + nutrition 1390) |
| 溯源(citation grounding) | 待落实到 `backend/agents/tcm_subagent.py` 等的 prompt 设计里 | 随便问 5 个问题,每个答案都能指回具体 chunk | ⏳ 待阶段 4 落实——现在只有检索本身,还没有"生成时要求引用 source_id"这层 |
| 混合检索(结构化预筛 + 向量排序) | `backend/mcp_server/tools/retrieve_tcm.py`/`retrieve_nutrition.py` 的 `filters` 参数 | `WHERE 体质匹配 AND source_status='verified' ORDER BY embedding <=> $1` 能跑 | ⏳ 待阶段 4,依赖 `user_profile` 表已建(✅ 已满足前提) |

> **agentic RAG 增强(MQE/HyDE)属于这里,但明确排后**——见 ARCHITECTURE.md §2.6(新增小节),不是本阶段的阻塞项,等 baseline recall 数字稳定之后再考虑要不要加。

---

### 阶段 3 · Eval 集冻结 + 早期 baseline

| 任务 | 文件 | 完成判据 | 状态 |
|---|---|---|---|
| 写 `evals/dataset.jsonl` ≥40 条(E1 15/E2a 10/E2b 5/E3 10) | `evals/dataset.jsonl` | 测试集冻结,写下冻结日期 | ⏳ **待办**,不依赖阶段 4,现在就能做 |
| 抽出 `smoke.jsonl` 15 条 | `evals/smoke.jsonl` | 日常迭代跑它 | ⏳ 待办 |
| 跑 B0(BM25+模板)/B1(单次LLM)/B3(通用助手) | `EVALUATION.md` 第一组数字 | 三个 baseline 有数 | ⏳ 待办,B0 可复用 `build_and_eval_bm25.py` |

**⚠️ 这一阶段必须在阶段 4 写路由/SubAgent 之前完成**——D9"eval 先于实现",顺序反了会导致循环论证(roadmap 第七节第 5 条点名 AI 协作时最容易在这里翻车)。

---

### 阶段 4 · 骨架 + 五段管线 ★(核心编码阶段,顺序不能乱)

下表按 roadmap 4.2 的编号顺序排列,**编号即执行顺序**,后面每一项都依赖前面。

| # | 任务 | 实现文件(占位已建) | 设计依据 | 测试文件(占位已建) | 完成判据 |
|---|---|---|---|---|---|
| 1 | LLM adapter 层 | `backend/llm/adapter.py` | ARCHITECTURE §7,DECISIONS D19 | (无独立文件,由后续集成测试间接覆盖 mock 路径) | 改一行配置能切换模型,业务代码不动 |
| 2 | Postgres schema 补全 + conflict_rules ingest | `db/load_conflict_rules.py` | ARCHITECTURE §1.2 | — | `conflict_rules` 表数据和 jsonl 条数对得上,重跑幂等 |
| 3 | `query_diet_log` 工具 | `backend/mcp_server/tools/query_diet_log.py` | ARCHITECTURE §2.2 | (集成测试覆盖,见下方 #6) | 按时间范围/聚合维度返回,不进上下文;相对日期("昨天")能解析 |
| 4 | MCP server 骨架 + 权限分层 | `backend/mcp_server/server.py` + `tools/*.py` | ARCHITECTURE §2 | `tests/unit/mcp_server/test_tool_whitelist.py` | 越权调用在协议层被拒绝,不是应用层 if 判断 |
| 5 | 中枢 agent + Agent Loop | `backend/agents/router.py`(先只做 loop 骨架) | ARCHITECTURE §3 | — | 单工具能调通,loop 靠 tool_use 有无判断终止,不是硬编码轮数 |
| 6 | 六条分支路由 | `backend/agents/router.py` | ARCHITECTURE §5.1,DECISIONS D12/D25 | `tests/integration/test_routing.py` | 六条分支各喂 5 个 query,分类对;"记录回顾 vs 事实查询""候选评估 vs 完整推荐"边界不混淆 |
| 7 | 两个 SubAgent | `backend/agents/tcm_subagent.py`/`nutrition_subagent.py` | ARCHITECTURE §5.2 步骤3 | `tests/integration/test_subagent_loop.py` | 打日志确认 TCM 上下文里没有营养学检索内容,反之亦然;资源限额触发终止 |
| 7.5 | SubAgent 循环状态提示 | `backend/memory/status_prompt.py` | ARCHITECTURE §4.5 | `tests/unit/memory/test_status_prompt.py` | 计数正确、不携带检索原文、边界值(恰好15次) |
| 8 | 调和层 | `backend/agents/reconciliation.py` | ARCHITECTURE §5.2 步骤5,DECISIONS D14 | `tests/integration/test_reconciliation.py` | 打日志确认只收到两侧结论,没收到原始 chunk |
| 9 | 核查 pass | `backend/agents/verification.py` | ARCHITECTURE §5.2 步骤6,DECISIONS D15 | `tests/integration/test_verification.py` | 手工构造无 source_id 的建议,看它被移除而不是被补全 |
| 10 | 3 份基础 Skill + registry | `backend/skills/{reconciliation_rubric,verification_checklist,recipe_and_shopping_list}.md` + `registry.py` | ARCHITECTURE §6 | `tests/unit/skills/test_registry.py` | Skill 内容确实被拼入对应调用的 prompt,不是常驻 system prompt |
| 11 | 流式输出 + FastAPI 最小闭环 | `api/main.py`、`api/schemas.py` | ARCHITECTURE §10,DECISIONS D26 | `tests/integration/test_api_chat_sse.py` | 首字节 <4s;核查必须在第一条 token 事件前完成 |
| 12 | 前端最小闭环 | `frontend/`(尚未初始化,见 §8.1 状态) | PRD §7 | — | 浏览器能问能答;延期时优先降级为 CLI/Swagger UI |
| 13 | `docker compose up` | 新增 `docker-compose.yml`(不在当前骨架里,阶段 4 收尾时再建) | ENGINEERING §9 | — | 删本地卷重来一遍,90s 内可用 |

**⭐ 与上表并行,不能推后**(ENGINEERING 各章节):超时分层、重试+jitter、写路径幂等键、`trace_id` 贯穿、ingest 幂等+字段校验、GIN 索引、`CACHE_DISABLED` 开关——这些不是独立任务,是上表每一项实现时就要带上的属性,不是写完主逻辑再回来补。

---

### 阶段 5 · 安全层

| 任务 | 文件 | 测试文件 | 完成判据 |
|---|---|---|---|
| 输入防护 + 输出拦截 | `backend/guardrails/input_filters.py`/`output_filters.py` | `tests/unit/guardrails/test_input_filters.py` | 手工构造 5 个恶意输入全部拦下 |
| ED 防护四条 | `backend/guardrails/ed_protection.py` | `tests/unit/guardrails/test_ed_protection.py` | **要求 100% 覆盖**,数值化表述硬拦截 |
| 过敏原硬阻断 | `backend/guardrails/*`(与 §4.2 菜品拆解共用集合比对逻辑) | `tests/unit/guardrails/test_allergen_block.py` | **要求 100% 覆盖**,不重生成直接移除 |
| `THREAT_MODEL.md` | `docs/THREAT_MODEL.md`(尚未创建) | — | ≥5 个逃逸场景,PRD §15 交付物清单项 |

---

### 阶段 6 · 可观测 + 测试与 CI

| 任务 | 文件 | 完成判据 |
|---|---|---|
| Langfuse 全链路接入 | `backend/llm/adapter.py` 埋点 | 打开一条 trace 能看出走了哪条路由、每段花多少钱多少秒 |
| Record/Replay fixture | `tests/fixtures/llm_replay/`(占位已建) | 真实跑一次录制,CI 回放时离线、零成本 |
| 故障注入 fixture | `tests/fixtures/fault_injection/`(占位已建) | 429/超时/格式错乱响应能触发对应降级路径 |
| CI pipeline | 新增 `.github/workflows/`(不在当前骨架里) | `lint → 单元 → 集成(replay) → smoke eval 15条`,任一指标跌破 Launch 阈值即失败 |

---

### 阶段 7 · 分层压缩 ★(技术制高点)

| 任务 | 文件 | 测试文件 | 完成判据 |
|---|---|---|---|
| 关键事实落库前置扫描 | `backend/memory/critical_fact_scanner.py` | `tests/unit/memory/test_critical_fact_scanner.py` | **要求 100% 覆盖**,跨分支触发,误报排除 |
| 分层压缩(优先级表 + 结构化归档摘要 + 两级触发) | `backend/memory/compression.py` | `tests/unit/memory/test_compression.py` | 构造 20 轮对话,第2轮说过敏,第20轮问推荐,过敏原不消失;自动化测试,不是手工点的 |
| 菜品拆解三级查找 + 晋升逻辑 | `backend/memory/dish_decomposition.py`/`dish_alias_promotion.py` | `tests/unit/memory/test_dish_decomposition.py`/`test_dish_alias_promotion.py` | 三级优先级顺序正确;晋升阈值边界(恰好3次)正确 |
| `dish_ingredient_map` 数据扩充到 100-200 条 | `knowledge/food/dish-decomposition.jsonl` | E3 记忆子集 eval 测覆盖率 | 现有 44 条,扩充方法见该文件 README §五(用真实饮食记录,不抄菜谱网站) |
| 首次引导 + CCMQ 计分 | `backend/onboarding/flow.py`/`ccmq_scoring.py` | `tests/unit/onboarding/test_ccmq_scoring.py` | 体质夹杂正确识别;40/30分边界正确 |

---

### 阶段 8 · 重跑 eval + B2 ablation ★(简历数字诞生)

| 任务 | 完成判据 |
|---|---|
| B2(单agent+双库同上下文)实现并对比 | 有结论;若无显著优势,在 D1 下追加修订记录并回退 |
| 全量 eval 交付档 + 开发档各跑一次 | 跨模型对比表 |
| 分切片报告(特禀质/补剂交互/症状类/weight_management) | 四个切片都有数字 |
| 三档阈值对照 | 每个指标标出 Launch/Target/未达标 |

产出:`docs/EVALUATION.md`(尚未创建,PRD §15 交付物)。

---

### 阶段 9 · 收口

| 任务 | 完成判据 |
|---|---|
| 英文 README + 架构图 | 首屏 30 秒看懂 |
| 2 分钟英文录屏 | 记录→推荐→冲突调和→溯源展开 |
| 部署或录屏代替 | Vercel+Render+Neon,或一键自建说明 |

---

## 2. 当前建议起点

按依赖关系,现在(骨架已建、RAG 已跑通)最没有阻塞、价值最高的两件事,不需要等任何前置条件:

1. **`git init` + 第一个 commit**——阶段 1 的产出物,拖得越久历史越难补(roadmap 第五节"每天一个 commit"的纪律从今天才能开始生效)。
2. **`evals/dataset.jsonl` 冻结**(阶段 3)——不依赖阶段 4 的任何代码,而且 D9 要求它必须先于实现存在。现在就能写,越晚写越容易在阶段 4 写代码时被诱惑"顺手"编几个断言让测试通过。

再之后才是阶段 4 的 `backend/llm/adapter.py`(roadmap 点名"必须第一个建")。
