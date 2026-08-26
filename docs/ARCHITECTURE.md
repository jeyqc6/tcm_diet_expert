# 实现设计文档

**版本** v0.5 · **状态** Draft(骨架未开工前的设计冻结)· **最后更新** 2026-08-26

关联:`PRD.md`(what/why)· `DECISIONS.md`(为什么选这个不选那个)· `ENGINEERING.md`(可靠性/测试怎么做)· `RAG_PIPELINE_DESIGN.md`(RAG 四环节细节)· `planning/roadmap.md`(按什么顺序学+做)

> **这份文档回答的问题,和其他文档不一样**:PRD 说"要做什么、为什么要做";DECISIONS 说"为什么选 A 不选 B";ENGINEERING 说"降级/重试/测试怎么实现";roadmap 说"按什么顺序学、每天干什么"。**本文档说的是"具体长什么样"**——工具的函数签名、表的字段、请求从进来到出去经过哪几步、MCP server 暴露哪些方法。写代码时应该直接对照这份文档,而不是从 PRD 反推。
>
> 本文档最后一节(§9)是**诚实的完成状态表**——目前只有 RAG 摄入管线和 recipes/knowledge_chunks 两张表的 SQL 是跑通过的真代码,MCP server、中枢 agent、两个 SubAgent、调和层、核查 pass、记忆写入、前端全部还没有一行实现代码。这不是疏漏,是如实记录:这份文档本身就是"骨架期开工前"的设计冻结,骨架期(roadmap 阶段 4)开始后每完成一项就回来勾掉。

---

## 0. 总览图

```
用户输入
   │
   ▼
① 输入防护(截断 / 指令注入过滤 / 疾病用药检测)              [PRD §10 输入防护]
   │
   ▼
② 关键事实扫描(跨路由分支,常驻)                             [新增,见 §5.4]
   │  · 命中过敏原/禁忌/补剂关键词 → write_memory(critical) → 人在环确认 → 继续
   │
   ▼
③ 中枢 agent:分级路由(六条分支)                              [D12/D25 · M13]
   │
   ├── 记录(写入) ──────────────────────────────┐
   ├── 记录回顾(查自己的 diet_log)───────────────┤
   ├── 事实查询(查静态知识库) ────────────────────┤
   ├── 候选评估 ─┬─ TCM SubAgent(任务框架:评估候选)│      [M3]
   │            └─ Nutrition SubAgent            │      [M4]
   ├── 单领域 ── 单 SubAgent ─────────────────────┤
   └── 完整推荐 ─┬─ TCM SubAgent                  │      [M3]
                 └─ Nutrition SubAgent            │      [M4]
                          │                       │
                  调和层(独立调用,双派发/多候选取舍时) │  [D14 · M5]
                          │                       │
                  核查 pass ◄─────────────────────┘      [D15 · M14]
                          │
              调和建议 + 双理由 + 菜谱 + 购物清单(流式输出)
              (候选评估分支:通过/不通过 + 理由,或候选间取舍)
                          │
                          ▼
              会话落库(供压缩)+ 隐式反馈埋点

工具层(经本地 MCP server 暴露,D7 修订):
  retrieve_tcm · retrieve_nutrition · query_weather · query_diet_log
  · write_memory · query_recipes_by_ingredients

存储层(单一 Postgres 实例,D4/D18/D23/D24):
  knowledge_chunks(✅已建表) · recipes(✅已建表)
  · user_profile(⏳待建,含 preferences 字段,D25)· diet_log(⏳待建) · conversation_sessions(⏳待建)
  · conflict_rules(⏳仍是 JSONL,未进库)

Skills(按需加载,D22,⏳全部待写):
  菜谱/购物清单模板 · 调和层 rubric(含 harm-reduction 语气原则,D25)· 核查 pass 检查清单(含候选评估判定规则,D25)
```

---

## 1. 数据层:Postgres 统一存储

延续 D4(pgvector)+ D18(用户记忆同库)+ D23(不引入图数据库)+ D24(recipes 走关系表)的既定方向:**全部数据在同一个 Postgres 实例**,没有第二个数据库系统。本节列出每张表的字段、索引、幂等设计,以及当前是否已经写出 SQL。

### 1.1 已建表(SQL 已写,`db/schema.sql`)

| 表 | 用途 | 关键字段 | 索引 | 状态 |
|---|---|---|---|---|
| `knowledge_chunks` | RAG 向量检索(D2/D4/D23) | `chunk_id`(唯一) · `domain`(`tcm`\|`nutrition`) · `text` · `metadata JSONB` · `embedding vector(1024)` · `embed_model` | `domain` · `source_file` · HNSW(`embedding`,cosine) | ✅ 建表 SQL 写好并 review 过;**未在真实 Postgres 上跑过**(项目尚无已部署实例) |
| `recipes` | 按食材精确查菜谱(D24) | `name` · `ingredients TEXT[]` · `instructions TEXT[]` · `source` | GIN(`ingredients`) | ✅ 建表 SQL 写好;`db/load_recipes.py` 对 2000 条真实数据做过 dry-run(仅解析,未连库) |

⚠️ **`recipe_xiachufang.json` 的 5000 条 chunk 目前挂在 `knowledge_chunks` 的哪个 `domain`**:表的 `CHECK` 约束只允许 `tcm`/`nutrition` 两个值,这 5000 条 recipe chunk 实际落在 `nutrition` domain 下(靠 `source_type` 字段区分是不是菜谱)。D24 已经把"按食材精确查"这条主路径收窄到走 `recipes` 表,这些 chunk 的职责收窄为"给定模糊意图时的语义召回补充"(D24 第 2 点)。写检索工具时(`retrieve_nutrition`)要意识到结果里可能混着菜谱 chunk,必要时按 `source_type` 过滤。

### 1.2 待建表(有设计,SQL 未写)

| 表 | 用途 | 建议字段 | 索引/约束 | 对应决策 |
|---|---|---|---|---|
| `user_profile` | 常驻上下文的来源:体质、过敏原、补剂、目标标签、口味与情境偏好 | `id` · `constitution TEXT`(主体质,CCMQ 九分类之一)· `constitution_secondary TEXT[]`(次要体质/体质夹杂,D28)· `constitution_source TEXT`(`self_reported`\|`ccmq_computed`\|`unconfirmed`,D28)· `constitution_confirmed_at` · `allergens TEXT[]` · `supplements JSONB` · `goal_tags TEXT[]` · `preferences JSONB`(忌口/口味耐受/长期性用餐场景限制,如"不吃香菜""能吃辣""办公室没法热饭",D25)· `updated_at` | GIN(`allergens`);V1 单用户可以只有一行,但字段里留 `user_id` 给以后扩展 | D5 · D16 · D25 · D28 · PRD §10.2 人在环 |
| `diet_log` | 饮食记录明细,供 `query_diet_log()` 聚合查询 | `id` · `user_id` · `logged_at`(这顿饭实际发生的时间,用户输入或按时段规则推断,幂等键用它)· `recorded_at TIMESTAMPTZ DEFAULT now()`(系统实际写入时间,审计用,可能晚于 `logged_at`,比如补记)· `meal_type TEXT`(早餐\|午餐\|晚餐\|夜宵\|下午茶\|加餐\|未知,§4.2 按关键词/时段确定性推断,不额外调模型) · `raw_input TEXT` · `dishes JSONB`(拆解后的菜品) · `ingredients TEXT[]` · `food_properties TEXT[]`(中医食性标签,如"温/寒/平") · `idempotency_key TEXT UNIQUE` | `(user_id, logged_at DESC)` · `(user_id, meal_type)` · GIN(`ingredients`) · `idempotency_key` 唯一约束 | D18 · ENGINEERING §1.2 写路径幂等键 `(user_id, logged_at, raw_input_hash)`;`meal_type`/`recorded_at` 填补此前 PRD/DECISIONS 均未提及的空白 |
| `conversation_sessions` / `messages` | 会话历史原文,供分层压缩读取和生成摘要 | `session_id` · `turn_index` · `role` · `content` · `compression_tier`(0/1/2/3)· `created_at` | `(session_id, turn_index)` | D8 · §5.3 分层压缩 |
| `conflict_rules` | 冲突规则表迁移进库,供调和层按体质/目标结构化查询,而非把 40 条整份塞进 prompt | 对齐 `evals/conflict_rules.jsonl` 现有字段:`rule_id` · `tcm_position/source` · `nutrition_position/source` · `relation` · `resolution` · `resolution_rationale` · `confidence` · `evidence_level` · `applicable_constitutions TEXT[]` · `applicable_goals TEXT[]` · `source_status` | GIN(`applicable_constitutions`) · GIN(`applicable_goals`) | D23(关系表建模)· `evals/README.md` |
| `user_dish_aliases` | 个人菜品简称的程序性记忆(D27 修订一),记录"这个用户这样说时具体指什么",命中晋升阈值后跳过 LLM 兜底 | `id` · `user_id` · `normalized_phrase TEXT`(去空白/标点后的原始说法)· `dishes JSONB` · `ingredients TEXT[]` · `hit_count INT` · `promoted_at TIMESTAMP`(为空表示仍是候选,未生效) | `(user_id, normalized_phrase)` 唯一约束 | D27 修订一 · §4.2 |

⚠️ **`preferences` 和 `goal_tags` 是两件不同的事,不要合并(D25)**:`goal_tags` 回答"身体希望往哪个方向调理"(如 `weight_management`),`preferences` 回答"在满足调理方向的前提下,方案要符合哪些约束"(如"不吃香菜")。一次性的情境信息(某一次"今晚加班到很晚")不写进 `preferences`,直接随对话原文进入 SubAgent 任务上下文(见 §5.2 的上下文预算表),为每种可能场景建字段是过度设计。`preferences` 的写入遵循与 `allergens` 相同的人在环原则(PRD §10.2):首次使用及每次新增都需要用户显式确认,不从对话隐式推断。

⚠️ **体质字段为什么要拆成"主 + 次 + 来源 + 确认时间"四个字段,而不是一个 `TEXT`(D28)**:CCMQ 简版问卷计分常见"体质夹杂"——多个体质分类同时达标,不是异常情况,只取分数最高的一类会丢信息。`constitution_source` 区分这条信息是用户自己报的还是问卷算出来的,直接影响 TCM SubAgent 给建议时该用多确定的语气;`constitution_confirmed_at` 让"这条信息是多久前确认的"可查询,支撑以后"提醒用户重新测一次"这类产品功能,而不是让系统自己悄悄按饮食记录去猜测/漂移体质(体质是相对稳定的个人属性,不该跟着某一餐饭自动变化)。`constitution` 为空时 TCM SubAgent 的降级行为见 §11.3。

⚠️ **`conflict_rules.jsonl`(40 条,18 条 verified/22 条 needs_source)现在是纯文件,不在 Postgres 里。** 调和层要能"按体质/目标标签查命中的规则"(§6 步骤 5),这个查询模式和 `recipes` 表的 GIN 包含查询是同一类问题(D24 的精神:精确/结构化过滤查询该走关系表,不该走模型自己在 prompt 里翻 40 条规则)。建表后 `evals/conflict_rules.jsonl` 变成"人工编辑的源文件",经一个幂等 ingest 脚本(`ON CONFLICT (rule_id) DO UPDATE`,呼应 ENGINEERING §4.1)灌进 `conflict_rules` 表,查询走表,编辑仍然走 JSONL——两者不冲突,JSONL 保留作为 diff 友好的编辑格式。

`conflict_gaps.jsonl`(PRD §11:冲突且无规则命中时记录)按 PRD 原文就是文件,不建表——它是"下一版规则表的扩充素材",人工审阅后手动并入 `conflict_rules.jsonl`,不需要查询能力,建表反而多余。

### 1.3 连接池与迁移(ENGINEERING §4.1/4.3 的落地)

- 迁移用 Alembic,`db/schema.sql` 现在是手写的"初版建表脚本",进 Alembic 后应转成第一个 migration,后续表结构变更(比如给 `user_profile` 加字段)都走新 migration,不再手改 `schema.sql`。
- ingest 脚本(`db/load_recipes.py`、`db/embed_bge_m3.py`、未来的 `conflict_rules` ingest)与 API 服务用不同的连接池,避免全量 ingest 时把 API 的连接吃光。

---

## 2. MCP Server 设计(D7 修订)

### 2.1 为什么是这个形态

D7 修订的结论:工具层整体经**本地 MCP server**(stdio/localhost,不对外网络暴露)暴露给中枢 agent 及各 SubAgent。核心理由是把 PRD §10"工具权限"这条 guardrail 从应用代码里的 if 判断,变成协议层边界——**未声明的工具在协议层就不存在**,越权调用连"被拒绝"的机会都没有。完整论证见 `DECISIONS.md` D7。

**这套检索层本质是 agentic RAG,不是一次性单发检索**——D20 行为点 #1 已经定了 TCM/Nutrition SubAgent 可以自主判断"要不要再检索一次、查什么",这本身就是"retrieval 交给 agent 自己掌握节奏"而非"固定一次 retrieve→generate"的经典 agentic RAG 形态,只是这份文档此前没有点名这个词。§2.6 在此基础上讨论要不要给检索本身(而不是"要不要调用检索"这个决策)再加一层查询改写/假设文档生成的增强,并明确这不是当前阶段的重点。

### 2.2 工具清单(具体签名)

| 工具 | 签名 | 底层存储/服务 | 调用方 |
|---|---|---|---|
| `retrieve_tcm` | `(query: str, top_k: int = 5, filters: dict \| None = None) -> list[RetrievedChunk]` | `knowledge_chunks WHERE domain='tcm'`,向量检索 | TCM SubAgent(事实查询/候选评估/单领域/完整推荐分支均可能调用) |
| `retrieve_nutrition` | `(query: str, top_k: int = 5, filters: dict \| None = None) -> list[RetrievedChunk]` | `knowledge_chunks WHERE domain='nutrition'` | Nutrition SubAgent(同上) |
| `query_recipes_by_ingredients` | `(ingredients: list[str], match: "any"\|"all" = "all", limit: int = 20) -> list[Recipe]` | `recipes` 表,GIN `&&`/`@>`(D24) | 菜谱生成工具(D17)· 完整推荐分支的输出组装步骤 |
| `query_weather` | `(city: str, date: str \| None = None, include_recent_days: int = 3) -> WeatherInfo` | Open-Meteo,3h 缓存(ENGINEERING §3),熔断后回退节气表(ENGINEERING §1.3) | 中枢 agent · TCM SubAgent(判断"气候骤变"需要近几日数据,PRD §3.1) |
| `query_diet_log` | `(time_range: str, aggregation: "by_ingredient"\|"by_property"\|"by_nutrient"\|"raw", limit: int \| None = None) -> DietLogSummary` | `diet_log` 表 | 中枢 agent(记录回顾分支、主动发现"连续同主料")· TCM SubAgent(食性分布)· Nutrition SubAgent(营养素构成) |
| `write_memory` | `(category: "critical"\|"daily_log", payload: dict, idempotency_key: str \| None = None) -> WriteResult` | `category=critical` → 写 `user_profile`(需人在环确认);`category=daily_log` → 写 `diet_log`(需 `idempotency_key`) | **仅中枢 agent**;SubAgent 不持有这个工具(见 §2.3) |

`filters` 字段(`retrieve_tcm`/`retrieve_nutrition`)对应阶段 2.4 提到的混合检索:`WHERE 体质匹配 AND source_status='verified' ORDER BY embedding <=> $1`,是向量检索前的结构化预筛,不是另一套检索机制。

⚠️ **`query_diet_log` 的 `time_range` 要接受相对表达("昨天""今天""上周"),不能只接受绝对日期区间**——"我昨天晚上吃了什么"这类记录回顾分支的问法(D25)天然是相对日期。解析这类表达需要一个明确的时区基准,建议用 `user_profile` 里用户所在城市推出的时区,而不是服务端所在机器的系统时区(两者在部署到云端时大概率不一致,是一个容易踩的坑)。具体用哪个基准目前还是待决问题(`DECISIONS.md` 待决问题表),这里先把坑点写清楚。

⚠️ **"候选评估"分支(D25)不需要新工具。** 它和"单领域"/"完整推荐"复用完全相同的 6 个工具,区别只在 SubAgent 收到的任务描述从"生成一份新方案"变成"评估我给你的这个/这几个候选",这是 prompt 层面的任务框架切换,不是工具能力的缺口——遇到"要不要为新场景加新工具"这类问题时,先看现有工具组合能不能通过换一种任务描述覆盖,这也是 D21/D24 一直坚持的"不加不必要的组件"的同一条原则。

### 2.3 权限分层:不同调用方看到不同的工具子集

这是把 §10 工具白名单具体落地的地方——**同一个 MCP server,按调用方角色返回不同的工具列表**,而不是所有角色共享一份工具清单靠业务代码里判断"这个角色能不能调这个工具"。

| 调用方 | 可见工具 | 不可见(协议层不存在) |
|---|---|---|
| 中枢 agent | 全部 6 个,含 `write_memory` | — |
| TCM SubAgent | `retrieve_tcm` · `query_weather` · `query_diet_log`(只读) | `write_memory`、`retrieve_nutrition`、`query_recipes_by_ingredients` |
| Nutrition SubAgent | `retrieve_nutrition` · `query_diet_log`(只读) | `write_memory`、`retrieve_tcm`、`query_weather` |
| 调和层 | 无工具(只做一次推理,D14"不接收原始检索内容") | 全部 |
| 核查 pass | 无工具(只做判定,D15"不做规划、不调工具") | 全部 |

**写权限只在中枢一处**这条本身就是一个 guardrail:两个 SubAgent 即便被注入攻击也无法写用户记忆(呼应 PRD §10.3 OWASP LLM08 过度自主的缓解)。

### 2.4 传输与部署

stdio 或 localhost-only(不对外网络暴露),与业务进程同机部署。开发期额外收益(D7 修订理由二):同一套工具可以直接挂到 Claude Desktop 上做交互式调试,不用起完整五段管线就能验证"这条 query 到底召回了什么"——`planning/step1-naive-rag/` 里现在只能命令行跑的原型,MCP 化之后可以交互式验证。

### 2.5 状态

⏳ **完全未实现。** `db/embed_bge_m3.py` 里的 `search` 子命令是这个能力最接近的雏形(直接查 `knowledge_chunks`),但它是独立 CLI 脚本,不是 MCP 工具,没有走协议层,也没有权限分层。真正的 MCP server 代码(`backend/mcp_server/` 下应该有的 `server.py` 和每个工具的实现)一行没写。

### 2.6 检索增强:MQE / HyDE(agentic RAG 的后置增强,非当前重点)

**先说清楚这条和 §2.1 提到的"agentic"是两件不同的事**:§2.1 说的是"要不要再调用一次检索工具"这个决策交给 SubAgent(D20 已经这样定了,现在就生效);本节说的是"每一次调用 `retrieve_tcm`/`retrieve_nutrition` 内部,检索这个动作本身要不要做得更聪明"——后者是工具实现内部的增强,不是新增一个 agent 决策点,不需要修改 D20 的"五处 agent 行为"清单。

**要解决的具体问题**:中医知识库是文言/术语密集文本("痰湿质忌肥甘厚味"),用户的真实提问是口语化的("我最近舌苔厚、人也油,该少吃点什么")——这正是 `planning/roadmap.md` 阶段 2 点名的"分数好看反而是坏消息"那类词汇鸿沟(vocabulary gap)。两种成熟技术分别从两个方向缓解:

| 技术 | 做法 | 解决什么 |
|---|---|---|
| **MQE**(Multi-Query Expansion,多查询扩展) | 把用户原始 query 改写成 2-4 个不同措辞/角度的查询(口语版、术语版、症状拆解版),分别检索后合并去重、按分数重排 | 用户一次提问里混了多个信息点(比如同时提到舌苔厚和怕冷),单一 query 的向量检索容易被其中一个信息点主导,漏掉另一个 |
| **HyDE**(Hypothetical Document Embeddings) | 先让 LLM 针对用户 query 生成一段"假设性的理想答案"(不要求事实正确,只要求措辞贴近知识库文体),用这段假设文本去做向量检索,而不是直接嵌入用户原始口语 query | 直接缓解上面那条词汇鸿沟——假设答案的行文风格天然更接近术语密集的知识库原文,检索到的是"文体相似"的段落,再用真实检索结果生成最终答案 |

**放在哪里实现**:两者都应该封装在 `backend/mcp_server/tools/retrieve_tcm.py`/`retrieve_nutrition.py` 内部,对调用方(SubAgent)透明——SubAgent 仍然只是调用 `retrieve_tcm(query, top_k, filters)`,不需要知道内部是不是做了查询改写。不通过给 SubAgent 暴露新参数(比如 `use_hyde: bool`)来实现,那会把"要不要用增强检索"变成模型每次要做的额外判断,增加一个没有 eval 证据支撑的决策维度,与 D20"可评估性优先于自主性"的一贯立场相反。

**为什么现在不做,放在后面**:
1. **没有 baseline 数字支撑**——阶段 2 的 recall@5 还没有跑出"这两个具体失败模式确实存在且频繁"的证据,现在加等于凭直觉引入复杂度,违反 D23/D9 反复强调的"用真实 recall 数字而非理论推演做决策"。
2. **每次检索多出 1-2 次 LLM 调用**(MQE 的查询改写、HyDE 的假设文档生成),直接影响 M10 首字节延迟和单次查询成本——这类成本在骨架期(阶段4)优先级低于把五段管线跑通。
3. 这类增强应该按 roadmap 阶段 2 的方法论来验证:先测出"哪类 query 因为词汇鸿沟召回失败"(错误归因),再决定上 MQE 还是 HyDE 还是都不需要,而不是两个都先加上。

**建议时机**:阶段 8(重跑 eval)前后,用当时的真实 recall@5 数字判断这两项增强能不能带来可测量的提升;引入前先在 `evals/` 里跑一次消融对比(加/不加),证明有提升才保留,呼应 D1 用 B2 ablation 验证架构选择的同一套方法论。引入后需要在 `DECISIONS.md` 新增一条决策记录,不是静默改代码。

---

## 3. 工具调用机制

### 3.1 两层机制,不要混为一谈

```
LLM 的原生 function-calling(tool_use)          MCP 协议
────────────────────────────────              ─────────────────────
模型 API 层面的能力:                            应用层面的传输协议:
模型看到工具的 JSON Schema 定义,                 客户端(中枢/SubAgent 进程)
决定"要不要调、调哪个、传什么参数",                通过 stdio/localhost
返回一个 tool_use 请求                          向 MCP server 发起调用,
                                                server 执行后把结果传回
```

这两层是正交的:**模型决定"要不要调用工具"和"调用哪个工具",这是 tool_use 机制;工具真正怎么执行、执行时有没有权限,这是 MCP 协议管的事。** MCP server 只声明"你(某个角色)能看到哪些工具",工具的 JSON Schema(参数长什么样)仍然要喂给模型的 tool_use 定义,两边要对齐。

### 3.2 一次调用的完整链路

```
1. 中枢/SubAgent 的 LLM 调用返回 tool_use(工具名 + 参数)
2. 业务代码里的 Agent Loop 拦截这个 tool_use
3. 通过 MCP client 向本地 MCP server 发起对应方法调用
4. MCP server 校验:这个角色的会话是否声明了这个工具(§2.3)
   → 未声明:协议层拒绝,记录越权尝试(PRD §10 工具权限"越权调用拒绝并记录")
   → 已声明:执行(查 Postgres / 调 Open-Meteo / 写 user_profile 等)
5. 结果经 MCP 协议传回业务代码
6. 业务代码把结果拼成下一轮的 tool_result,喂回 LLM 上下文
7. LLM 决定是否继续调用工具,还是终止 loop 给出结论
   → 终止条件靠 tool_use 的有无判断,不是硬编码的固定轮数
   (roadmap 阶段 1.1 Hello-Agents 第 4 章练习的正是这个 loop)
```

### 3.3 SubAgent 的工具子集怎么落地

两种实现方式,选哪个是实现期要定的细节,不是架构级决策:

| 方式 | 做法 | 权衡 |
|---|---|---|
| **A. 单 server,按角色初始化不同 session**(推荐) | MCP server 启动一次,每次 SubAgent 派发时以对应角色身份建立一个新的 client session,server 按角色返回工具列表 | 只需一个 server 进程;角色与工具映射集中在一处(§2.3 那张表),改起来简单 |
| B. 每个角色一个独立 server 进程 | TCM/Nutrition/中枢各自起一个 MCP server,物理隔离 | 隔离性更强,但进程管理复杂度上升,对 V1 单机部署(D4"单机部署简化交付")没有必要的收益 |

**推荐 A**,和 D4/D23 一以贯之的"能不加新组件就不加"的精神一致。

### 3.4 状态

⏳ **完全未实现。** Agent Loop 本身(接收 tool_use → 执行 → 喂回结果 → 判断终止)也还没有代码;roadmap 阶段 1.1 里安排了 Hello-Agents 第 4 章的手写练习作为这部分的"肌肉记忆"前置,阶段 4.2 第 4 项("中枢 agent + Agent Loop")是真正落地的地方。

---

## 4. 记忆系统设计

### 4.1 三层记忆,访问方式不同(PRD §12.4 的具体化)

| 层 | 内容 | 访问方式 | 对应存储 | 为什么这样分层 |
|---|---|---|---|---|
| 常驻上下文 | 用户画像(0.5k)· 长期趋势标签(0.3k) | 每次推理直接读,永不压缩 | `user_profile` | 体积小、每步都需要;这是 M6/M6b 的保护对象 |
| 数据库查询 | 饮食记录明细与聚合 | 工具查询 `query_diet_log()`,不进上下文 | `diet_log` | D18:数据已持久化,查询比携带更准确(可按需聚合),且不占上下文预算 |
| 会话历史(分层压缩) | 当前会话与近期会话的对话原文/摘要 | 常驻中枢上下文的"会话历史"分区(≤10k tokens) | `conversation_sessions` | 体积会随对话轮数增长,是唯一需要压缩的部分(连同检索结果) |

**长期趋势标签怎么来**:不是用户手填的,是从 `diet_log` 周期性聚合出来的("最近偏肥甘厚味"这类判断),写回 `user_profile.trend_tags`(或独立小表)。这条链路(何时重新计算、由谁触发)是待实现细节,建议:每次"记录"分支写入后,异步触发一次轻量重算,不放在请求关键路径上,避免拖慢 M10 首字节延迟。

### 4.1.1 用两把外部尺子检查这套三层设计(D27,新增)

D27 用《AI Agents in Depth》第 3 章的两套分类法反过来检验上面这张表,结论是"方向对,但有一处字段设计可以补强",不是推翻重来。

**认知类型(书 §3.1.5:语义 / 情景 / 程序性记忆)**:

| 认知类型 | diet_expert 对应内容 | 现有位置 |
|---|---|---|
| 语义记忆(稳定的一般性知识) | 体质 · 过敏原 · `preferences` · `goal_tags` · `trend_tags` | 常驻上下文(`user_profile`) |
| 情景记忆(具体事件) | `diet_log` 明细 · `conversation_sessions` 单轮记录 | 数据库查询 + 会话历史 |
| 程序性记忆(行为模式) | **个人菜品简称(D27 修订一)**:用户反复用同一种说法记录同一道菜,系统从"LLM 兜底解析 + 人在环确认"这条回路里学出一条可复用的捷径 | `user_dish_aliases`(§1.2、§4.2),晋升规则是确定性计数阈值,不是模型判断 |

**修订(D27 修订一,2026-08-26)**:最初判断"程序性记忆 V1 不做",理由是"系统层面的行为模式没有真实场景"——这个判断对"系统怎么做事"这个粒度是对的,但漏看了一个更小、更具体的粒度:用户自己反复使用的说法。这属于程序性记忆的教科书场景(从重复行为里学出可复用的做事方式),且有客观的正确性标准(人在环确认过的结果就是 ground truth),不引入不可验证的自适应黑盒。语义/情景两类记忆的结论不变,只推翻了程序性记忆这一格。完整论证见 `DECISIONS.md` D27 修订一。

**存储格式四分法(书 §3.1.3:Simple Notes → Enhanced Notes → JSON Cards → Advanced JSON Cards,选择标准是"关键、低容量→Advanced JSON Cards;大容量、非关键→Simple Notes")**:`user_profile` 里体质/过敏原/`preferences` 完全符合"关键、低容量",验证了做成结构化字段(而非自由文本备注)是对的方向;但简单数组(如 `allergens TEXT[]`)只有值、没有"这条信息是什么时候/以什么方式确认的"——不是完整的 Advanced JSON Cards。这一点在体质字段上最要紧,直接导出 D28 的 `constitution_source`/`constitution_confirmed_at` 设计(§1.2)。`trend_tags` 符合 Simple Notes 定位(一句话结论,容错率高),现有设计不改。`diet_log` 是结构化事务性日志,不套用这套"记忆条目怎么表示"的分类,继续走 SQL 建表(§1.2 既有方向)。完整论证见 `DECISIONS.md` D27。

### 4.2 新信息怎么写入——"记录"分支详细流程

```
用户输入(自由文本,如"晚上吃了番茄炒蛋加一小碗白米饭")
   │
   ▼
1. 菜品拆解                                    [确定性优先,见下,三级查找,D27 修订一]
   · 先查 dish_ingredient_map 表(人工维护的菜名→食材映射,全体用户共享)
   · 未命中 → 查 user_dish_aliases(仅本用户,仅 promoted_at 非空的行)
   · 仍未命中 → 兜底走一次低成本 LLM 结构化输出调用,标记 confidence=low
     · 若人在环确认(步骤 4)时用户未修改这次的拆解结果 →
       normalized_phrase 计数 +1;累计达到晋升阈值(建议 3 次)→ 写入/更新
       user_dish_aliases 并置 promoted_at,之后同样的说法直接命中第二级,不再兜底
   │
   ▼
2. 过敏原即时检查                               [确定性代码,不经过模型]
   · 拆解出的 ingredients 与 user_profile.allergens 做集合比对
   · 命中 → 硬阻断该食材,提示用户,记录日志(呼应 PRD §10 输出拦截)
   │
   ▼
3. 关键事实落库                                 [write_memory(daily_log)]
   · idempotency_key = hash(user_id, logged_at, raw_input)
   · 写入 diet_log,重复调用(比如重试)不产生重复记录
   │
   ▼
4. 人在环确认                                   [PRD §10.2]
   · 展示拆解结果,用户确认或修改
   · 若本轮同时命中"首次提及的过敏原/补剂"→ 额外触发 user_profile 的 critical 写入确认
```

**"菜品拆解"为什么优先做成确定性查表,不是直接丢给模型**:ENGINEERING §7.3"确定性优先原则"——能不经过模型判断的就不经过模型,`dish_ingredient_map` 命中的部分可以 100% 单测覆盖,这是过敏原防护这条 Critical 档安全边界要求"可穷举验证"的前提。查表未命中才退回 LLM,这部分的输出天然置信度较低,需要人在环确认兜底。

⚠️ **`dish_ingredient_map` 本身是一个待建的数据资产**(常见菜名 → 典型食材构成的映射表),目前不存在,覆盖率也没有评估过。这是"记录"分支能不能达到 PRD §12.5 的 < 3s 目标延迟的关键——如果大部分输入都要走 LLM 兜底,延迟和成本都会上升。建议先收集 100-200 条高频家常菜作为种子表,覆盖率不足的部分用 eval(E3 记忆子集)测出来再补。`user_dish_aliases`(D27 修订一)是这条覆盖率问题的个人化补充——全局种子表覆盖不到的"这个用户自己的说法"(比如口头简称、方言叫法),靠重复确认自动晋升,不需要人工维护。

⚠️ **`user_dish_aliases` 的晋升阈值(建议 3 次)是待实测调整的常数,不是钉死的**:阈值太低会把偶然一次的措辞误判成稳定说法(比如用户那次其实打错字,LLM 恰好猜对了两次),阈值太高则个人化收益要等很久才生效。建议留成配置项,阶段 7/8 跑 eval 时用真实数据校准,不是本文档现在就能定死的数字。

### 4.3 关键事实落库前置——跨路由分支的常驻检查(新增设计)

PRD §12.4 写的是"关键事实须在压缩发生前落库",但字面上没说清楚这个检查只发生在"记录"分支,还是每一轮对话都要过一遍。**本文档在此明确一个设计决策(不与已有决策冲突,是填补一个空白)**:

用户可能在**任何**分支说出关键事实——比如在问"今天该吃什么"(完整推荐分支)时顺带提一句"对了我对虾过敏"。如果只在"记录"分支做落库检查,这句话会被完整推荐分支当成普通输入,最终随会话历史一起被压缩掉,重现 PRD §7 那个经典 bug 场景。

**因此**:在路由判断(总览图③)之前,加一步跨分支的关键事实扫描(总览图②)——对当轮原始输入跑一次确定性关键词/规则扫描(过敏原词表、"我在吃/服用"这类补剂提及模式),命中就调用 `write_memory(critical)` 并走人在环确认,然后再进入原有路由,不影响六条分支(D25)各自的判断逻辑。这一步是确定性代码,不额外增加 LLM 调用,对延迟影响可忽略。

### 4.4 分层压缩(D8,D27 修订,roadmap 阶段 7)

压缩对象:会话历史 + 检索结果(D18 之后,饮食记录已经移出压缩范围)。

| 层级 | 内容 | 保真度 |
|---|---|---|
| Tier 0 | `user_profile` 字段 | 不参与压缩,永久原文 |
| Tier 1 | 当前会话最近 N 轮 | 原文 |
| Tier 2 | 当前会话中较早的轮次 | 结构化归档摘要(见下) |
| Tier 3 | 跨会话的历史会话 | 日期 + 结构化归档摘要 + `trace_id` 指针(不含原文) |

触发条件:上下文预算超限(PRD §12.3 各角色的 token 预算)时触发;若压缩后仍超限,丢弃最旧的低价值记录(PRD §11 fallback 表,即压缩的熔断兜底)。

**压缩优先级表(D27,新增)**——原设计没有回答"检索结果里哪部分该先删",补一条显式规则,直接回应"工具调用的记忆是不是可以先删掉"这个问题:

| 内容类型 | 处理方式 | 理由 |
|---|---|---|
| 检索到但最终未被结论引用的 chunk | 压缩触发时**直接删除,不摘要** | 摘要噪声本身就是浪费——没进最后结论,说明它对当前判断没价值,不值得先花一次 LLM 调用去总结它 |
| 检索到且被结论引用的 chunk | 只保留 `source_id` + 一句话结论,原文可丢弃 | 溯源展开(§5.2 步骤 8)靠 `source_id` 回查数据库,会话历史里留原文副本是重复存储 |
| 调和层/核查 pass 的结论与理由 | 保留结构化摘要(见下) | 本次交互真正有价值的产出,优先级最高 |
| 未通过核查、被移除的建议条目 | 保留"被拒绝 + 拒绝理由"一行,不留完整原文 | 是一条有用的负样本,但不需要携带当时的完整生成内容 |

**结构化归档摘要(D27,替代自由文本摘要)**——Tier 2/3 的摘要固定用这个模板,不是自由文本 LLM 摘要:

```
{turn_id} | {branch}(六条分支之一) | 结论:{一句话} | 引用:{source_id 列表} | 被拒建议:{若有} | 触发的 guardrail:{若有}
```

每条摘要是独立的结构化记录(类比"像 `git log` 不是 `git squash`"),这样 Tier 3 可以被结构化查询(比如"上次因为过敏原被拒绝的建议是什么"),自由文本摘要做不到。完整论证与"为什么工具调用内容该先丢"的详细分析见 `DECISIONS.md` D27。

**压缩内容的"重要性判断"用固定阈值(workflow 形态),不用模型判断**——这部分维持原判断不变,呼应 D20"可评估性优先于自主性"的一贯立场;模型判断哪些内容"重要到该留原文"作为 V1.5 的可选升级,升级前需要专门的 eval 案例证明它比固定阈值更好。

### 4.4.1 压缩触发时机(D27 修订二,具体设计)

原设计只说"固定阈值",没给出阈值和检查点本身。触发机制分两级,对应两种不同生命周期的压缩对象:

**SubAgent 内(单次请求生命周期,同步,不需要 LLM 调用)**:每次 `tool_result` 追加后,与 §4.5 状态提示同一处代码顺带检查"检索结果"子分区(§12.3,预算 12k)的估算 token 数;超过约 80%(≈9.6k)时,按上面的压缩优先级表就地丢弃未被引用的 chunk,直到回落到阈值以下再继续下一轮工具调用。这是纯内存操作,不落库、不异步——SubAgent 的生命周期本来就只有一次请求那么长,没有"异步"这个选项。

**中枢会话历史(跨轮次/跨会话,persistent 到 `conversation_sessions`)**:检查点分别在 §5.2 步骤 2 和步骤 9,职责不同:

| 检查点 | 时机 | 做什么 |
|---|---|---|
| 步骤 9(会话落库) | 响应已发出**之后**,异步 | Tier 1 累计估算 token 超过约 6k(会话历史 10k 预算的 60%,给 Tier 2/3 留空间)→ 对最旧若干轮做一次结构化归档摘要(§4.4 模板,需一次低成本 LLM 调用),写入 Tier 2,原文可移除。会话判定结束(空闲超过约 30 分钟,阈值待实测调整)→ 该会话全部 Tier 2 摘要折叠进 Tier 3。和 `trend_tags` 异步重算(§4.1)是同一个模式,不占用当前请求的响应时间 |
| 步骤 2(准备派发上下文) | 组装**下一次**请求的上下文时,同步,只做兜底 | 正常情况下步骤 9 应该已经让 Tier 1 回落到阈值以下;万一没有(异步任务积压/失败,或短时间连续多轮消息来不及处理),**不等待、不同步调用 LLM 摘要**(压缩允许有延迟,但正在被回答的这一轮不能被压缩逻辑拖慢请求本身),直接应用 PRD §11 fallback 表已有的"丢弃最旧的低价值记录"——从 Tier 3 最旧的一条开始丢,不从还没来得及摘要的 Tier 1 丢(丢 Tier 1 原文等于丢了本该异步生成、还没生成的摘要,信息损失更大) |

这套"正常路径异步、紧急路径同步硬兜底"的设计,保证压缩本身不出现在任何一次请求的关键路径耗时里,与 D26(FastAPI,SSE 优先保证首字节延迟)的取向一致。完整论证见 `DECISIONS.md` D27 修订二。

### 4.5 SubAgent 循环状态提示(D27,新增)

D20 的五处"agent 行为"里,只有行为点 #1(TCM/Nutrition SubAgent 自主决定是否再调用一次工具)是**开放式循环**——路由、调和、核查都是固定 workflow 步骤,不存在"要不要继续"这个问题。状态提示只加在这一处,不是给整条链路都装一个通用状态栏组件。

**设计**:SubAgent 每一轮决定"是否再调用工具"之前,由**代码**(不经过任何 LLM 调用)拼接一条消息,追加到该 SubAgent 上下文末尾:

```
[状态] 已用工具调用:{n}/15(资源限额,§5.4)· 已检索到的候选信息要点:{code 生成的短列表,不是全文摘要}
```

两条硬约束:(1)必须由代码计算,不能让模型自己总结"到目前为止发生了什么"——自总结的状态提示更容易和实际状态脱节;(2)"候选信息要点"是从已发生的 `tool_use`/`tool_result` 事件里用确定性规则抽取的短列表(比如已查过哪些食材、已查过哪几天天气),不携带任何检索原文,和 §4.4 的压缩优先级表(工具调用原始内容优先丢弃)保持一致。

⚠️ **风险(书中称"状态栏投毒")**:模型会无条件信任这条状态消息,如果计数或抽取逻辑本身有 bug,错误会直接传导成错误决策(比如提前终止本该继续检索的循环)。因此这段代码必须被单测覆盖(ENGINEERING §7.3"确定性优先"),不能等到集成测试才发现计数错了。完整论证见 `DECISIONS.md` D27。

### 4.6 状态

⏳ **完全未实现。** `user_profile`(含 D28 新增体质字段)、`diet_log`、`conversation_sessions`、`dish_ingredient_map` 四张表都没有 schema;压缩组件(含 D27 的优先级表/结构化摘要)、关键事实扫描、状态提示、`write_memory` 工具均无代码。roadmap 阶段 7 是这部分的主战场。

---

## 5. 用户交互 Pipeline(完整请求生命周期)

### 5.1 分级路由回顾(D12/D25,六条分支)

D25(2026-08-26)按真实用户问法走查后,把原来的四条分支扩为六条——新增"记录回顾"与"候选评估",完整论证见 `DECISIONS.md` D25。

| 分支 | 触发条件 | 路径 | 目标延迟 |
|---|---|---|---|
| 记录 | 陈述性的饮食记录输入("晚上吃了番茄炒蛋") | §4.2 全流程 | < 3s |
| 记录回顾 | 查自己的历史记录("我昨天晚上吃了什么") | `query_diet_log` 单工具查询,不经过 SubAgent | < 3s |
| 事实查询 | 单一事实性提问,查静态知识库("红枣是什么性味") | 检索 + 回答 + 核查 | < 4s |
| 候选评估 | 给定具体候选,要求判断("这个能不能吃""黄焖鸡和米线选哪个""已经吃了大补的,还能吃什么") | 双派发(任务框架=评估候选)+ 视候选数量决定是否调和 + 核查 | < 20s |
| 单领域 | 明确只涉及一侧体系的问题 | 单派发 + 核查 | < 14s |
| 完整推荐 | 需要综合判断的问题("今天该吃什么") | 双派发 + 调和 + 核查 | < 35s |

路由判断本身由谁完成(中枢模型 / 轻量分类器 / 规则)是待决问题(`DECISIONS.md` 待决问题表),留到实现路由时定;不管选哪种,M13(路由准确率)都要能独立测量。**"记录回顾"vs"事实查询"、"候选评估"vs"完整推荐"这两组分支容易被合并,不要合并**——前一组的检索目标不同(用户自己的数据 vs 静态知识库),后一组的核查判定标准不同(候选评估的核查规则见 §6),合并会让 M13/M14 的判据变得模糊,详见 D25。

### 5.2 完整推荐分支——详细步骤

这是最复杂的一条路径,也是简历叙事里"双专家 + 调和"的核心展示对象。

```
0. 输入防护(截断/指令注入过滤/疾病用药检测) ── §0 总览图①
0.5 关键事实扫描(跨分支,§4.3) ── §0 总览图②
1. 路由判断 → 命中"完整推荐" ── D12,M13
2. 中枢准备派发上下文:
   读 user_profile(常驻)+ trend_tags(常驻)
   + 按需 query_diet_log(聚合"最近饮食食性分布")
3. 并行派发两个 SubAgent(asyncio.gather + return_exceptions=True,ENGINEERING §2)
   ├─ TCM SubAgent(独立 24k 上下文):工具 = retrieve_tcm · query_weather · query_diet_log(只读)
   │  agent 行为点 #1(D20):可多次检索/查天气,可从饮食记录里主动发现"连续三天同一主料"并自行判断是否提示
   │  每轮工具调用前追加代码维护的状态提示(§4.5,D27):已用调用次数/上限 + 已检索要点短列表
   │  `constitution` 为空/未确认时(D28):任务提示词显式声明"体质未知",建议收窄为体质无关的普适性温和建议,末尾附一句引导完善信息,不阻塞对话
   └─ Nutrition SubAgent(独立 24k 上下文):工具 = retrieve_nutrition · query_diet_log(只读)
      同样追加状态提示(§4.5,D27)
   每个 SubAgent 的循环终止:无新增 tool_use,或触达 §10 资源限额(单会话 ≤15 次工具调用)
   任务上下文(PRD §12.3 的 2k 那部分)携带用户的原始提问文本,不只是结构化字段——"加班到很晚了"这类一次性情境信息靠 SubAgent 直接读原文理解,不需要为它单独建结构化输入维度(D25)
4. 收敛两侧结论(各自 ≤6k token 摘要,回传中枢)
5. 调和层(独立 LLM 调用,D14,≤16k 上下文)
   输入:两侧结论与依据(≤10k)+ 命中的 conflict_rules(≤2k)+ user_profile(0.5k)
   不接收原始检索内容(避免重新引入上下文污染)
   加载 Skill:调和层 rubric(D22)
   agent 行为点 #2(D20):依据不足时可回退请求 SubAgent 补充查询(建议上限 1 次,与核查 pass 的重试上限一致,实现后按实测调整)
6. 核查 pass(独立调用,D15,≤12k 上下文)
   输入:调和结论全文 + user_profile + 被引用 chunk 原文(不含对话历史)
   加载 Skill:核查清单(D22,PRD §10.1 七条检查项)
   只拒绝不改写;不通过 → 移除条目或退回调和层(有限次数)
7. 菜谱/购物清单组装(需要时)
   调用 query_recipes_by_ingredients(D24,精确食材过滤)
   加载 Skill:菜谱/购物清单模板(D22)
8. 流式输出 + 溯源可展开(前端渲染 source_id → 原文 chunk)
9. 会话落库(写 conversation_sessions)+ 按需触发异步分层压缩
10. 隐式反馈埋点就绪(次日饮食记录命中率,T+1 天才能算出,PRD §14.1)
```

### 5.3 其余五条分支(与完整推荐共享步骤,只是裁掉部分环节或换任务框架)

| 分支 | 与完整推荐的差异 |
|---|---|
| 记录 | 不经过步骤 3-8;直接走 §4.2 的菜品拆解 → 过敏原检查 → 落库 → 人在环确认。**这是唯一的写入链路**,路由误判为查询会导致记录丢失,后果比其他误判严重,因此单独计入 M13 |
| **记录回顾**(新增,D25) | 不经过步骤 2-8 的任何一步。中枢直接调用 `query_diet_log(time_range, aggregation="raw")`,把返回的记录格式化成自然语言回答;`time_range` 的相对日期解析见 §2.2 的时区提醒。不派发 SubAgent、不走调和层,核查 pass 简化为"回答里出现的记录条目确实来自 `diet_log` 查询结果"这一条确定性检查,而不是 PRD §10.1 那七条(那七条是为生成式建议设计的) |
| 事实查询 | 跳过步骤 3 的双派发和步骤 5 的调和层,只走单库检索(`retrieve_tcm` 或 `retrieve_nutrition`)+ 步骤 6 核查 pass |
| **候选评估**(新增,D25) | 步骤 3 的两个 SubAgent 收到的任务描述从"生成一份新方案"换成"评估用户给定的候选(dish_a[, dish_b, ...])",复用完全相同的工具;候选只有一个时跳过步骤 5 调和层(直接核查通过/不通过的结论),候选有多个需要排序取舍时才经过调和层;步骤 6 核查 pass 用 §6 里为这个分支单独加的规则,而不是要求"结论"字符串本身带 `source_id`;不生成菜谱/购物清单,跳过步骤 7 |
| 单领域 | 只派发一个 SubAgent(步骤 3 单边),跳过调和层(步骤 5),仍走核查 pass(步骤 6) |

核查 pass 在**所有查询路径**上都执行,不可跳过(PRD §12.5),但候选评估和记录回顾分支用的是各自简化/定制过的判定标准,不是完整推荐分支那套七条规则原样套用;调和层只在双派发且需要取舍/调和的路径执行(完整推荐总是需要,候选评估视候选数量而定)。

### 5.4 Guardrails 挂载点(对照 PRD §10)

| Guardrail | 挂载在哪一步 |
|---|---|
| 输入防护(指令注入/超长截断/疾病用药检测) | 总览图① |
| 关键事实落库前置 | 总览图②(§4.3) |
| 工具白名单 | MCP server 协议层(§2.3),不是某一步,是贯穿全程的边界 |
| 资源限额(≤15 次工具调用/≤80k token/≤60s) | SubAgent 循环内部 + 整链路超时(ENGINEERING §1.1 三层超时) |
| 循环防护(连续 3 轮无新增信息) | SubAgent 循环终止条件的一部分 |
| 输出拦截(诊断性表述/过敏原/无 source_id) | 核查 pass(步骤 6) |
| ED 防护四条 | 核查 pass(确定性规则部分)+ 输出分类器 |
| 人在环检查点 | §4.2 步骤 4;体质确认/低置信度建议在各自触发点 |

### 5.5 状态

⏳ **完全未实现。** 目前唯一跑通的是 RAG 检索本身(`db/embed_bge_m3.py search` 子命令能查 `knowledge_chunks`,但连不上真实 Postgres 前无法验证),路由、双派发、调和层、核查 pass、前端流式输出全部没有代码。roadmap 阶段 4 的任务 4-9 覆盖这一节的全部内容,且规定了顺序不能乱(后面依赖前面)。

---

## 6. Agent Skills 设计(D22)

D22 已经决定把哪些能力做成 Skill(而不是常驻 system prompt),本节给出具体文件清单**以及加载机制本身**(此前只有"要不要做成 Skill"的决定,没有"怎么加载"的具体设计)。

### 6.1 Skill 文件清单

| Skill 文件(建议路径) | 内容 | 加载时机 | 版本化方式 |
|---|---|---|---|
| `backend/skills/recipe_and_shopping_list.md` | 菜谱输出格式模板、购物清单格式模板 | 完整推荐分支步骤 7(需要具体菜谱时) | 文件内 header 写版本号,改一次记一行进 `worknotes.md` 的 prompt 迭代记录表 |
| `backend/skills/reconciliation_rubric.md` | 调和层的仲裁准则(如何在两套结论冲突时给出立场,而非折中回避);**新增一条(D25)**:用户明确表达"想吃不那么健康的东西"时,默认给出"怎么吃更聪明"的调和建议(harm reduction),而非单纯劝阻——除非命中过敏原/ED 硬性阻断规则 | 调和层每次调用(步骤 5) | 同上 |
| `backend/skills/verification_checklist.md` | 核查 pass 的七条检查项(PRD §10.1);**新增一条候选评估分支专用规则(D25)**:该分支输出是"结论(能/不能/选哪个)+ 理由",要求结论能拆解出至少一条带 `source_id` 的支持理由,而不是要求结论字符串本身携带 `source_id` | 核查 pass 每次调用(步骤 6);候选评估分支走这份清单里 D25 新增的那一条,而不是其余六条 | 同上 |
| `backend/skills/ccmq_questionnaire.md`(D22 补充) | CCMQ 简版问卷题库(九类体质)、追问话术、"不确定"选项处理、计分口径(D28 §11.3) | 首次使用引导走到问卷分支时(§11.2 步骤 3b) | 同上 |
| `backend/skills/ed_risk_response.md`(D22 补充) | ED 风险响应的预先审阅话术模板——命中风险后如何表达共情与转介,明确列出不该说的措辞 | 输出分类器/核查 pass 判定命中 ED 风险信号时(PRD §16,§5.4) | 同上;这份文件比其余几份更需要人工审校(内容涉及高敏感度表述),改动应记入 `worknotes.md` 时额外标注审校人 |

五份都不获得独立上下文或独立 LLM 调用,只是被装进已有的某次调用的 prompt 里(调和层自己的调用、核查 pass 自己的调用、引导流程自己的调用),不破坏 D14/D15"各自只有一次独立 LLM 调用"的性质。被考虑过但否决的候选 Skill(交互语气通用规范、特殊人群专项建议、节气细则、候选真实性预检查、追问澄清策略)及否决理由见 `DECISIONS.md` D22 补充。

### 6.2 加载机制:三层渐进式披露(参照《AI Agents in Depth》§2.5,按 diet_expert 的实际情况裁剪)

书中描述的通用 Skills 机制分三层:Layer 1 元数据目录(name + description,~300 token,常驻,供模型自主判断"这次该不该加载这个 skill")→ Layer 2 完整 SKILL.md(选中后才加载,~2k token)→ Layer 3 子文档(需要更深内容时按需再加载)。diet_expert 借用这套分层,但有一处关键差异需要显式说明,而不是照抄:

**diet_expert 的 Skill 加载是确定性的,不是模型自主选择的。** D20 把主链路定为固定 workflow——"调和层该不该加载 rubric"不是由模型读一段 description 后自己判断,而是由代码在到达步骤 5(调和层)时**必然**加载 `reconciliation_rubric.md`,到达步骤 6(核查 pass)时**必然**加载 `verification_checklist.md`。这意味着 Layer 1 元数据目录在 diet_expert 里**不承担运行时路由职责**(不存在"模型看了 description 决定不加载"这种情况),它的价值收窄为两个更朴素的东西:(1)人类可读的文档索引,写代码/加新 Skill 时用来确认"这个 skill 该在哪一步被触发";(2)给 `worknotes.md` 的 prompt 迭代记录表提供锚点。**这是刻意的简化,不是漏做了模型自主路由——D20 的判断是"结论正确性通过固定流程更可控",Skill 加载时机同理不适合交给模型自由裁量。**

具体分层:

| 层 | 内容 | 何时加载 | 落地方式 |
|---|---|---|---|
| Layer 1(目录) | 每个 skill 的 id/路径/一句话用途/触发步骤,~300 token | 不进任何 LLM 调用的 prompt,只是一份供人/文档索引查阅的注册表 | `backend/skills/registry.py`:一个 dataclass 列表,`{id, path, description, load_at_step, version}` |
| Layer 2(核心内容) | 三份 Skill 文件各自的完整正文(rubric 准则 / 七条检查项 / 输出模板) | 到达对应 pipeline 步骤时,由代码读取文件、拼进那一次调用的 prompt | 进程启动后首次读取即缓存在内存,不必每次请求都重新读盘;文件变更需要重启服务或显式失效缓存(V1 用重启即可,不做热重载) |
| Layer 3(子文档) | 调和层需要的、按体质/目标结构化查询出的 `conflict_rules` 命中行(D23/D24) | 调和层这次调用命中了具体规则时才拼入(§5.2 步骤 5 已经这样设计:"命中的 conflict_rules ≤2k") | 复用已有的 `conflict_rules` 表查询,不是新机制——这次只是把它显式归类为 Skill 体系里的 Layer 3 |

**关于 KV Cache 复用的一处诚实说明**:书中 Layer 1/2 的一个重要收益是"emit once, 之后的多轮对话都能复用 KV Cache 前缀"。这个收益在 diet_expert 里**不完全成立**——调和层和核查 pass 各自只有一次独立 LLM 调用(D14/D15),不是同一个长会话里的连续多轮,三份 Skill 之间也互不共享上下文。所以 diet_expert 从渐进式披露里拿到的主要收益是**预算收益**(只在真正用得上的那一次调用里装入对应内容,不常驻在 2.5k 的中枢 system prompt 预算里,呼应 D22 理由一),而不是书中强调的**跨轮次 cache 复用收益**——如实标注这一点,避免过度套用书里的收益论证。

### 6.3 状态

⏳ **五份 Skill 文件本身还没写,`registry.py` 也没有。** D22 只是决定了"要不要做成 Skill"以及"哪些能力该做成 Skill",本节(§6.2)补的是加载机制的具体设计,具体的 rubric 内容(调和层遇到冲突具体该怎么判断立场)、检查清单的判定标准细化、CCMQ 题库与计分口径、ED 响应话术,以及 `registry.py` 本身,都还没有落到代码/文件里。

---

## 7. LLM Adapter 层(D19,ENGINEERING §1 的落地位置)

严格来说这一层不是"agent 设计"的一部分,但所有对模型的调用(路由判断、两个 SubAgent、调和层、核查 pass、输出分类器)都要经过它,放在这里做个索引,细节见 `ENGINEERING.md` §1。

| 职责 | 建议文件 |
|---|---|
| 超时分层(20s/45s/90s) | `backend/llm/adapter.py` |
| 重试 + 指数退避 + jitter | 同上 |
| 熔断器(按外部依赖分别维护) | 同上 |
| 模型档切换(`MODEL_TIER=dev\|prod`,D19) | 同上,读环境变量,业务代码不感知具体模型名 |

### 状态

⏳ 未实现,roadmap 阶段 4.2 第 1 项明确标注"后补代价极大,必须第一个建"——建议在骨架期第一个动手的就是这一层,而不是先搭路由或 SubAgent。

---

## 8. 代码目录结构与测试结构(尚不存在,供开工时参考)

### 8.1 目录树

```
backend/
  llm/
    adapter.py                       § 7
  mcp_server/
    server.py                        § 2
    tools/
      retrieve_tcm.py
      retrieve_nutrition.py
      query_weather.py
      query_diet_log.py
      write_memory.py
      query_recipes.py
  agents/
    router.py                        § 5.1
    tcm_subagent.py                  § 5.2 步骤 3
    nutrition_subagent.py            § 5.2 步骤 3
    reconciliation.py                § 5.2 步骤 5
    verification.py                  § 5.2 步骤 6
  memory/
    critical_fact_scanner.py         § 4.3
    compression.py                   § 4.4/4.4.1(优先级表 + 两级触发)
    status_prompt.py                 § 4.5(SubAgent 循环状态提示,D27)
    dish_decomposition.py            § 4.2(dish_ingredient_map → user_dish_aliases → LLM 兜底三级查找)
    dish_alias_promotion.py          § 4.2(D27 修订一:计数与晋升逻辑,与查找逻辑分文件,便于单独测阈值行为)
  skills/
    registry.py                      § 6.2,Layer 1 目录(id/路径/触发步骤/版本)
    recipe_and_shopping_list.md      § 6.1
    reconciliation_rubric.md
    verification_checklist.md
    ccmq_questionnaire.md            D22 补充
    ed_risk_response.md              D22 补充
  guardrails/
    input_filters.py
    output_filters.py
    ed_protection.py
  onboarding/
    ccmq_scoring.py                  § 11.3,CCMQ 简版计分逻辑(含多体质夹杂判定)
    flow.py                          § 11.2,渐进式引导对话步骤
db/
  schema.sql                         已有,需扩充 user_profile(含 D28 体质字段)/diet_log/conversation_sessions/conflict_rules/dish_ingredient_map/user_dish_aliases
  migrations/                        Alembic,ENGINEERING §4.1
  load_recipes.py                    已有
  embed_bge_m3.py                    已有
  load_conflict_rules.py             ⏳待写,幂等 ingest,呼应 §1.2
api/
  main.py                            FastAPI,§10,ENGINEERING §9(容器化/部署)
  schemas.py                         § 10.1 Pydantic 请求/响应模型
frontend/                            Next.js,流式响应 + 溯源可展开(PRD §7)
tests/                                目录结构镜像 backend/,不与 evals/ 混淆(见 §8.2 的边界说明)
  conftest.py                        共享 fixture:mock LLM adapter、record/replay 加载器、测试用最小 schema
  fixtures/
    llm_replay/                      ENGINEERING §7.2 的录制响应,按 (调用方, 请求指纹) 命名
    fault_injection/                 429/超时/格式错乱响应,测 §7 重试与熔断
  unit/
    memory/
      test_critical_fact_scanner.py
      test_compression.py
      test_status_prompt.py
      test_dish_decomposition.py
      test_dish_alias_promotion.py
    guardrails/
      test_allergen_block.py
      test_ed_protection.py
      test_input_filters.py
    onboarding/
      test_ccmq_scoring.py
    skills/
      test_registry.py
    mcp_server/
      test_tool_whitelist.py
  integration/
    test_subagent_loop.py
    test_routing.py
    test_reconciliation.py
    test_verification.py
    test_api_chat_sse.py
    test_fallbacks.py                § 8.2:PRD §11 fallback 表逐行覆盖
```

### 8.2 测试结构与覆盖要求(ENGINEERING §7 三层金字塔的具体落地)

ENGINEERING §7.1 定了金字塔形状(单元/集成/eval)和一条硬要求("过敏原与 ED 相关路径要求 100%"),但没有落到"每个源文件对应哪个测试文件、测什么"这一层。下表把这层具体化——这是本次新增内容,直接回应"test 要写什么"这个问题。

| 源文件 | 测试文件 | 层级 | 必须验证什么 |
|---|---|---|---|
| `memory/critical_fact_scanner.py`(§4.3) | `unit/memory/test_critical_fact_scanner.py` | 单元,**要求 100%** | 过敏原/补剂关键词命中的穷举用例;跨分支触发(不止"记录"分支);误报(相似但非过敏原的词不触发) |
| `guardrails/*`(过敏原阻断、ED 拦截) | `unit/guardrails/*` | 单元,**要求 100%** | ENGINEERING §7.3"确定性优先"点名的三类:过敏原命中、药食同源白名单、ED 数值拦截,逐条穷举而非抽样 |
| `memory/compression.py`(§4.4/4.4.1) | `unit/memory/test_compression.py` | 单元 | 压缩优先级表的分支逻辑(未引用 chunk 直接删、被引用 chunk 只留 source_id)、结构化归档摘要模板的字段完整性、Tier 1→2→3 的阈值判断(不测真实 LLM 摘要质量,那是 eval 的事) |
| `memory/status_prompt.py`(§4.5) | `unit/memory/test_status_prompt.py` | 单元,**建议按 Critical 档同等严格** | D27 明确点名"状态栏投毒"风险的那段代码——计数正确性、要点抽取不携带原始检索文本、边界值(恰好等于 15 次时的行为) |
| `memory/dish_decomposition.py` + `dish_alias_promotion.py`(§4.2) | `unit/memory/test_dish_decomposition.py` / `test_dish_alias_promotion.py` | 单元 | 三级查找的优先级顺序(全局表 > 已晋升个人别名 > LLM 兜底)、晋升计数的边界(恰好达到阈值、被打断的计数序列)、`normalized_phrase` 归一化规则本身的用例(标点/空白差异应视为同一短语) |
| `onboarding/ccmq_scoring.py`(§11.3) | `unit/onboarding/test_ccmq_scoring.py` | 单元 | 九类体质转化分计算、体质夹杂(多个≥40分)的正确识别、"倾向是"(30-40分)进入次要体质而非丢弃、边界分值(恰好 40/30 分) |
| `skills/registry.py`(§6.2) | `unit/skills/test_registry.py` | 单元 | 文件读取缓存(不重复读盘)、version 号解析、触发步骤映射正确 |
| `mcp_server/server.py`(§2.3 权限分层) | `unit/mcp_server/test_tool_whitelist.py` | 单元 | 越权调用在协议层被拒绝(D7 修订的核心论点——不是"应用层判断",要在这一层验证) |
| SubAgent 循环编排(`agents/tcm_subagent.py` 等) | `integration/test_subagent_loop.py` | 集成(mock LLM) | 资源限额(≤15 次工具调用)触发终止、状态提示确实被追加、连续 3 轮无新增信息的循环防护 |
| 六分支路由(`agents/router.py`) | `integration/test_routing.py` | 集成(mock LLM 或规则) | D25 六条分支各自命中对应示例问法;"记录回顾 vs 事实查询""候选评估 vs 完整推荐"这两组易混淆分支的边界用例 |
| 调和层/核查 pass(`agents/reconciliation.py`/`verification.py`) | `integration/test_reconciliation.py`/`test_verification.py` | 集成(mock LLM) | Skill 内容确实被拼入对应调用的 prompt(而不是常驻);候选评估分支走的是 D25 新增规则而非其余六条;核查 pass 的"只拒绝不改写" |
| `api/main.py` 的 `/api/chat`(§10) | `integration/test_api_chat_sse.py` | 集成 | SSE 事件顺序(核查必须在第一条 `token` 事件前完成,§10.3 的硬约束);`trace_id` 贯穿 |
| PRD §11 fallback 表 | `integration/test_fallbacks.py` | 集成 | ENGINEERING §7.1 原文"没测过的降级路径 = 不存在的降级路径"——表里每一行一条对应用例,包含本次新增的"Tier 1 回落失败时从 Tier 3 丢弃"这条(D27 修订二) |

**与 `evals/` 的边界,避免混淆**:`tests/` 测的是"代码逻辑对不对"(确定性可断言,CI 门禁,ENGINEERING §8),`evals/` 测的是"输出质量好不好"(M1-M14,允许波动,不是 pass/fail 的单测)。两者都要写,但不是同一件事——`tests/integration/` 里出现的"调和层被正确调用"和 `evals/` 里出现的"调和层给出的调和建议质量如何"是两个不同层面的问题,写代码时容易把后者也塞进单测,应该避免。

### 8.3 状态

⏳ **骨架已建,逻辑全部未实现。** `backend/`、`api/`、`tests/`、`db/migrations/`、`frontend/` 目录与本节列出的文件已按此结构建出(2026-08-26),但每个源文件只有指向本文档对应章节的 docstring 占位,每个测试文件用 `pytest.mark.skip` 标注待实现——`pytest --collect-only tests/` 能跑通、17 条占位用例全部可收集,但没有一行真实逻辑。具体的"先写哪个、写完怎么验证"执行顺序见 `docs/BUILD_PLAN.md`,不在本节重复。

## 9. 完成状态总表(诚实版)

### 9.1 已完成(真代码,至少跑通过一次)

| 组件 | 文件 | 验证方式 |
|---|---|---|
| RAG 格式转换 + 切块管线 | `planning/step1-naive-rag/ingest.py` | 对全部真实知识源(JSON/JSONL/XML/MD/PDF,含 OCR 文本)跑通,产出 `knowledge/_processed/{tcm,nutrition}_chunks.jsonl` |
| BM25 baseline + recall 评测 | `planning/step1-naive-rag/{naive_rag.py,eval_recall.py,build_and_eval_bm25.py}` | 产出了 D23/D24 决策依据的真实 recall@5 数字 |
| `knowledge_chunks` 表结构 | `db/schema.sql` | SQL 已 review,**未在真实 Postgres 实例上执行过**(项目目前没有已部署的 Postgres) |
| `recipes` 表结构 | `db/schema.sql` | 同上 |
| recipes 灌库脚本 | `db/load_recipes.py` | 对 2000 条真实数据 dry-run 过解析逻辑,未连真实 DB |
| BGE-M3 → pgvector 脚本 | `db/embed_bge_m3.py` | 已写完 `load`/`search` 两个子命令,**未执行过**——需要用户在有网络、装好 `FlagEmbedding`/`torch` 的机器上,配合一个真实 Postgres 实例才能跑 |
| 冲突规则表内容 | `evals/conflict_rules.jsonl` | 40 条(18 verified / 22 needs_source),未进数据库 |

### 9.2 有完整设计,零代码

| 组件 | 设计位置 |
|---|---|
| MCP server(全部 6 个工具) | 本文档 §2 |
| Agent Loop / 中枢 agent 路由编排 | 本文档 §3、§5.1 |
| TCM SubAgent / Nutrition SubAgent | 本文档 §5.2 步骤 3 |
| 调和层 | 本文档 §5.2 步骤 5,`DECISIONS.md` D14 |
| 核查 pass | 本文档 §5.2 步骤 6,`DECISIONS.md` D15 |
| `write_memory` / 关键事实落库前置扫描 | 本文档 §4.2、§4.3 |
| 分层压缩组件 | 本文档 §4.4,`DECISIONS.md` D8 |
| `user_profile`(含 `preferences` 字段,D25)/ `diet_log` / `conversation_sessions` / `dish_ingredient_map` 表 schema | 本文档 §1.2 |
| `conflict_rules` 从 JSONL 迁移进关系表 | 本文档 §1.2 |
| "记录回顾""候选评估"两条新分支的具体实现(D25) | 本文档 §5.1、§5.3 |
| 3 份 Agent Skills 文件(其中 reconciliation_rubric/verification_checklist 各多一条 D25 新增内容待写)+ `skills/registry.py`(D27 加载机制) | 本文档 §6 |
| 分层压缩的压缩优先级表 + 结构化归档摘要模板(D27) | 本文档 §4.4,`DECISIONS.md` D27 |
| SubAgent 循环状态提示(D27) | 本文档 §4.5,`DECISIONS.md` D27 |
| `user_profile` 体质字段扩展(`constitution_secondary`/`constitution_source`/`constitution_confirmed_at`,D28) | 本文档 §1.2、§11.3,`DECISIONS.md` D28 |
| 首次使用引导对话流程 + CCMQ 计分逻辑(含体质夹杂判定,D28) | 本文档 §11 |
| FastAPI 路由/schema/SSE 事件设计(D26) | 本文档 §10,`DECISIONS.md` D26 |
| `user_dish_aliases` 表 + 三级查找/晋升逻辑(程序性记忆,D27 修订一) | 本文档 §1.2、§4.2,`DECISIONS.md` D27 修订一 |
| 压缩触发时机(SubAgent 内同步 + 中枢异步/同步兜底两级,D27 修订二) | 本文档 §4.4.1,`DECISIONS.md` D27 修订二 |
| 2 份新增 Skill 文件(CCMQ 问卷、ED 响应话术,D22 补充) | 本文档 §6.1 |
| 目录结构(§8.1)与逐文件测试覆盖表(§8.2) | 本文档 §8 |
| LLM adapter 层(超时/重试/熔断/双档切换) | 本文档 §7,`ENGINEERING.md` §1 |
| Guardrails(输入防护/输出拦截/ED 防护/循环防护) | `PRD.md` §10 |
| `evals/dataset.jsonl`(E1/E2a/E2b/E3,≥40 条)+ `smoke.jsonl` | `PRD.md` §8.1,`planning/roadmap.md` 阶段 3——**目前 `evals/` 下只有 `conflict_rules.jsonl` 和 `reference_tables/`,组装好的测试集本身还不存在** |
| `EVALUATION.md` / `THREAT_MODEL.md` | `PRD.md` §15 交付物清单,两份文件均未创建 |
| Langfuse 可观测接入 | `ENGINEERING.md` §6 |
| 前端(Next.js) | `PRD.md` §7 |
| `docker compose up` 一键部署 | `ENGINEERING.md` §9 |

### 9.3 与 `planning/roadmap.md` 的对应关系

roadmap 是"按什么顺序做",本文档是"做出来长什么样"。两者不重复,交叉引用:

| roadmap 阶段 | 对应本文档章节 |
|---|---|
| 阶段 4.2 任务 1(adapter 层) | §7 |
| 阶段 4.2 任务 2(Postgres schema + ingest) | §1 |
| 阶段 4.2 任务 3(`query_diet_log` 工具) | §2.2、§4.1 |
| 阶段 4.2 任务 4(中枢 agent + Agent Loop) | §3、§5.1 |
| 阶段 4.2 任务 5(分级路由) | §5.1 |
| 阶段 4.2 任务 6(两个 SubAgent) | §5.2 步骤 3 |
| 阶段 4.2 任务 7(调和层) | §5.2 步骤 5 |
| 阶段 4.2 任务 8(核查 pass) | §5.2 步骤 6 |
| 阶段 4.2 任务 9(流式输出) | §5.2 步骤 8 |
| 阶段 5(安全层) | §5.4 |
| 阶段 7(分层压缩) | §4.4 |

---

## 10. API 层设计:FastAPI(D26)

D26 把 FastAPI 定为唯一后端 Web 框架,本节给出具体路由与 schema——此前 §8 目录结构里的 `api/main.py` 只是一个占位注释,没有具体设计。

### 10.1 路由清单

| 方法 + 路径 | 用途 | 请求体(Pydantic) | 响应 |
|---|---|---|---|
| `POST /api/chat` | 主入口,承载六条分支(D12/D25)的全部对话交互 | `ChatRequest{session_id, message}` | SSE 流(`text/event-stream`),事件类型见 §10.3 |
| `GET /api/sessions/{session_id}/messages` | 拉取会话历史(前端刷新/重连时用) | — | `MessageList{messages: [...]}` |
| `GET /api/profile` | 读取当前 `user_profile` | — | `UserProfile{...}`(含 D25 `preferences`、D28 体质四字段) |
| `PATCH /api/profile` | 更新画像字段,承载人在环确认后的写入(PRD §10.2) | `ProfileUpdate{field, value, confirmed: true}` | `UserProfile{...}` |
| `POST /api/onboarding/start` | 触发首次使用引导(§11) | `OnboardingStart{}` | `OnboardingStep{...}`,见 §11.2 |
| `POST /api/onboarding/answer` | 提交引导对话中的一轮回答(含 CCMQ 问卷分批作答) | `OnboardingAnswer{step_id, answer}` | `OnboardingStep{...}` 或 `OnboardingResult{constitution, constitution_secondary, ...}` |

**没有单独的"记录"或"记录回顾"REST 端点**:D25 的六条分支全部是自然语言输入,统一走 `POST /api/chat`,由中枢 agent 做路由判断(§5.1)——单独开 `/api/diet-log` 这类端点等于在 API 层重新做一遍路由,和"路由是中枢 agent 的职责"这个既有设计矛盾,不做。

### 10.2 与 D1/D7/D19 的接口关系

`/api/chat` 处理函数内部的调用链:输入防护(§0①)→ 关键事实扫描(§0②,同步调用 `write_memory`,走 MCP client)→ 路由判断(D12/D25)→(视分支)`asyncio.gather` 并行派发两个 SubAgent(D1,通过 MCP client 调工具)→ 调和层/核查 pass(D14/D15,经 LLM adapter,D19)→ SSE 逐块吐出。整条链路是一条 `await` 链,不引入额外线程池,这是 D26 选择 FastAPI 的理由一(原生异步)的具体落地。

### 10.3 SSE 事件设计(服务 M10 首字节延迟)

```
event: token       data: {"text": "..."}              # 生成内容的增量片段
event: source      data: {"source_id": "...", ...}     # 溯源信息,前端渲染可展开引用(§5.2 步骤8)
event: guardrail   data: {"type": "...", "detail": "..."}  # 命中的拦截/降级(§5.4)
event: done        data: {"trace_id": "..."}           # 结束,携带 trace_id 供前端后续查询
```

首字节延迟(M10)对应第一条 `token` 事件的发出时刻;核查 pass(D15,只拒绝不改写)必须在第一条 `token` 事件发出**之前**完成——先核查后流式输出,不能边流式吐边核查,否则"只拒绝不改写"这条规则在流式场景下无法实施(已经吐出去的 token 撤不回来)。这是 SSE 设计对既有 D15 决策的一个具体约束,写在这里防止实现时顺手做成"边生成边核查"。

### 10.4 状态

⏳ **完全未实现。** 路由清单、Pydantic schema、SSE 事件设计均为规划,`api/main.py` 尚不存在。

---

## 11. 首次使用引导与体质获取对话流程(D28)

D28 决定了"渐进式引导 + 体质主/次/来源字段",本节给出具体对话步骤,供实现 `POST /api/onboarding/*`(§10.1)时对照。

### 11.1 触发条件

`user_profile` 不存在,或存在但核心字段(过敏原、体质)均为空,且当前是本次会话的第一轮——满足即由中枢在给出第一条回复前,先插入引导开场白,而不是等用户问完事实性问题后再打断。

### 11.2 对话步骤

```
0. 开场白:说明会先了解基本情况(过敏原/体质/口味),几个问题,随时可跳过,跳过不影响基础功能
1. 收集过敏原/禁忌(开放式提问,命中关键词即结构化——复用 §4.2 步骤 1/2 的确定性抽取逻辑)
2. 收集口味与情境偏好(开放式,写入 preferences,D25)
3. 体质:
   3a. 反问"你知道自己的中医体质类型吗?"
       ├─ 已知 → 询问具体是哪一类(九分类之一)
       │         → 写入 constitution + constitution_source="self_reported"
       │         → 人在环确认(PRD §10.2,自述也要过一遍确认展示)
       │         → 跳到步骤 4
       └─ 不知道/不确定 → 3b
   3b. 对话式 CCMQ 简版问卷,每轮 2-3 题,允许"不确定"选项,分批收集完整题目
   3c. 计分(见 §11.3)→ 呈现主/次体质结果
   3d. 人在环确认:确认 / 修改 / "不太准,先跳过"
       → 写入 constitution(主)+ constitution_secondary(次,可空)
         + constitution_source="ccmq_computed" + constitution_confirmed_at
4. 完成,进入正常问答。任何一步被跳过都不阻塞后续使用,体质/过敏原可以在之后任意一次会话里
   通过"我想重新测一下体质"这类主动请求再次触发步骤 3
```

### 11.3 CCMQ 计分与"体质夹杂"

CCMQ 简版每题 5 级李克特量表,按九类体质(平和/气虚/阳虚/阴虚/痰湿/湿热/血瘀/气郁/特禀)分别计算转化分(0-100)。判定规则:转化分 ≥ 40 记为"是"(可作为 `constitution` 主体质候选),30-40 记为"倾向是"(进入 `constitution_secondary`)。**允许同时命中多个体质**——中医"体质夹杂"是常见且有临床意义的结果,不是计分异常,只取最高分会让 TCM SubAgent 拿到的信息比问卷实际结果更贫乏(这一点在 D28 里已有完整论证)。

**体质未知/未确认时的降级行为**(呼应 §5.2 步骤 3 新增的说明):`constitution` 为空,不代表 TCM SubAgent 不可用——它的建议范围收窄为体质无关的普适性温和建议,并引导用户完善信息,不报错、不套用默认体质(套用错误体质的风险高于"不知道")。

### 11.4 状态

⏳ **完全未实现。** 对话步骤、CCMQ 计分逻辑、`constitution_secondary` 相关的 schema 变更(§1.2)均为规划,`db/schema.sql` 尚未包含 D28 新增字段,CCMQ 简版量表的具体题目/计分系数表尚未整理成可执行的查表结构(现状与 D5 相同——量表来源已确定,但结构化数据文件本身还没做)。

---

## 变更日志

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-08-26 | v0.5 | §2.1 明确点名现有检索层已经是 agentic RAG 形态(D20 行为点#1);新增 §2.6:MQE/HyDE 检索增强设计,明确定位为工具内部增强、非新增 agent 决策点,且明确排后(需先有 recall 数字支撑,阶段8前后再评估);新增 `docs/BUILD_PLAN.md`(roadmap 阶段顺序 × §8.1/8.2 文件与测试映射的执行清单);按 §8.1 建出 `backend/`/`api/`/`tests/`/`db/migrations/`/`frontend/` 骨架目录与占位文件 |
| 2026-08-26 | v0.1 | 首个版本。整合 `PRD.md` §7/§10/§12、`DECISIONS.md` D1-D24、`ENGINEERING.md`、`RAG_PIPELINE_DESIGN.md` 与已写出的 `ingest.py`/`db/*.py` 实际状态;新增两处此前未落定的设计(§4.3 关键事实跨分支扫描、`dish_ingredient_map` 查表优先的菜品拆解),标注为本文档新增而非既有决策 |
| 2026-08-26 | v0.3 | 新增 D26(FastAPI,§10 API 层设计)、D27(记忆架构升级:§4.1.1 认知类型/存储格式反向验证、§4.4 压缩优先级表与结构化归档摘要、§4.5 SubAgent 循环状态提示)、D28(§11 首次使用引导与体质获取对话流程,§1.2 体质字段扩展);§6 补充 Skills 三层加载机制的具体设计(此前只有文件清单,没有加载机制) |
| 2026-08-26 | v0.4 | 整合 D27 修订一/二:程序性记忆重新纳入范围(§4.1.1、§4.2 三级查找/晋升逻辑、`user_dish_aliases` 表 §1.2)、压缩触发时机具体化为两级机制(§4.4.1);整合 D22 补充:新增 `ccmq_questionnaire.md`/`ed_risk_response.md` 两份 Skill(§6.1);§8 从占位目录树扩为完整目录结构 + 逐文件测试覆盖表(§8.1/8.2) |
| 2026-08-26 | v0.2 | 按用户交互场景走查后随 `DECISIONS.md` D25 同步更新:路由从四条扩到六条(§0/§5.1/§5.3,新增"记录回顾""候选评估"及其判定标准);`user_profile` 新增 `preferences` 字段并与 `goal_tags` 明确区分(§1.2);`query_diet_log` 补充相对日期与时区处理提醒、确认候选评估分支不需要新工具(§2.2);SubAgent 任务上下文明确携带用户原始提问文本(§5.2 步骤 3);两份 Skill 文件补充 harm-reduction 语气原则与候选评估核查规则(§6) |
