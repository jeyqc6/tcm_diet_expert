# 02 · 项目路径

---

## 一、核心原则：学习和项目并行，不串行

```
学习模块   A ────── B ────── C ────── D
              ↘        ↘        ↘        ↘
项目层      骨架     RAG层    安全层    观测层
```

**不要「学完再做」**，那样两周只够学。每学完一个模块，立刻落到项目对应的一层。

---

## 二、架构：偷五层，换内容

同构但换域 —— 这是最好的面试故事。

| 层 | MewCode（Coding Agent） | 你的（示例：设计规范合规检查） |
|---|---|---|
| 1 交互 | TUI 终端 | Next.js Web + 流式响应 |
| 2 引擎 | Agent Loop + SubAgent | Agent Loop + **检索路由** |
| 3 工具 | ReadFile / Bash / Grep | 规范检索 / 图纸解析 / 条款比对 / 报告生成 |
| 4 记忆 | 上下文压缩 + 跨会话 | 同左（**这层几乎可以直接搬**） |
| 5 安全 | 权限 + Worktree 隔离 | 输入防护 + 工具白名单 + **强制引用溯源** |
| **+ RAG** | ❌ MewCode 没有 | ✅ **这是你的增量** |
| **+ 观测** | ❌ MewCode 没有 | ✅ Langfuse 全链路 trace |

> ⚠️ 上表的领域只是**示例**，用来说明「合格的选题长什么样」，不是替你选题。

---

## 三、⭐ 选题的终极检验：verify 信号

这个「同构换域」的做法会逼你回答一个真正难的问题：

> **Coding agent 有天然的 verify 信号 —— 跑测试。
> 你的领域里，verify 信号是什么？**

- **答得上来** → 这是你的原创贡献，面试里能讲 20 分钟
- **答不上来** → 选题有问题，趁早换

**现在就拿这个问题去筛你的候选选题。**

### 四条选题标准

1. **数据源是不是你能拿到、别人拿不到的？** → 决定是不是「又一个 chat with PDF」
2. **任务是不是必须多步？** → 单步能解决就不需要 agent。**如果一个 workflow 能解决，你的 agent 就是过度设计**，面试官会当场戳穿
3. **做错了有没有代价？** → 有代价才需要 guardrail、才需要 eval、才有「工业级」可讲
4. **和 Computational Design 背景搭不搭？** → 这是你区别于 CS 科班的地方

---

## 四、两周日历

### Week 1 · 学 + 骨架

| 天 | 任务 | 产出 |
|---|---|---|
| D1-2 | 模块 A（6h） | 选题定稿、PRD v1 |
| D3-4 | 模块 B 前半（5h） | RAG 跑通，文档能进能查 |
| D5 | 模块 B 后半（3h） | **eval 集 + 第一次跑分**（难看是正常的） |
| D6-7 | 搭骨架 | FastAPI + Postgres/pgvector + Next.js + Docker Compose 一键起 |

### Week 2 · 深度 + 收口

| 天 | 任务 | 产出 |
|---|---|---|
| D8-9 | 模块 C（6h） | 安全层落地 + `THREAT_MODEL.md` |
| D10 | 模块 D 的 Langfuse（3h） | 全链路 trace 打通 |
| D11-12 | **自己手写一个核心部件** | 推荐上下文压缩（你在 SharedState 上摸过边，能做得比别人深） |
| D13 | 重跑 eval | **「优化前 vs 优化后」对比数字** ← 这就是简历 bullet |
| D14 | 收口 | 英文 README + 架构图 + 2 分钟录屏 + 部署 |

MewCode 阅读穿插在 D1-2、D8-9、D11-12，每次 2-3 小时，只读对应层。

---

## 五、必须交付 vs 可以砍掉

### ❌ 不能砍（这就是「工业级」的全部含义）

- [ ] **eval 集**（≥30 条）+ 跑出来的分数
- [ ] **引用溯源** —— 每个答案能指回原文哪一段。这几乎是 RAG 项目「有没有认真做」的唯一外部可见信号
- [ ] **一个自己实现的核心部件**（上下文压缩 / 路由 / 记忆，选一个）
- [ ] **可复现部署** —— `docker compose up` 一键起
- [ ] **英文 README + 架构图**

### ✅ 可以砍

- 漂亮 UI（能用就行）
- 多 agent 协作 —— **单 agent + 好的工具设计 > 花哨的多 agent**。面试官更爱听你解释「我为什么**没有**上多 agent」
- MCP（除非场景真需要接外部工具）
- 用户测试

---

## 六、简历怎么写

简历上一个项目只有 3-4 个 bullet，招聘方看 30 秒。所以你真正要产出的不是「功能很多」，而是**3 句能被追问 20 分钟的话**。

**好的 bullet（有数字、有取舍、可追问）**
> Designed a token-aware context compaction strategy for long-running agent tasks, cutting context overflow from 34% to 6% while preserving 95% of task-critical state.

> Built a 30-case evaluation harness with citation-grounding checks; iterating on chunking strategy raised retrieval precision from 61% to 88%.

> Implemented a 4-tier permission gate and sandboxed tool execution; documented threat model covering 7 escape scenarios.

**差的 bullet（无数字、无取舍、一问就空）**
> Built a multi-agent system using LangChain and RAG.

---

## 七、成本控制

| 服务 | 方案 | 成本 |
|---|---|---|
| 前端 | Vercel | 免费 |
| 后端 | Render / Railway free tier | 免费 |
| 数据库 | Supabase / Neon free tier（含 pgvector） | 免费 |
| 观测 | Langfuse 自托管 | 免费 |
| LLM API | Claude / OpenAI | ~$10-20 |

**唯一的风险是 live demo 被刷爆。** 三个对策：

1. Demo 模式只允许预置数据集 + IP 限流
2. Bring-your-own-key
3. **2 分钟录屏 + 一键自建说明**（推荐）

> 很多资深工程师的 portfolio 就是录屏 + 好 README。
> **没人会因为你没挂 live demo 扣分，但会因为你的 demo 挂了扣分。**

---

## 八、目录结构建议

```
project-root/
├─ README.md               ← 英文，30 秒能读懂
├─ CLAUDE.md               ← vibe coding 项目指令
├─ docker-compose.yml      ← 一键起
├─ docs/
│  ├─ PRD.md               ← ⭐ 单独成文件，PM 方向的作品
│  ├─ ARCHITECTURE.md      ← 五层架构图
│  ├─ THREAT_MODEL.md      ← ⭐ 面试杀器
│  ├─ EVALUATION.md        ← 指标定义 + 跑分结果
│  ├─ DECISIONS.md         ← 为什么没用多 agent、为什么选这个 chunking
│  └─ zh/                  ← 中文版
├─ backend/
│  ├─ agent/
│  │  ├─ loop.py           ← ⭐ 自己写
│  │  ├─ router.py
│  │  └─ compaction.py     ← ⭐ 自己写，核心亮点
│  ├─ rag/
│  │  ├─ ingest.py
│  │  ├─ retrieve.py
│  │  └─ citation.py       ← 溯源
│  ├─ guardrails/
│  │  ├─ permissions.py
│  │  └─ sandbox.py
│  └─ main.py
├─ frontend/
├─ evals/
│  ├─ dataset.jsonl        ← ⭐ 30-50 条
│  └─ run_eval.py
└─ tests/
```

标 ⭐ 的是面试时会被重点看的文件。
