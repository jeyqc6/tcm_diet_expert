# 01 · 学习路线

---

## 一、先做体检：用五层架构看自己缺什么

MewCode 提出的五层架构是个好用的心智模型，而且**可以搬到任何 agent 上**，不只是 coding agent。用它当体检表：

```
┌──────────────────────────────────────────┐
│ 1 交互层  TUI / Slash Command / Skill / 对话管理  │  ← 门面
├──────────────────────────────────────────┤
│ 2 引擎层  Agent Loop / LLM Client / SubAgent    │  ← 大脑
├──────────────────────────────────────────┤
│ 3 工具层  Core Tools / MCP / Hook / Registry    │  ← 手
├──────────────────────────────────────────┤
│ 4 记忆层  Context 管理 / 自动压缩 / 长期记忆      │  ← 地基
├──────────────────────────────────────────┤
│ 5 安全层  权限系统 / Worktree 隔离 / 工具过滤     │  ← 地基
└──────────────────────────────────────────┘
```

### 缺口分析

| 层 | 已有基础 | 状态 | 缺什么 |
|---|---|---|---|
| **1 交互层** | 流式响应、多轮对话、WebSocket、前端 dashboard | ✅ 基本无缺口 | Skill / Slash Command 这类「约定式扩展」的设计思路 |
| **2 引擎层** | Autogen 多智能体、路由逻辑、结构化输出 | 🟡 用过没写过 | 从零实现 agent loop；终止条件的协议层判断 |
| **3 工具层** | GPT-4 function calling；flag injection | 🟡 会调不会设计 | MCP 协议；工具注册与权限元数据设计 |
| **4 记忆层** | **去重 SharedState + 多轮记忆优化** | 🟡 最接近的一块 | token-aware 的自动压缩机制 |
| **5 安全层** | — | ❌ **完全空白** | 全部 |
| **RAG**（MewCode 不含） | — | ❌ **完全空白** | 全部 |
| **Evals** | **已做过 automated evaluation + synthetic test generation** | 🟡 做过但未系统化 | 形式化的指标体系、Ragas 这类框架 |
| **可观测性** | — | ❌ 空白 | trace / 成本追踪 |

### 两个结论

**第一，缺口比想象的小。** 真正从零开始的只有 **RAG** 和 **安全/沙箱** 两块。其余都有真实工程经验垫底，只是没系统化。**约 28 小时可补齐，不是 200 小时。**

**第二，差异化坐标已经清楚：安全层 + eval。**
这两块是「你既缺、又稀缺、又能讲深」的交集。大部分求职项目在这两块是空的。

---

## 二、四个学习模块（约 28 小时）

> **规则：每个模块必须有产出物。没有产出物的学习等于没学。**

### 模块 A · Agent 内核（6h）—— 先建立判断力

| 资源 | 时长 | 拿走什么 |
|---|---|---|
| Anthropic《Building Effective Agents》 | 1h | workflow vs agent 的边界。**读完要能回答「我的场景为什么必须用 agent」** |
| Anthropic《Building agents with the Claude Agent SDK》 | 1h | `gather context → take action → verify work → repeat`。**注意 verify 那一环** |
| MewCode 第 1-2 层理论 + 任一语言的 Agent Loop 源码 | 3h | loop 的终止条件（靠 tool_use 的有无判断，而非硬编码）、事件流设计 |
| 微软 AI Agents for Beginners 第 7、9 课 | 1h | Planning / 元认知自纠错 |

**产出物**：一页《什么情况下不该用 agent》，存进 `docs/`。这页后面直接变成 PRD 第 2 节。

---

### 模块 B · RAG（8h）—— 唯一真正的零基础区

| 资源 | 时长 | 拿走什么 |
|---|---|---|
| Anthropic《Contextual Retrieval》 | 1h | 为什么朴素 chunking 会烂、怎么修。**RAG 只读一篇就读它** |
| Chroma 的 chunking 评测技术报告 | 1h | 分块策略的实证对比，不是拍脑袋 |
| 动手：pgvector + 自己的一批文档 | 3h | 端到端跑通一次 |
| **Ragas** 文档 + 实跑一次 | 2h | context precision / recall / faithfulness / answer relevancy |
| RAGFlow 的 citation grounding 源码 | 1h | 引用溯源怎么实现 —— **这是「工业级」和「demo 级」的分界线** |

**产出物**：**一个 30-50 条的 eval 集 + 第一次跑分结果。**
👉 这是整个计划里**最重要的单一产出物**。第一次分数难看是正常的。

---

### 模块 C · 安全层 / 沙箱（6h）—— 差异化所在

| 资源 | 时长 | 拿走什么 |
|---|---|---|
| **OWASP Top 10 for LLM Applications** | 1.5h | guardrail 领域的行业标准清单，免费。你的防护设计要能对上它的条目 |
| MewCode 第 5 层理论 + 源码 | 2h | 5 层分级权限的具体实现（这块是它的强项） |
| **E2B**（开源代码沙箱）源码/文档 | 1.5h | 真沙箱怎么做：进程隔离、文件系统边界、超时与资源限额 |
| Docker seccomp / 只读挂载 / 网络隔离 实操 | 1h | 在已会的 Docker 上加安全约束 |

**产出物**：`docs/THREAT_MODEL.md` —— 列出 agent 的危险动作、每个动作的拦截层级、逃逸场景。
👉 **这份文档在面试里的杀伤力超过大部分代码。**

---

### 模块 D · 可观测 + Eval 工程化（8h）—— 升级已有经验

| 资源 | 时长 | 拿走什么 |
|---|---|---|
| 吴恩达 Agentic AI 的 evals / error analysis 章节 | 2h | 决定 agent 项目成败的最关键因素就是评测和错误分析流程 |
| **Langfuse**（开源、可自托管、免费）Docker 自建 | 3h | trace 每次 LLM 调用、tool 调用、token 消耗。**同时满足「全栈 + 数据库 + Docker」的诉求** |
| LLM-as-Judge 的做法与陷阱 | 1h | judge 本身怎么验证 |
| MewCode 第 4 层（上下文压缩）理论 | 2h | token 预算、压缩策略、「刚注入的长期记忆下一步就被压缩掉」这类真实 bug |

**产出物**：Langfuse 跑起来，能看到完整调用链和成本。截图进 README。

---

## 三、⭐ 优先读：李博杰《深入理解 AI Agent》（免费，2026-07-28 发现）

全书 10 章 + 92 个可运行配套实验，Apache 2.0 完全开源。
**覆盖范围事实上顶替了 MewCode 的大部分价值，还补齐了 MewCode 完全没有的 RAG 和 Eval。**

| 章节 | 内容 | 对应模块 | 优先级 |
|---|---|---|---|
| 第 1 章 · Agent 基础 | Agent = LLM + 上下文 + 工具；harness 工程是竞争力 | 模块 A | 🔴 必读 |
| 第 2 章 · 上下文工程 | KV Cache、提示工程、Agent Skills、**上下文压缩** | 模块 A/D | 🔴 必读 —— **最贴近你要手写的那个核心部件** |
| 第 3 章 · 用户记忆和知识库 | 记忆、**RAG**、结构化索引、知识图谱 | 模块 B | 🔴 必读 —— **MewCode 完全没有的一块** |
| 第 4 章 · 工具 | MCP 协议、工具三分类、异步 agent | 模块 A | 🟠 推荐 |
| 第 5 章 · Coding Agent 全景 | 生产级 coding agent，**待确认是否覆盖权限/沙箱** | 模块 C（待验证） | 🟠 推荐，读完据此决定是否还买 MewCode |
| 第 6 章 · Agent 的评估 | 评估环境、指标、**统计显著性** | 模块 D | 🔴 必读 —— **MewCode 完全没有，且比大部分免费教程严谨** |
| 第 10 章 · 多 Agent 协作 | 协作架构、失败模式、Agent Society | 选题验证 | 🟢 可选，验证"为什么不上多 agent" |
| 第 7、8、9 章 | 模型后训练、持续进化、多模态/实时交互 | — | ⛔ 跳过（与目标无关；语音部分你已有实战经验） |

**预计投入：6-8 小时（第 1/2/3/4/6 章）。**

---

## 四、MewCode 使用说明（读完上面这本书之后再决定要不要买）

**关键问题：第 5 章有没有把安全层/权限/沙箱讲透？**
- 讲透了 → MewCode 剩余价值只剩"SubAgent/Worktree/Teams 的取舍"，238 元买一个点，性价比存疑，可以不买
- 没讲透 → 按下表使用，**总预算压缩到 4-5 小时**（原 8-10 小时，因为大头已被免费书覆盖）

| 何时读 | 读什么 | 时长 |
|---|---|---|
| 模块 C 期间 | 第 5 层：权限系统、工具过滤 | 2h |
| 有余力时 | SubAgent / Worktree / Agent Teams 的取舍 | 1-2h |
| **永远不读** | 面试题、简历写法、Vibe Coding 跟做步骤、第 1/2/4 层（书里已覆盖） | 0h |

**语言版本选 Java**（你的强项），源码读起来最快。

---

## 四、覆盖度对照：MewCode vs 免费资源

| 知识点 | MewCode | 免费替代 | 谁赢 |
|---|---|---|---|
| Agent Loop | ✅ 详解+源码 | HKUDS/nanobot + Anthropic blog | 打平 |
| 工具系统设计 | ✅ | nanobot + Claude Agent SDK 文档 | 打平 |
| MCP 协议 | ✅ | MCP 官方 spec | 免费更好 |
| **上下文压缩** | ✅ 系统讲解 | Claude Code 泄漏分析（碎片） | **MewCode** |
| 权限 / 沙箱 | ✅ 5 层设计 | E2B 源码 + OWASP | 打平（都看） |
| 跨会话记忆 | ✅ | nanobot memory 模块 | 打平 |
| **多 Agent 三种模式取舍** | ✅ | Anthropic 文章（较浅） | **MewCode** |
| RAG | ❌ **无** | Contextual Retrieval + Ragas | 免费 |
| Evals | ❌ **无** | 吴恩达课 + Ragas | 免费 |
| 可观测性 | ❌ **无** | Langfuse | 免费 |

**一句话**：MewCode 买的不是独家知识，是「整理好」。省 5 小时筛选就回本。但要警惕它偷走你做项目的时间。
