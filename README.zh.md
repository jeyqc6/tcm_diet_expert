# Diet Expert · 项目索引

**English overview / architecture / quick start：见 [README.md](./README.md)。本文件是给项目维护者/施工用的中文索引，链接到下面这些中文设计文档。**

中西医结合的个人饮食管家 agent。

**怎么跑起来**（Docker 一键起 / 本地开发 / 改代码后怎么重部署）见 [RUN.md](./RUN.md)。

**当前处于阶段 4–9 均已落地，阶段 7/8/9 是 Partial**。五段管线、MCP 工具、SSE `/api/chat`、前端最小闭环、guardrails、Langfuse、首次引导、会话落库均已有真代码。**阶段 7 是 Partial，不是 Done**：会话压缩、菜品拆解、Level 1 FIFO 检索压缩、关键事实 pending 确认均已接；独立 3 日菜谱组装仍未做。**阶段 8 是 Partial**：B0/双档 B1/B3/切片/三档阈值已跑(M1 仍 53.3%，未达 Launch 70%，见 `docs/EVALUATION.md` §7)；B2 ablation(单 agent + 双库同上下文 vs 当前双 SubAgent+调和层)跑了两轮——keyword rubric 版 B2 三项全胜，但抽查发现 rubric 对同义词不宽容，加做 LLM-as-judge 复核后**内容质量其实几乎打平**(7.93/8 vs 8.00/8)，13.3 个百分点的原始差距主要是字面用词导致；**延迟(203.2s vs 414.3s)/LLM 调用次数(33 vs 78)的差距不受影响**，B2 仍是不到一半成本打平质量。`docs/DECISIONS.md` D1 已追加两版修订记录，但**是否把生产架构真的换成单 agent 尚未执行**，留待人决定范围。**阶段 9 是 Partial**：英文 `README.md`(本文件的英文版)+ Mermaid 架构图已完成；2 分钟英文录屏、Vercel+Render+Neon 云部署均未做，`README.md` 里如实标注。

**最后更新** 2026-08-30

---

## 一、文件结构

```
diet_expert/
│
├─ README.md                       英文总览(项目介绍/架构图/怎么跑/怎么部署)
├─ README.zh.md                    ← 本文件，中文项目索引
├─ RUN.md                          ← 怎么把服务跑起来(中文，更细)
│
├─ docs/                           ★ 设计与交付文档
│  ├─ PRD.md                       产品需求
│  ├─ ARCHITECTURE.md              实现设计(长什么样)
│  ├─ DECISIONS.md                 设计决策记录
│  ├─ ENGINEERING.md               后端工程化:可靠性/并发/测试/CI
│  ├─ ASYNC_DESIGN.md              Agent async/sync 分层、并发点、已知债与改进工时
│  ├─ BUILD_PLAN.md                分阶段实施清单(诚实状态)
│  ├─ EVALUATION.md                指标与 baseline 跑分
│  ├─ THREAT_MODEL.md              逃逸场景与控制现状
│  ├─ LANGFUSE.md                  可观测接入说明
│  └─ prompts/                     system prompt 快照 + 免责声明模板
│
├─ api/                            FastAPI：/api/chat · /api/profile · /api/onboarding · /api/sessions
├─ backend/                        五段管线、MCP 工具、记忆、guardrails、LLM adapter
├─ frontend/                       Next.js 最小聊天页(SSE)
├─ db/                             schema.sql · ingest / embed / load_conflict_rules
│
├─ knowledge/                      ★ 知识库源数据(派生 jsonl 在 .gitignore)
├─ evals/                          测试集(dataset/smoke) · conflict_rules · run_baselines.py
├─ tests/                          单元 + 集成(mock LLM / replay，不烧 token)
│
├─ docker-compose.yml
└─ .github/workflows/ci.yml        lint → pytest → smoke-eval(缺 knowledge 派生文件时会跳过)
```

更细的目录树见 `docs/ARCHITECTURE.md` §8.1。

---

## 二、怎么读这些文档

| 文件 | 回答的问题 |
|---|---|
| `PRD.md` | 要做什么、为什么做 |
| `ARCHITECTURE.md` | 具体长什么样(签名/表/请求生命周期) |
| `DECISIONS.md` | 为什么选 A 不选 B |
| `ENGINEERING.md` | 超时/降级/测试/CI 怎么落地 |
| `ASYNC_DESIGN.md` | Agent 哪层 async、哪层 sync、并发与改进路线 |
| `BUILD_PLAN.md` | 这一阶段写哪个文件、用哪个测试验收 |
| `EVALUATION.md` | baseline 数字与阈值 |
| `THREAT_MODEL.md` | 会怎样给出有害建议、现在拦不拦得住 |
| `RUN.md` | clone 下来怎么跑 |

---

## 三、当前欠账(按优先级)

对照代码现状，不是对照设计愿望。阶段 8(B2 ablation / 全量 eval)不在本表。

| # | 事项 | 说明 |
|---|---|---|
| 1 | **独立 3 日菜谱组装管线未做** | Nutrition 白名单已含 `query_recipes_by_ingredients`；完整推荐且用户要菜谱/购物清单时加载 Skill。没有单独的「步骤 7 拼 3 日计划」阶段 |
| 2 | **两侧 SubAgent 都失败没有 naive RAG** | 当前是 `both_subagents_failed` guardrail，没有另建 RAG 降级管线 |
| 3 | **空检索没有静态兜底表** | 证据修复可保留并标注通用知识；不假装有 fallback 表或伪造引用 |
| 4 | **B1/B3 / B2 ablation** | 属阶段 8。当前 B0 全量 recall@5=53.3%，未达 Launch 70%。CI smoke 用 fixture，门槛 0.6，不是 Launch 数字 |

---

## 四、谁看哪些文件

| 文件 | 读者 |
|---|---|
| `docs/` | 面试官、招聘方、实现对照 |
| `knowledge/`、`evals/` | 施工用,也是面试时能打开的实物 |
| `planning/` | 自用工作文件；不要把红队预演当对外交付 |
