# Diet Expert · 项目索引

中西医结合的个人饮食管家 agent。**当前处于阶段 1**(判断力 + 知识底座),尚未开始写代码。

**最后更新** 2026-08-01

---

## 一、文件结构

```
diet_expert/
│
├─ README.md                       ← 本文件
│
├─ docs/                           ★ 交付物,未来直接进 repo
│  ├─ PRD.md                       产品需求(生产版,给别人看)
│  ├─ DECISIONS.md                 D1-D21 设计决策记录
│  ├─ ENGINEERING.md               ⭐ 后端工程化:可靠性/并发/测试/CI
│  └─ prompts/                     system prompt 快照 + 免责声明模板
│
├─ knowledge/                      ★ 知识库源数据
│  ├─ 00-SOURCES.md                已核实并落地的来源
│  ├─ 01-DOWNLOAD-LIST.md          ⭐ 待下载清单(按类别 + 入库形态 + 下载顺序)
│  ├─ tcm/core-tables.md           体质九分类 · 六淫 · 药食同源 · 五行对应
│  ├─ nutrition/evidence-notes.md  烹饪维生素 · 补剂交互 · 矿物质拮抗 · 精准营养
│  ├─ allergen/                    ⭐ 中餐隐藏过敏原 22 条(Critical 档)
│  ├─ food/                        ⭐ 菜品拆解表 44 道
│  └─ _raw/                        ← 下载的原始文件放这
│
├─ evals/                          ★ 评估资产
│  ├─ conflict_rules.jsonl         ⭐ 冲突规则表 40 条(E2 的 ground truth)
│  ├─ README.md                    字段说明 · 八种关系类型 · 扩充指南
│  └─ reference_tables/            ⭐ 体质 × 季节 45 条(E1 的 ground truth)
│
├─ planning/                       自用工作文件,⚠️ 不进 repo
│  ├─ roadmap.md                   执行路线图:十阶段 + 学习资料 + 完成判据
│  ├─ worknotes.md                 待办 · 风险 · 红队预演 · 目录结构 · 简历 bullet
│  ├─ resources.md                 学习资源汇总(按主题)
│  └─ PRD-working-draft.md         PRD 合并阅读版(含全部过程注释)
│
└─ _archive/                       ← 已被取代,确认无遗漏后可整个删掉
   ├─ 01-learning-path.md          有用内容已并入 planning/resources.md
   ├─ 02-project-plan.md           有用内容已并入 planning/worknotes.md
   └─ 08-domain-knowledge-*.md     有用内容已并入 knowledge/
```

---

## 二、这轮整理做了什么

| 操作 | 详情 |
|---|---|
| **移动** | `10/11/12` → `planning/`,并全部重命名去掉数字前缀 |
| **合并** | `01-learning-path` + `05-resources` → `planning/resources.md`;`02` 的目录结构与成本表 → `planning/worknotes.md` 与 `resources.md` |
| **提取** | `08` 的五行对应表 → `knowledge/tcm/`;精准营养学证据 → `knowledge/nutrition/` |
| **抢救** | `09` 已被你删除,但四段 system prompt 从上下文中恢复,存入 `docs/prompts/` |
| **归档** | `01`、`02`、`08` 移入 `_archive/` |
| **修正** | 全部交叉引用改为新路径;成本 $10-20 → $50-100;"不上多 agent"一条标注为已被 D1/D17 修订 |

**`_archive/` 里三份的有用内容都已提取完毕,可以整个删掉。** 保留只是给你一次核对机会。

---

## 三、当前欠账(按优先级)

| # | 事项 | 说明 |
|---|---|---|
| 1 | **跑两个 spike** | A1 手记饮食 5 天(每天 2 分钟,今天就能开始)、A4 playground 试三个 prompt。验证的是地基假设,见 `planning/worknotes.md` 第一节 |
| 2 | **找中医背景的人抽检** | 规则表抽 10 条 + 参考表抽 10-15 条,记录一致率。**成本极低但对 R1 循环论证的缓解效果最好**,不要省 |
| 3 | **补齐 22 条 `needs_source`** | 规则表 40 条但只有 18 条可用于 eval。**优先级高于新增规则**——数量不是目标,可用数量才是 |
| 4 | **食材性味归经表** | 300-500 条,🔴 P0,三张自建表里唯一还没动的。没有它中医侧检索是空的 |
| 5 | **下载 P0 资料** | ZYYXH/T157-2009 全文 · USDA FDC 子集 · RealFood.gov 指南 |
| 6 | **补写两段 prompt** | 调和层与核查 pass 还没有 prompt,见 `docs/prompts/system-prompts.md` §5、§6 |

### 一处已确认要改但还没动的

- **E2 应拆成 E2a / E2b**。E2a 命中规则表测应用,E2b 用表外冲突测迁移。全部来自规则表的话,M5 测的是"会不会查表"而不是"会不会调和"。需要同步改 `docs/PRD.md` §8.1 与 §8.2

---

## 四、谁看哪些文件

| 文件 | 读者 |
|---|---|
| `docs/` | 面试官、招聘方 |
| `knowledge/`、`evals/` | 施工用,也是面试时能打开的实物 |
| `planning/` | **只有你**。`worknotes.md` 里有红队预演和自省风险,不要放进 repo |
