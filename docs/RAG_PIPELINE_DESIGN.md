# RAG 完整方案：格式转换 · 切块 · Embedding · 存储

**日期** 2026-08-25 · 关联：`docs/DECISIONS.md`(D2/D3/D4/D8/D18/D19)· `planning/step1-naive-rag/`

> 这份文档做三件事：科普每个环节在解决什么问题、给出 2-3 个真实可选方案对比、给出针对本项目的推荐和理由。
> 结论会补进 `DECISIONS.md` 作为新决策（D23），过程放这里，理由讲透。

---

## 零、先把四个环节的边界画清楚

你问的其实是四个独立的问题，容易被混在一起：

```
原始文件(PDF/JSON/XML/...) → ① 格式转换 → 纯文本
纯文本 → ② 切块(chunking) → 一堆 chunk
每个 chunk → ③ embedding(tokenization 是它内部的一步) → 向量
chunk + 向量 + 元数据 → ④ 存储 → 检索时能查出来
```

**一个常见误解要先澄清**：中英文在③（embedding）这一步**几乎不需要分开处理**。现代 embedding 模型（BGE、OpenAI、Voyage 这些）内部自带子词分词器（subword tokenizer，比如 BPE/SentencePiece），是直接在原始中文字符上训练出来的，**不需要 jieba 这类分词工具**。上次我用 jieba 是因为 BM25 这种"数词频"的老方法必须先知道"词"的边界，这是 BM25 独有的局限，不是 RAG 通用问题。真正需要区分中英文的地方是①（PDF 排版/OCR）和②（切块时按 token 数还是字符数算长度——中文一个字通常算 1.5-2 个 token，英文一个词约 0.75 个 token，同样"500"这个数字在两种语言里对应的实际信息量差很多）。

---

## 一、格式转换：要不要上 markitdown

### 先说个反直觉的结论

**markitdown 这类"万物转 Markdown"工具，对你手上这批资料，用处比想象的小。** 你的原始文件里，PDF 只占一部分（8 个），剩下的 JSON/JSONL/XML/MD 本身已经是结构化数据——把 JSON 转成 Markdown 反而是把结构拆掉再让 embedding 模型自己猜回来，不如直接按原始结构解析（这正是我 `ingest.py` 现在干的事：JSON 走 `json.load`，XML 走 `ElementTree`，各自保留字段语义）。**markitdown 类工具该用在"给人看的文档格式"上（PDF/Word/PPT/图片），不该用在"给程序看的结构化格式"上。**

所以问题收窄成：**PDF 这一类，markitdown vs 其他工具怎么选。**

### 三个真实选项对比（查了实测数据）

| 工具 | 谁做的 | 速度(14页PDF) | 表格质量 | 安装体积 | 适合场景 |
|---|---|---|---|---|---|
| **MarkItDown** | Microsoft | 0.6 秒 | ❌ 表格直接抽成一段乱文字 | 轻量 | 干净的纯文本 PDF，要求速度 |
| **Docling** | IBM Research | 41 秒 | ✅ 表格还原成规整 Markdown 表格 | 2.4GB(带 PyTorch) | **表格多、CPU 环境、要程序化访问结构** |
| **Marker** | Datalab | 2分14秒 | ✅ 表格+公式都好，图片单独存 PNG | 需要 GPU 才快 | 含大量数学公式的文档；商用超 500万美元营收要付费 |

（来源见文末，实测机构在 14 页 EU 法规 PDF 和 22 页 arXiv 论文上分别测的）

### 推荐

**不引入 markitdown，PDF 这块升级成 Docling，只用在两个真正需要表格结构的文件上：`Daily Serving Sizes.pdf`（现在那个表头错位的 bug 就是 pdfplumber 表格提取不够准）和 `GB7718-2025.pdf`（过敏原分类表）。** 其余纯文字 PDF（FDA 指南、dietary guidelines）继续用现在的 `pdfplumber` 按页抽取就够，没必要为了统一工具链换掉已经跑通的部分——Docling 2.4GB 的安装体积对这几个文件不值得。

**为什么不是"全部换成一个统一转换器"**：你的资料格式本身就是异构的（JSON 的营养成分、XML 的补剂说明、PDF 的国标条文、MD 的自建表），**格式转换层的目标应该是"保留住每种格式里对检索最有用的结构"，不是"长得整齐"**。这也是为什么 `ingest.py` 现在是按格式分函数处理，而不是先全转 Markdown 再统一切块——这个设计已经是对的，不用推倒重来。

---

## 二、切块（chunking）+ tokenization：怎么切、按什么算长度

### 现状

`ingest.py` 现在是"按结构切"（D3 已经定的方向）：MD 按表格行/标题分节、JSONL 按记录、XML 按 h2/h3、PDF 按页。这个方向本身没问题，**真正该升级的是"怎么判断一个 chunk 够不够大/够不够小"**——现在完全没有长度控制，PDF 按页切，有的页几十字有的页上千字，参差不齐。

### 三个选项对比

| 方案 | 怎么做 | 优点 | 缺点 |
|---|---|---|---|
| **A. 固定字符数滑窗**（你 `naive_rag.py` 原来那版） | 每 500 字符切一块，重叠 50 字符 | 实现最简单 | 完全不管语义边界，"性-味-归经"这种完整信息会被从中间切断——你 Step1 README 自己就写了这是"故意写烂的" |
| **B. 结构优先切分**（现在 `ingest.py` 在做的） | 按数据自身的边界切（表格行/标题/XML 节/JSON 记录） | 每个 chunk 是一个语义完整的"条目"，符合 D3 | 有的结构块可能过大（比如`34.中医食疗学.pdf`一整段没有明显小标题的正文），也可能过小（一行表格只有几个字） |
| **C. Token 感知的二次切分**（推荐：在 B 基础上加一层） | 结构切完之后，检查每个 chunk 用**目标 embedding 模型自己的 tokenizer** 数一下 token 数，超过阈值（比如 300-500 token）的再按句子边界二次切；过小的（比如不到 20 token）考虑跟相邻块合并 | 兼顾语义完整性和长度均匀性，直接对齐模型的实际输入限制 | 需要多写一层逻辑，且要选定 embedding 模型才能拿到对应的 tokenizer |

### 推荐

**用方案 C：保留现在的结构优先切分（B）作为第一刀，再加一层 token 感知的二次处理。** 具体到中英文的差异点：**用你选定的 embedding 模型自带的 tokenizer 来数 token 数，不要用字符数或者 jieba 分词数**——比如用 `tiktoken`（OpenAI 系）或者 HuggingFace 的 `AutoTokenizer.from_pretrained("BAAI/bge-m3")` 来数，这样"中文 300 token"和"英文 300 token"用的是同一把尺子，模型看到的实际输入长度是一致的，不会出现"中文 chunk 明明字数不多但已经快超限，英文 chunk 字数很多却还早得很"的情况。

---

## 三、Embedding 模型：三个选项对比

查了当前的评测数据（多语言/中英文跨语言检索能力，MTEB 一类的评测集）：

| 模型 | 类型 | 体量 | 中英文混合检索能力 | 成本 | 适合阶段 |
|---|---|---|---|---|---|
| **BGE-M3**（智源） | 开源，本地跑 | 568M 参数 | 支持 100+ 语言，中文这块是训练重点之一；文档超过 8K token 后质量会下降 | 免费(自己出算力) | **开发/迭代期，Step 1-3 都用它** |
| **Voyage**（Anthropic 推荐的 embedding 供应商） | API | — | 多语言表现扎实，价格和质量平衡好 | 按量付费，比 OpenAI 便宜 | D19 说的"正式跑分/交付档" |
| **OpenAI text-embedding-3-large** / **Gemini Embedding** | API | — | 评测里跨语言检索分数最高的一档（Gemini 在"内容对齐"这个子项接近满分） | 比 Voyage 贵 | 可选的"天花板对比"，不一定要用在生产 |

（有个反直觉但很关键的发现：**体量小的纯英文模型（比如 335M 参数的 mxbai-embed-large）在跨语言任务上的分数掉到了 0.16 左右**——不是"小一点效果差一点"，是"不支持中文的模型在中文任务上几乎不可用"，这解释了为什么不能随便拿个热门的英文 embedding 模型来用。）

### 推荐

**开发迭代期用 BGE-M3（本地、免费、中文强）；正式交付/跑分时切到 Voyage（API，成本可控，多语言表现好）。** 这正好对上你自己 D19 已经定的"双档模型策略"，不用另开一条决策——BGE-M3 就是"低成本档"的具体落地，Voyage 就是"交付档"的候选之一（跟 D19 一样，PRD 里不锁定具体模型名，这里只是给你一个可以先用起来的默认值）。

**不推荐现在就上 OpenAI/Gemini 那一档**：分数最高不等于性价比最高，你项目的成本模型（PRD §14.3）本来就卡得紧，BGE-M3 + Voyage 这个组合已经能覆盖开发到交付两个阶段，没必要为了多出的几个百分点分数多花钱。

---

## 四、存储：要不要图数据库

### 先说你的直觉对在哪

你说"要思考语义检索 + 图数据库联合检索"，这个直觉是对的——**你的冲突规则表本质上就是一张图**：体质节点、食材节点、补剂节点，边是"宜/忌/交互"关系，"西洋参 → 抑制 P450 → 削弱华法林药效"这种链条本身就是一次图上的多跳查询。纯向量检索确实不擅长这种"精确的多跳关系"，这也是查到的资料里反复强调的点：向量库擅长"找相似"，图数据库擅长"按关系精确走"。

### 三个选项对比

| 方案 | 怎么做 | 优点 | 缺点 |
|---|---|---|---|
| **1. 纯 pgvector**（现在 D4 定的） | 关系表 + 向量都在 Postgres 一个库里 | 部署最简单，D4 的理由(免费、同库 JOIN 预筛、单机部署)依然成立 | 多跳关系查询要手写递归 SQL(Postgres 支持 `WITH RECURSIVE`，能做，但不如图数据库原生的 Cypher 查询直观) |
| **2. Postgres(关系表建模图结构) + pgvector**（推荐） | 冲突规则、体质-食材关系等**用普通关系表 + 外键/关联表建模**，需要多跳时写递归 CTE；语义检索继续用 pgvector | 不引入新系统，一个数据库搞定关系查询、向量检索、用户记忆(D18)三件事；你现在的规模（40 条冲突规则、9 种体质、106 种食材）用关系表完全够用，图数据库的优势要在成千上万节点、深层多跳的场景才明显 | 关系很复杂时(比如要做"找出所有间接影响华法林代谢的食材路径"这种深度不确定的查询) SQL 会变得难写 |
| **3. pgvector + 独立图数据库(Neo4j)** | 语义检索留在 Postgres，关系推理挪到 Neo4j，两库联合查询 | 图查询语言(Cypher)表达多跳关系远比 SQL 直观；未来关系复杂度上升时扩展性更好 | **多一套系统要部署、维护、备份**；对你现在的数据规模是过度设计；两个数据库之间的数据一致性又是新问题 |

查到的资料里有一条建议正好适用于你现在的阶段："对单用户项目，先用向量数据库起步，等语义检索的局限性真正暴露出来了，再有选择地引入知识图谱"——这跟你自己 D4 的"单机部署简化交付"精神是一致的。

### 推荐

**方案 2：不引入独立图数据库，冲突关系用 Postgres 的关系表建模，需要多跳查询时写递归 CTE。** 三个理由：

1. **规模不支撑**：你现在的关系数据是"40 条规则、9 种体质、106 种食材"这个量级，图数据库的性能优势在几十万节点以上才明显，你这里用不上
2. **和已有决策一致**：D18 已经把用户画像、饮食记录都放进同一个 Postgres；D4 的"单机部署"理由对图数据库同样成立——**这正好回答你问的"哪个便于之后做 agent 时还要存其他记忆"：答案是同一个 Postgres 库，用户记忆(D18)、冲突关系表、向量检索(D4)三样东西共用一个数据库实例，agent 需要的所有存储都在一个地方，不用管理跨库一致性**
3. **留了退路**：如果 V2 真的要做更复杂的多跳推理（比如自动发现规则表里没覆盖的间接冲突），到时候再引入 Neo4j 是加一层，不是推倒重来——数据本身(哪些是节点、哪些是边)已经在关系表里了，迁移成本可控

---

## 五、完整 pipeline（今天定下来的样子）

```
① 格式转换
   PDF(表格多) ──Docling──┐
   PDF(纯文字) ──pdfplumber──┤
   JSON/JSONL/XML/MD ──原生解析(ingest.py 已有)──┤
                                              ↓
② 切块                                    纯文本 + 元数据
   结构优先切分(现有逻辑) → 用 BGE-M3 的 tokenizer 数 token → 超限的二次切句子级
                                              ↓
③ Embedding                              chunk 列表
   开发期: BGE-M3(本地) ｜ 交付期: Voyage(API)  ← D19 双档策略的具体落地
                                              ↓
④ 存储                                    chunk + 向量 + 元数据
   Postgres 一个实例:
     - pgvector 存向量(tcm/nutrition 两个 collection, D2)
     - 普通关系表存冲突规则/体质-食材关系(图结构用外键建模)
     - 同库存用户画像/饮食记录(D18)
```

### BGE-M3 → pgvector 落地脚本

实现见 `db/embed_bge_m3.py` + `db/schema.sql` 里的 `knowledge_chunks` 表（dense 1024 维，按 `domain` 区分 tcm/nutrition）。

```bash
psql "$DIET_EXPERT_PG_DSN" -f db/schema.sql
pip install FlagEmbedding psycopg2-binary pgvector torch
python3 db/embed_bge_m3.py load --root .            # 读 _processed/*.jsonl，UPSERT
python3 db/embed_bge_m3.py search --query "气虚质适合吃什么" --domain tcm
```

---

## 六、几件你提到的具体事，顺带处理

**黄帝内经白话文版本**：能解决之前"疑似导读非原文"的顾虑，但白话文翻译本身通常是译者的著作权作品（跟古文原文的公有领域属性不是一回事）——摄入的时候在 `metadata` 里把译者/出版信息记全，`conflict_rules.jsonl` 引用这类内容时的 `source_status` 建议标成"白话译本，非原文"而不是直接当一次来源，这样不会破坏你自己 R1 的防循环论证要求。文件放好之后告诉我文件名，我帮你把 `ingest.py` 里那段指向新文件。

**`KA4224...倪世美.pdf`**：你说了只是自用 prototype，没问题，保持"不整本入库、只做人工核对补页码"的用法就行，不用改。

**`FoodData_Central_sr_legacy`**：`ingest.py` 已经支持，加个参数就行：

```bash
python3 planning/step1-naive-rag/ingest.py --root . --include-sr-legacy
```

---

## 参考来源

- [MarkItDown vs Docling vs Marker 对比实测](https://www.danilchenko.dev/posts/markitdown-vs-docling-vs-marker/)
- [Best PDF Parsers for AI and RAG Workflows in 2026](https://www.firecrawl.dev/blog/best-pdf-parsers)
- [Embedding 模型 2026 基准评测](https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html)
- [Qwen-3 vs BGE-M3 多语言检索对比](https://medium.com/@mrAryanKumar/comparative-analysis-of-qwen-3-and-bge-m3-embedding-models-for-multilingual-information-retrieval-72c0e6895413)
- [Vector Databases vs. Graph RAG for Agent Memory](https://machinelearningmastery.com/vector-databases-vs-graph-rag-for-agent-memory-when-to-use-which/)
- [HybridRAG: 何时结合向量与知识图谱](https://memgraph.com/blog/why-hybridrag)
