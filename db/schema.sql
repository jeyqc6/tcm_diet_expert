-- diet_expert · recipes 表
-- 对应 docs/DECISIONS.md D24："按食材查菜谱"走 SQL 精确/包含过滤，不走 RAG 语义检索。
-- 和 D4(pgvector)、D18(用户饮食记录)共用同一个 Postgres 实例，这里不建新数据库/新实例。
--
-- 用法：
--   psql "$DIET_EXPERT_PG_DSN" -f db/schema.sql

CREATE TABLE IF NOT EXISTS recipes (
    id              BIGSERIAL PRIMARY KEY,
    -- 数据源里的行号，用来支持"重新导入时先删同源数据"这种幂等操作，不是业务字段
    source_row      INTEGER,
    name            TEXT NOT NULL,
    dish            TEXT,
    description     TEXT,
    -- 食材保留成数组，是这张表存在的核心原因：GIN 索引下可以做 `ingredients && ARRAY[...]`
    -- 包含查询，百万级数据也是毫秒级；RAG 的向量/BM25 检索做不到"一定包含某个词"这件事。
    ingredients     TEXT[] NOT NULL DEFAULT '{}',
    instructions    TEXT[] NOT NULL DEFAULT '{}',
    author          TEXT,
    source          TEXT NOT NULL DEFAULT 'XiaChuFang Recipe Corpus',
    -- 学术数据集，主页未声明明确 license，按"仅供研究/个人原型使用，不整库对外分发或商用"处理，
    -- 见 knowledge/_raw/README.md 三、营养学 表里 recipe_xiachufang.json 那一行。
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 核心索引：食材数组的包含查询
CREATE INDEX IF NOT EXISTS idx_recipes_ingredients ON recipes USING GIN (ingredients);

-- 可选：如果以后想支持"菜名模糊搜索"（比如用户直接搜"红烧"），
-- 需要先 `CREATE EXTENSION IF NOT EXISTS pg_trgm;`（一般云托管 Postgres 都自带，本地要确认装了 contrib），
-- 再打开下面这行。现在 D24 的范围只到"按食材查"，先不建，免得引入用不到的依赖。
-- CREATE INDEX IF NOT EXISTS idx_recipes_name_trgm ON recipes USING GIN (name gin_trgm_ops);

-- 查询示例：
--   同时含有"山药"和"红枣"的菜：
--     SELECT name, ingredients FROM recipes WHERE ingredients && ARRAY['山药','红枣'] LIMIT 20;
--   必须同时含有两者（不是"任一"）：
--     SELECT name, ingredients FROM recipes WHERE ingredients @> ARRAY['山药','红枣'] LIMIT 20;
--   `&&` = 有交集（任一命中），`@>` = 完全包含（全部命中）——按食材查菜谱通常想要后者。


-- =============================================================================
-- knowledge_chunks · RAG 向量表（D2 / D4 / D23）
-- =============================================================================
-- 对应 docs/RAG_PIPELINE_DESIGN.md §三 / §四：
--   - 开发期 embedding = BAAI/bge-m3（dense 1024 维）
--   - tcm / nutrition 两个 collection 用 domain 字段区分（同表，便于同库 JOIN 预筛）
--   - 文本来源：knowledge/_processed/{tcm,nutrition}_chunks.jsonl（由 ingest.py 产出）
--
-- 需要先装 pgvector 扩展（本地: CREATE EXTENSION；Supabase/Neon 一般已启用）。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id              BIGSERIAL PRIMARY KEY,
    chunk_id        TEXT NOT NULL UNIQUE,
    domain          TEXT NOT NULL CHECK (domain IN ('tcm', 'nutrition')),
    source_file     TEXT NOT NULL,
    source_type     TEXT,
    text            TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- BGE-M3 dense 向量固定 1024 维；换模型要改维度并重建表/索引
    embedding       vector(1024) NOT NULL,
    -- BGE-M3 sparse(词法)向量——同一次 model.encode()顺带产出的第二路输出，
    -- 不是另一个模型。维度=BGE-M3 底层 XLM-R tokenizer 的 vocab_size(250002，
    -- 实测值，见 db/embed_bge_m3.py EMBED_DIM_SPARSE 注释)，每个非零位对应
    -- 一个 token id 的词法权重。可空——旧数据/增量 ingest 还没跑过 sparse
    -- 编码时，混合检索(_retrieval_common.py)按"这一路没有数据"静默降级，
    -- 不強制要求这一列非空。用于缓解纯稠密向量对"疏肝""祛湿"这类专业术语
    -- 精确命中不够稳的问题(2026-08-30，检索评分方法优化)。
    sparse_embedding sparsevec(250002),
    embed_model     TEXT NOT NULL DEFAULT 'BAAI/bge-m3',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_domain
    ON knowledge_chunks (domain);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source
    ON knowledge_chunks (source_file);

-- 余弦距离索引。向量写入前做 L2 归一化后，`<=>` 与「1 - 余弦相似度」等价。
-- 数据量现在约 1 万级，HNSW 即可；百万级再考虑调 m / ef_construction。
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding_hnsw
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);

-- sparsevec 目前不建索引——同样是万级行量级，顺序扫描足够快(同 D4 对稠密
-- 向量在这个规模下"HNSW 只是锦上添花,不是必需"的判断一致)，且 pgvector
-- 的 sparsevec HNSW 支持相对新，不必要为了这个规模的数据引入额外风险。

-- 检索示例（限定中医 collection，取 top 5）：
--   SELECT chunk_id, source_file, left(text, 120) AS preview, 1 - (embedding <=> $1) AS score
--   FROM knowledge_chunks
--   WHERE domain = 'tcm'
--   ORDER BY embedding <=> $1
--   LIMIT 5;


-- =============================================================================
-- user_profile · 常驻上下文（D5 / D16 / D25 / D28）
-- =============================================================================
-- 对应 docs/ARCHITECTURE.md §1.2。V1 单用户，但字段里留 user_id 给以后扩展，
-- 一个 user_id 对应一行（唯一约束），不是靠应用代码保证"只插一行"。

CREATE TABLE IF NOT EXISTS user_profile (
    id                          BIGSERIAL PRIMARY KEY,
    user_id                     TEXT NOT NULL DEFAULT 'default_user' UNIQUE,
    -- 前端用户切换器显示用的名字（如"我"/"老公"）——纯展示字段，不参与任何匹配/
    -- 查询逻辑，`user_id` 才是真正的外键/隔离键。为空时前端退回显示 user_id 本身。
    display_name                TEXT,
    -- 主体质（CCMQ 九分类之一，如 qi_xu/yang_xu/tan_shi/ping_he...），CCMQ 计分下最高转化分对应的类型，或用户自述
    constitution                TEXT,
    -- 次要体质（体质夹杂，D28）：CCMQ 计分下达到"倾向是"阈值但非最高分的类型；用户自述路径下通常为空
    constitution_secondary      TEXT[] NOT NULL DEFAULT '{}',
    -- D28：这条信息是自述还是问卷算出来的，直接影响 TCM SubAgent 给建议时该多确定的语气
    constitution_source         TEXT CHECK (constitution_source IN ('self_reported', 'ccmq_computed', 'unconfirmed')),
    -- 人在环确认发生的时间；为空表示还没走过确认流程。体质是相对稳定的属性，不随饮食记录自动漂移
    constitution_confirmed_at   TIMESTAMPTZ,
    allergens                   TEXT[] NOT NULL DEFAULT '{}',
    -- 用户所在城市（自然语言，如"上海"/"San Francisco"），两个消费者：
    -- 1) query_weather 的 city 参数直接从这读，不用每次对话都问一遍用户在哪
    -- 2) timezone 为空时，query_diet_log 等需要"今天"这类相对日期的地方可以拿它做兜底提示
    --    （但不做"从城市名自动反查时区"这种地理编码，容易在时区边界/夏令时上出错——
    --    timezone 才是被代码实际使用的字段，city 主要是给 query_weather 用、给 timezone 兜底提供人类可读的线索）
    city                        TEXT,
    -- IANA 时区名（如 "Asia/Shanghai"），相对日期解析("今天"/"昨天")和"当前时间"判断的权威依据。
    -- 为空时退到 DIET_EXPERT_TZ 环境变量，再没有就是 Asia/Shanghai——见 backend/mcp_server/tools/query_diet_log.py。
    -- 首次使用引导（§11）问 city 时应该一起问/推导 timezone 并显式确认，不从 city 静默猜。
    timezone                    TEXT,
    -- 在服补剂，结构未强定（如 [{"name": "维生素D", "dose": "1000IU/day"}]）
    supplements                 JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- D16：定性目标标签（如 weight_management），不含热量缺口/体重目标这类数值
    goal_tags                   TEXT[] NOT NULL DEFAULT '{}',
    -- D25：忌口/口味耐受/长期性用餐场景限制（如 {"dislikes": ["香菜"], "spice_tolerance": "high"}）
    -- 回答"方案要符合哪些约束"，和 goal_tags"身体希望往哪个方向调理"是两件不同的事，不要合并
    preferences                 JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- True after the first-conversation intro finishes (answered or「全部跳过」).
    -- create_user() stubs stay FALSE so chat still starts onboarding; do not
    -- infer this from empty constitution / constitution_source.
    onboarding_done             BOOLEAN NOT NULL DEFAULT FALSE,
    -- UI / conversation language for this user (frontend toggle). Default zh
    -- so existing rows and clients stay Chinese. Not inferred from Accept-Language.
    locale                      TEXT NOT NULL DEFAULT 'zh' CHECK (locale IN ('zh', 'en')),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Existing databases created before locale existed: ADD COLUMN is a no-op when
-- the column is already there (fresh CREATE TABLE above already includes it).
ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS locale TEXT NOT NULL DEFAULT 'zh';
UPDATE user_profile SET locale = 'zh' WHERE locale IS NULL;
ALTER TABLE user_profile ALTER COLUMN locale SET DEFAULT 'zh';
ALTER TABLE user_profile ALTER COLUMN locale SET NOT NULL;
ALTER TABLE user_profile DROP CONSTRAINT IF EXISTS user_profile_locale_check;
ALTER TABLE user_profile ADD CONSTRAINT user_profile_locale_check CHECK (locale IN ('zh', 'en'));

CREATE INDEX IF NOT EXISTS idx_user_profile_allergens ON user_profile USING GIN (allergens);


-- =============================================================================
-- diet_log · 饮食记录明细（D18）
-- =============================================================================
-- 供 query_diet_log() 聚合查询；写路径幂等键对应 ENGINEERING §1.2:
-- idempotency_key = hash(user_id, logged_at, raw_input_hash)。

CREATE TABLE IF NOT EXISTS diet_log (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             TEXT NOT NULL DEFAULT 'default_user',
    -- 这顿饭实际发生的时间：用户输入或按时段规则推断；幂等键用它，不是 recorded_at
    logged_at           TIMESTAMPTZ NOT NULL,
    -- 系统实际写入的时间，审计用；可能晚于 logged_at（比如用户补记）
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 早餐/午餐/晚餐/夜宵/下午茶/加餐/未知，§4.2 按关键词/时段确定性推断，不额外调模型
    meal_type           TEXT NOT NULL DEFAULT '未知'
        CHECK (meal_type IN ('早餐', '午餐', '晚餐', '夜宵', '下午茶', '加餐', '未知')),
    raw_input           TEXT NOT NULL,
    -- 拆解后的菜品列表，如 [{"dish": "番茄炒蛋", "confidence": "high"}]
    dishes              JSONB NOT NULL DEFAULT '[]'::jsonb,
    ingredients         TEXT[] NOT NULL DEFAULT '{}',
    -- 中医食性标签，如 "温"/"寒"/"平"
    food_properties     TEXT[] NOT NULL DEFAULT '{}',
    idempotency_key     TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_diet_log_user_logged_at ON diet_log (user_id, logged_at DESC);
CREATE INDEX IF NOT EXISTS idx_diet_log_user_meal_type ON diet_log (user_id, meal_type);
CREATE INDEX IF NOT EXISTS idx_diet_log_ingredients ON diet_log USING GIN (ingredients);


-- =============================================================================
-- conversation_sessions / messages · 会话历史原文（D8 · §5.3 分层压缩）
-- =============================================================================
-- 供分层压缩读取和生成摘要。ARCHITECTURE.md §1.2 只给了 message 级字段，
-- session 级字段（user_id/起止时间）是本次建表时按最小设计补的，供 messages 外键引用。

CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default_user',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES conversation_sessions (session_id),
    turn_index          INTEGER NOT NULL,
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content             TEXT NOT NULL,
    -- D27 补充(2026-08-28，backend/memory/compression.py 接线)：结构化归档摘要
    -- (TurnRecord/ArchivedSummary)需要的字段，raw(tier0)行写入时就填好，归档
    -- 时不需要重新解析 content 才能拿到 branch/结论/引用/被拒建议/guardrail。
    branch                  TEXT,
    conclusion              TEXT,
    cited_source_ids        TEXT[] NOT NULL DEFAULT '{}',
    rejected_suggestions    TEXT[] NOT NULL DEFAULT '{}',
    triggered_guardrails    TEXT[] NOT NULL DEFAULT '{}',
    -- compression_tier 取值约定(backend/memory/session_store.py 消费)：
    --   0 = Tier1，原文，当前会话，尚未归档
    --   1 = Tier2，结构化归档摘要，会话仍在进行中(未判定空闲)
    --   2 = 保留未使用
    --   3 = Tier3，结构化归档摘要，会话已判定结束(跨会话)
    -- 见 ARCHITECTURE.md §4.4/§4.4.1、DECISIONS.md D27/D27 补充。
    compression_tier    SMALLINT NOT NULL DEFAULT 0 CHECK (compression_tier IN (0, 1, 2, 3)),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, turn_index)
);


-- =============================================================================
-- conflict_rules · 冲突规则表（D23 关系表建模；字段对齐 evals/conflict_rules.jsonl）
-- =============================================================================
-- evals/conflict_rules.jsonl 保留作为人工编辑的源文件（diff 友好），
-- 经幂等 ingest 脚本 ON CONFLICT (rule_id) DO UPDATE 灌进本表，查询走表、编辑走 JSONL。

CREATE TABLE IF NOT EXISTS conflict_rules (
    id                          BIGSERIAL PRIMARY KEY,
    rule_id                     TEXT NOT NULL UNIQUE,
    line                        TEXT,
    topic                       TEXT NOT NULL,
    tcm_position                TEXT NOT NULL,
    tcm_source                  TEXT,
    nutrition_position          TEXT NOT NULL,
    nutrition_source            TEXT,
    relation                    TEXT NOT NULL CHECK (relation IN (
        'conflict', 'partial_conflict', 'conditional_conflict',
        'aligned', 'aligned_negative', 'complementary',
        'tcm_internal', 'nutrition_internal'
    )),
    resolution                  TEXT,
    resolution_rationale        TEXT,
    confidence                  TEXT CHECK (confidence IN ('high', 'medium', 'low')),
    -- 自由文本（如"双边实证"/"中医传统+营养实证"），取值种类多，不建 CHECK
    evidence_level              TEXT,
    applicable_constitutions    TEXT[] NOT NULL DEFAULT '{}',
    applicable_goals            TEXT[] NOT NULL DEFAULT '{}',
    source_status               TEXT NOT NULL CHECK (source_status IN ('verified', 'needs_source')),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conflict_rules_constitutions ON conflict_rules USING GIN (applicable_constitutions);
CREATE INDEX IF NOT EXISTS idx_conflict_rules_goals ON conflict_rules USING GIN (applicable_goals);


-- =============================================================================
-- user_dish_aliases · 个人菜品简称（D27 修订一）
-- =============================================================================
-- 程序性记忆：记录"这个用户这样说时具体指什么"。晋升规则是确定性代码（计数阈值），
-- 不是模型判断——§4.2 查表顺序：dish_ingredient_map → user_dish_aliases(仅命中已晋升的行)
-- → LLM 兜底。LLM 兜底结果经人在环确认且未被修改，才写入本表候选行并计数。

CREATE TABLE IF NOT EXISTS user_dish_aliases (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             TEXT NOT NULL DEFAULT 'default_user',
    -- 去空白/标点后的原始说法，如"西红柿炒蛋加饭"
    normalized_phrase   TEXT NOT NULL,
    dishes              JSONB NOT NULL DEFAULT '[]'::jsonb,
    ingredients         TEXT[] NOT NULL DEFAULT '{}',
    hit_count           INTEGER NOT NULL DEFAULT 1,
    -- 为空表示仍是候选、未生效；晋升阈值命中后写入时间戳
    promoted_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, normalized_phrase)
);


-- =============================================================================
-- pending_critical_facts · PRD §10.2 human-in-the-loop (D34)
-- =============================================================================
-- Scanner hits land here until the user confirms. Not merged into the current
-- turn's UserProfileContext and not UPSERTed into user_profile until confirm.

CREATE TABLE IF NOT EXISTS pending_critical_facts (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default_user',
    session_id      TEXT NOT NULL,
    allergens       TEXT[] NOT NULL DEFAULT '{}',
    supplements     JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Pending clarification (D20 #3 / D33) and in-progress chat onboarding.
-- Process-local InMemory stores lose these on restart; Postgres is the default.

CREATE TABLE IF NOT EXISTS pending_clarifications (
    session_id      TEXT PRIMARY KEY,
    original_text   TEXT NOT NULL,
    branch          TEXT NOT NULL,
    domain_hint     TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS onboarding_sessions (
    user_id         TEXT PRIMARY KEY,
    step_id         TEXT NOT NULL,
    state           JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent for existing volumes: CREATE TABLE IF NOT EXISTS does not add columns.
ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS onboarding_done BOOLEAN NOT NULL DEFAULT FALSE;

-- Rows that already have constitution data (or the previous skip latch) have
-- been through intro; brand-new create_user stubs stay FALSE.
UPDATE user_profile
SET onboarding_done = TRUE
WHERE onboarding_done = FALSE
  AND (constitution IS NOT NULL OR constitution_source IS NOT NULL);
