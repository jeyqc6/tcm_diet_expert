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
    -- 主体质（CCMQ 九分类之一，如 qi_xu/yang_xu/tan_shi/ping_he...），CCMQ 计分下最高转化分对应的类型，或用户自述
    constitution                TEXT,
    -- 次要体质（体质夹杂，D28）：CCMQ 计分下达到"倾向是"阈值但非最高分的类型；用户自述路径下通常为空
    constitution_secondary      TEXT[] NOT NULL DEFAULT '{}',
    -- D28：这条信息是自述还是问卷算出来的，直接影响 TCM SubAgent 给建议时该多确定的语气
    constitution_source         TEXT CHECK (constitution_source IN ('self_reported', 'ccmq_computed', 'unconfirmed')),
    -- 人在环确认发生的时间；为空表示还没走过确认流程。体质是相对稳定的属性，不随饮食记录自动漂移
    constitution_confirmed_at   TIMESTAMPTZ,
    allergens                   TEXT[] NOT NULL DEFAULT '{}',
    -- 在服补剂，结构未强定（如 [{"name": "维生素D", "dose": "1000IU/day"}]）
    supplements                 JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- D16：定性目标标签（如 weight_management），不含热量缺口/体重目标这类数值
    goal_tags                   TEXT[] NOT NULL DEFAULT '{}',
    -- D25：忌口/口味耐受/长期性用餐场景限制（如 {"dislikes": ["香菜"], "spice_tolerance": "high"}）
    -- 回答"方案要符合哪些约束"，和 goal_tags"身体希望往哪个方向调理"是两件不同的事，不要合并
    preferences                 JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
    -- D27：0=原文,1/2=按 Tier 摘要,3=跨会话摘要指针，见 ARCHITECTURE.md §4.4
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
