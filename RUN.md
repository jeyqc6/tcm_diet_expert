# Diet Expert · 运行指南

根目录 `README.md` 是项目索引。**这份讲怎么把服务跑起来。**

两条路径：

| | Docker 一键起 | 本地开发 |
|---|---|---|
| 适合 | 验证「clone 下来能不能跑」、演示 | 改代码立刻看到效果 |
| 热更新 | 没有。代码打进镜像，改完要 `--build` | 有（uvicorn `--reload` + Next `dev`） |
| 首次耗时 | 长（镜像要装 torch；首次还会拉 BGE-M3 权重） | 看本机有没有现成 Postgres / 依赖 |

跑起来之后：

- 前端：http://localhost:3000
- API：http://localhost:8123
- 健康检查：`GET http://localhost:8123/healthz`

后端已经不止聊天：`POST /api/chat`（SSE）、`GET/PATCH /api/profile`、`POST /api/onboarding/{start,answer}`、`GET /api/sessions/{session_id}/messages`。会话轮次会写入 Postgres（`backend/memory/session_store.py`）；前端用 `localStorage` 记住 `session_id`，刷新后拉历史。

前端最小闭环现在处理 `token` / `source` / `guardrail` / `done` / `clarification` / `task` / `task_done` / `critical_fact_pending`，并检查 `resp.ok`。关键事实确认条打 `POST /api/profile/critical-facts/{confirm,revoke}`。没有单独的问卷页；引导问题走 `/api/chat` 的 token 流。

---

## 0. 共同前提

- Docker Desktop（走 Docker 路径）或本机 Postgres + Python 3.11 + Node 22（走本地路径）
- 项目根目录有一份 `.env`（至少填 LLM 相关变量才能真的聊天；不填的话 healthz 照样通，`/api/chat` 会报错）

```bash
cp .env.example .env
```

最少要配通 **一种** LLM：

```bash
# 方案 A：本机 Ollama（免费，不用 API key）
#   brew install ollama && ollama pull qwen3:0.6b
LLM_PROVIDER_DEV=ollama
LLM_MODEL_DEV=qwen3:0.6b
MODEL_TIER=dev

# 方案 B：Anthropic / OpenAI（要 key）
# LLM_PROVIDER_DEV=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
# 或
# LLM_PROVIDER_DEV=openai
# OPENAI_API_KEY=sk-...

# 方案 C：OpenRouter（一个 key 转发到几十家上游模型，按量计费/部分模型免费）
# LLM_PROVIDER_DEV=openrouter
# OPENROUTER_API_KEY=sk-or-...
# LLM_MODEL_DEV=anthropic/claude-haiku-4.5   # 模型名去 https://openrouter.ai/models 查
```

完整说明在 `.env.example`。`MODEL_TIER=dev|prod` 切双档，不用改代码。

Langfuse 全链路是可选的。不会用、没配过：见 **[docs/LANGFUSE.md](./docs/LANGFUSE.md)**（从零解释 trace/span、怎么注册、怎么用 `trace_id` 在网页上打开一条请求）。不配密钥时聊天照跑，只是不上报。

Ollama 若跑在**宿主机**、API 跑在**容器**里，compose 已经把 `OLLAMA_BASE_URL` 指到 `http://host.docker.internal:11434/v1`。容器里的 `localhost` 是容器自己，不是你的 Mac。

---

## 1. Docker 一键起

在项目根目录：

```bash
cp .env.example .env   # 第一次；填 LLM 相关变量
docker compose up --build
```

打开 http://localhost:3000。后台跑加 `-d`：

```bash
docker compose up --build -d
```

启动顺序：Postgres → 建表 → 灌 40 条 `conflict_rules` → API → 前端。

**知识库向量不在这条路径上。** 空库时 `/healthz` 仍是 200，但聊天会因为没有 `source_id` 被核查拦截——这是设计如此（ENGINEERING §9：ingest 不挡 90s 冷启动）。要能真正检索，见下面「灌知识库」。

### 1.1 改了代码怎么重新部署

**不会自动部署。** 源码在 `docker build` 时 `COPY` 进镜像，容器里也没有 `--reload`。

```bash
# 日常：重建镜像并重启，保留数据库
docker compose up --build -d

# 只改了后端（api/、backend/）
docker compose up --build -d api

# 只改了前端（frontend/）
docker compose up --build -d frontend
```

然后浏览器硬刷新 http://localhost:3000（Cmd+Shift+R），避免旧 JS 缓存。

不要用这条当日常重部署：

```bash
docker compose down -v && docker compose up --build -d
```

`-v` 会删掉 `pgdata`（库变空）和 `hf_cache`（BGE-M3 权重可能要重下）。那是「干净环境验证冷启动」用的。

| 你以为 | 实际 |
|---|---|
| `docker compose up -d` | 用的还是旧镜像，代码改了也不生效 |
| 改 Python 保存即热重载 | 容器里 uvicorn 没有 `--reload`，源码也不在 volume 里 |
| 改 Next 页面马上看到 | 前端是 `npm run build` 的生产镜像，必须 `--build` |
| 改 `db/schema.sql` 自动迁表 | 只在 Postgres **第一次**初始化（数据目录为空）时执行；卷还在就要自己迁 |
| 改 `evals/conflict_rules.jsonl` | seed 是 bind-mount，重跑即可：`docker compose up seed` |

看状态 / 停掉：

```bash
docker compose ps
docker compose logs -f api
docker compose down          # 停容器，不删数据
```

### 1.2 灌知识库（检索要用）

知识源是 `knowledge/_processed/*.jsonl`。`knowledge/` 在 `.gitignore` 里，干净 clone 没有这份数据，ingest 会诚实失败。

本机已有同结构的 `knowledge_chunks` 时，拷进容器最快（compose 把 Postgres 映到宿主机 **5433**，避开本机常见的 5432）：

```bash
pg_dump -d diet_expert -t knowledge_chunks --data-only --no-owner \
  | docker exec -i diet_expert-postgres-1 psql -U diet_expert -d diet_expert
```

没有现成库、要在容器里重新嵌入（CPU，5837 条是**小时级**，不是 <10 分钟那条指标——那条默认宿主机 MPS/GPU）：

```bash
docker compose --profile ingest up ingest
```

或在宿主机对着容器库嵌入（发布端口 5433）：

```bash
DIET_EXPERT_PG_DSN=postgresql://diet_expert:diet_expert@127.0.0.1:5433/diet_expert \
  python3 db/embed_bge_m3.py load --root .
```

### 1.3 端口

| 服务 | 容器内 | 宿主机 |
|---|---|---|
| 前端 | 3000 | **3000** |
| API | 8000 | **8123** |
| Postgres | 5432 | **5433** |

从宿主机连容器 Postgres 用 `localhost:5433`，不要用 5432——这台机器（以及很多本机 Postgres）已经占了 5432，连错库不会报「连不上」，只会查到另一份数据。

---

## 2. 本地开发

改代码想立刻看到效果，走这条。Postgres 仍建议用 Docker 起（本机装 pgvector 很烦）；API 和前端跑在宿主机上。

### 2.1 Postgres

只要数据库，不要整套 compose：

```bash
docker compose up -d postgres
```

第一次会自动执行 `db/schema.sql`。之后：

```bash
# .env 里改成容器映射端口
DIET_EXPERT_PG_DSN=postgresql://diet_expert:diet_expert@127.0.0.1:5433/diet_expert
```

如果你坚持用本机 Postgres：需要 **pgvector** 扩展（官方 `postgres` 镜像没有，`CREATE EXTENSION vector` 会失败）。建库后手动：

```bash
psql "$DIET_EXPERT_PG_DSN" -f db/schema.sql
```

### 2.2 Python 依赖 + 灌规则表

在项目根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 db/load_conflict_rules.py
```

`load_conflict_rules.py` 幂等：JSONL 是源头，每次跑都让表和 `evals/conflict_rules.jsonl` 完全一致。

可选：灌检索向量（要有 `knowledge/_processed/`）：

```bash
python3 db/embed_bge_m3.py load --root .
# 冒烟：python3 db/embed_bge_m3.py load --root . --limit 50
```

### 2.3 起 API

仍在项目根目录（不要进 `api/`，导入路径依赖仓库根）：

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8123
```

`--reload` 改 `api/`、`backend/` 会自动重启。本地默认 `/healthz` 只检查进程存活；compose 里才设 `HEALTHZ_CHECK_DB=1` 去查库。

```bash
curl -s http://127.0.0.1:8123/healthz
```

### 2.4 起前端

另开一个终端：

```bash
cd frontend
cp .env.local.example .env.local   # 默认已指向 http://localhost:8123
npm install
npm run dev
```

打开 http://localhost:3000。换 API 地址改 `frontend/.env.local` 里的 `NEXT_PUBLIC_API_BASE_URL`。

`NEXT_PUBLIC_*` 是构建期变量。Docker 镜像里已经烧成 `http://localhost:8123`，compose 里再传环境变量不会生效。本地 `next dev` 会读 `.env.local`。

### 2.5 停掉

Ctrl+C 停 uvicorn / `next dev`。Postgres 容器要停的话：

```bash
docker compose stop postgres
```

---

## 3. 怎么确认真的能聊

1. `curl -s http://localhost:8123/healthz` 返回 `{"status":"ok"}`（或带 DB 检查的等价 200）
2. 打开 http://localhost:3000，问一句事实类问题，例如「红枣是什么性味？」
3. 知识库已灌时：回答里应出现 `source_id` 标签（如 `tcm_002541`），且引用事件在正文 token 之前
4. 知识库未灌时：核查会因缺少 `source_id` 拦截——先去灌库，不是前端坏了

没有配 LLM key / Ollama 没起来：healthz 仍可能 200，聊天请求会失败。

---

## 4. 测试（可选）

```bash
pip install pytest httpx
pytest tests/unit -q
pytest tests/integration/test_api_chat_sse.py -q
```

集成测试用 mock LLM，不打真实网络、不烧 token。
