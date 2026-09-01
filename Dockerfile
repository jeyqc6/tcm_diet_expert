# diet_expert · API 服务镜像（ENGINEERING §9，roadmap 阶段4.2 "docker compose up + healthz"）
#
# ⚠️ 诚实说明镜像为什么不小：backend/mcp_server/tools/_retrieval_common.py 的
# retrieve_tcm/retrieve_nutrition 在**请求时**（不是只在离线 ingest 时）要用
# BGE-M3 给用户的 query 现算一个向量，所以 torch/FlagEmbedding 是这个服务的
# 运行时依赖，不是只有 db/embed_bge_m3.py 那次性摄入脚本才用得到——不能因为
# "看起来是数据处理脚本的依赖"就从这个镜像里拿掉，拿掉检索功能直接就是坏的。
# 代价是镜像大、且首次启动会去 Hugging Face 下载 BGE-M3 权重（约 1GB+），
# docker-compose.yml 里给了一个持久化的 HF 缓存卷缓解"每次 down 再 up 都要
# 重新下载"，但"删掉本地卷重来一遍"这条完成判据里的卷如果连缓存卷一起删，
# 第一次启动多花的下载时间是网络带宽决定的，不是这个 Dockerfile 能优化掉的。

FROM python:3.11-slim

WORKDIR /app

# psycopg2-binary 已经是预编译 wheel，理论上不需要 libpq-dev/gcc；保留是因为
# torch/FlagEmbedding 这类科学计算依赖有时会退化到从源码编译某些子依赖，
# 缺编译工具链会在毫无征兆的地方报错，这几十 MB 换的是"装依赖这一步不会
# 因为环境差异莫名其妙失败"。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 只装 API 服务真正需要的子集，不是整份 requirements.txt——pdfplumber/
# beautifulsoup4/scikit-learn 这些是 planning/step1-naive-rag 离线原型和
# ingest.py 用的，这个长期运行的服务容器不需要它们。
RUN pip install --no-cache-dir \
    fastapi uvicorn python-dotenv \
    psycopg2-binary pgvector \
    FlagEmbedding torch \
    openai anthropic \
    "langfuse>=3.0,<4"

COPY backend/ backend/
COPY api/ api/
COPY db/ db/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
