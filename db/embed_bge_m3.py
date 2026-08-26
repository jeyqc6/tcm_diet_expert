#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 BGE-M3 给 knowledge/_processed/*_chunks.jsonl 做 dense embedding，写入 Postgres + pgvector。

对应：
  docs/RAG_PIPELINE_DESIGN.md §三（开发期用 BGE-M3）
  docs/DECISIONS.md D2 / D4 / D23
  db/schema.sql · knowledge_chunks

用法：
    # 1) 建表（只需一次）
    psql \"$DIET_EXPERT_PG_DSN\" -f db/schema.sql

    # 2) 依赖
    pip install FlagEmbedding psycopg2-binary pgvector torch

    # 3) 嵌入并入库（默认读 knowledge/_processed/）
    export DIET_EXPERT_PG_DSN=\"postgresql://user:pass@localhost:5432/diet_expert\"
    python3 db/embed_bge_m3.py load --root .

    # 只跑前 N 条做冒烟测试：
    python3 db/embed_bge_m3.py load --root . --limit 50

    # 只灌某一个 domain：
    python3 db/embed_bge_m3.py load --root . --domain tcm

    # 4) 试检索
    python3 db/embed_bge_m3.py search --query \"气虚质适合吃什么\" --domain tcm --top-k 5

说明：
  - BGE-M3 dense 维数 = 1024；写入前 L2 归一化，检索用 cosine 距离 `<=>`
  - 同一 chunk_id 重复跑会 UPSERT 覆盖向量（幂等）
  - 交付期若切 Voyage，换 embed_model + 向量维度后另开表或迁移，不要和 1024 维混存
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None

MODEL_ID = "BAAI/bge-m3"
EMBED_DIM = 1024
DEFAULT_BATCH = 16  # M3 较大，本地 CPU/小显存先保守一点


def require_pg():
    if psycopg2 is None:
        print("需要 psycopg2：pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)
    if register_vector is None:
        print("需要 pgvector：pip install pgvector", file=sys.stderr)
        sys.exit(1)


def connect(dsn: str):
    conn = psycopg2.connect(dsn)
    register_vector(conn)
    return conn


def load_chunks(paths: list[Path], limit: int | None = None) -> list[dict]:
    out = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("text") or not rec.get("chunk_id"):
                    continue
                out.append(rec)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def resolve_chunk_files(root: Path, domain: str | None) -> list[Path]:
    processed = root / "knowledge" / "_processed"
    mapping = {
        "tcm": processed / "tcm_chunks.jsonl",
        "nutrition": processed / "nutrition_chunks.jsonl",
    }
    if domain:
        path = mapping[domain]
        if not path.exists():
            print(f"找不到 {path}，先跑 ingest.py", file=sys.stderr)
            sys.exit(1)
        return [path]
    missing = [p for p in mapping.values() if not p.exists()]
    if missing:
        print("缺少 chunk 文件：", ", ".join(str(p) for p in missing), file=sys.stderr)
        print("先跑：python3 planning/step1-naive-rag/ingest.py --root .", file=sys.stderr)
        sys.exit(1)
    return list(mapping.values())


class BgeM3Embedder:
    """Thin wrapper so load/search share one model instance."""

    def __init__(self, model_id: str = MODEL_ID, use_fp16: bool = True):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            print(
                "需要 FlagEmbedding：pip install FlagEmbedding torch",
                file=sys.stderr,
            )
            sys.exit(1)
        # Devices: CUDA → fp16 ok; Apple MPS / CPU → fp16 可能不稳，交给库自己处理
        self.model = BGEM3FlagModel(model_id, use_fp16=use_fp16)
        self.model_id = model_id

    def encode(self, texts: list[str], batch_size: int = DEFAULT_BATCH) -> np.ndarray:
        """Return L2-normalized dense vectors, shape (n, 1024)."""
        # BGE-M3: return_dense=True is enough for pgvector; sparse/colbert 另存
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"unexpected embedding shape {arr.shape}, expected (*, {EMBED_DIM})"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return arr / norms


def upsert_batch(cur, rows: list[tuple]):
    """rows: (chunk_id, domain, source_file, source_type, text, metadata_json, embedding, embed_model)"""
    sql = """
        INSERT INTO knowledge_chunks
            (chunk_id, domain, source_file, source_type, text, metadata, embedding, embed_model)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
            domain = EXCLUDED.domain,
            source_file = EXCLUDED.source_file,
            source_type = EXCLUDED.source_type,
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            embed_model = EXCLUDED.embed_model
    """
    psycopg2.extras.execute_values(cur, sql, rows, page_size=len(rows))


def cmd_load(args):
    require_pg()
    if not args.dsn:
        print("没有连接串。传 --dsn 或 export DIET_EXPERT_PG_DSN=...", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root).resolve()
    paths = resolve_chunk_files(root, args.domain)
    chunks = load_chunks(paths, limit=args.limit)
    if not chunks:
        print("没有可读的 chunk", file=sys.stderr)
        sys.exit(1)
    print(f"准备嵌入 {len(chunks)} 条 · model={MODEL_ID} · dim={EMBED_DIM}")

    embedder = BgeM3Embedder(use_fp16=not args.no_fp16)
    conn = connect(args.dsn)
    cur = conn.cursor()

    t0 = time.time()
    written = 0
    batch_size = args.batch_size
    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            vecs = embedder.encode(texts, batch_size=batch_size)
            rows = []
            for c, v in zip(batch, vecs):
                meta = c.get("metadata") or {}
                rows.append(
                    (
                        c["chunk_id"],
                        c["domain"],
                        c["source_file"],
                        c.get("source_type"),
                        c["text"],
                        psycopg2.extras.Json(meta),
                        v.tolist(),
                        MODEL_ID,
                    )
                )
            upsert_batch(cur, rows)
            conn.commit()
            written += len(rows)
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            print(
                f"  upserted {written}/{len(chunks)} "
                f"({rate:.1f} chunks/s, last batch {len(rows)})"
            )
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    print(f"完成：写入/更新 {written} 条到 knowledge_chunks，用时 {time.time() - t0:.1f}s")


def cmd_search(args):
    require_pg()
    if not args.dsn:
        print("没有连接串。传 --dsn 或 export DIET_EXPERT_PG_DSN=...", file=sys.stderr)
        sys.exit(1)
    if not args.query:
        print("search 需要 --query", file=sys.stderr)
        sys.exit(1)

    embedder = BgeM3Embedder(use_fp16=not args.no_fp16)
    qvec = embedder.encode([args.query])[0]

    conn = connect(args.dsn)
    cur = conn.cursor()
    sql = """
        SELECT chunk_id, domain, source_file,
               left(text, 160) AS preview,
               1 - (embedding <=> %s::vector) AS score
        FROM knowledge_chunks
        WHERE (%s::text IS NULL OR domain = %s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    cur.execute(sql, (qvec.tolist(), args.domain, args.domain, qvec.tolist(), args.top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("没有命中（表是否为空？先跑 load）")
        return
    print(f"query: {args.query!r}\n")
    for i, (chunk_id, domain, source_file, preview, score) in enumerate(rows, 1):
        print(f"{i}. [{domain}] score={float(score):.4f}  {chunk_id}  ← {source_file}")
        print(f"   {preview.replace(chr(10), ' ')}")
        print()


def main():
    ap = argparse.ArgumentParser(description="BGE-M3 → Postgres/pgvector")
    ap.add_argument(
        "--dsn",
        default=os.environ.get("DIET_EXPERT_PG_DSN"),
        help="Postgres DSN；默认读 DIET_EXPERT_PG_DSN",
    )
    ap.add_argument(
        "--no-fp16",
        action="store_true",
        help="关闭 fp16（CPU / 部分 MPS 环境更稳）",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load", help="嵌入 chunk JSONL 并 UPSERT 进 knowledge_chunks")
    p_load.add_argument("--root", default=".", help="项目根目录")
    p_load.add_argument("--domain", choices=["tcm", "nutrition"], default=None)
    p_load.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    p_load.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p_load.set_defaults(func=cmd_load)

    p_search = sub.add_parser("search", help="用同一模型编码 query，做余弦近邻检索")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--domain", choices=["tcm", "nutrition"], default=None)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
