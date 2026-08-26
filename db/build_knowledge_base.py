#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键跑通"知识源 -> chunk -> 向量 -> 入库"整条 RAG 数据管线。

只覆盖走向量检索的那部分数据(knowledge_chunks 表，tcm + nutrition 两个 domain)。
明确不包含：
  - recipes 表——D24 决定"按食材查菜谱"走 SQL 精确/包含过滤，不走向量检索，
    这批数据用 db/load_recipes.py 单独灌，不在本脚本范围内
  - conflict_rules 表——D23 决定这是结构化关系查询，不走向量检索
这两类数据"故意不向量化"是既有架构决策，不是遗漏，见 docs/ARCHITECTURE.md §1.2。

两步：
  1. planning/step1-naive-rag/ingest.py —— 把 knowledge/_raw 下的原始资料
     (JSON/JSONL/XML/MD/PDF/OCR文本)按结构切块 + token 感知二次切分，
     写入 knowledge/_processed/{tcm,nutrition}_chunks.jsonl
  2. db/embed_bge_m3.py load            —— 用 BGE-M3 给每个 chunk 做 dense embedding，
     upsert 进 Postgres 的 knowledge_chunks 表（同一 chunk_id 重复跑会覆盖，幂等）

用法：
    export DIET_EXPERT_PG_DSN="postgresql://user:pass@localhost:5432/diet_expert"
    python3 db/build_knowledge_base.py --root .

    # chunk 文件没变的话，跳过重新切块，只重新跑 embedding（更快）
    python3 db/build_knowledge_base.py --root . --skip-ingest

    # 透传给 ingest.py：额外收编菜谱/sr_legacy 大文件（默认跳过，见 ingest.py --help）
    python3 db/build_knowledge_base.py --root . --include-recipes --include-sr-legacy
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(name, cmd, env=None):
    print(f"\n{'=' * 70}\n[{name}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[{name}] 失败（退出码 {result.returncode}），用时 {elapsed:.0f}s", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[{name}] 完成，用时 {elapsed:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".", help="diet_expert 项目根目录")
    ap.add_argument("--dsn", default=os.environ.get("DIET_EXPERT_PG_DSN"),
                     help="Postgres DSN；默认读 DIET_EXPERT_PG_DSN 环境变量")
    ap.add_argument("--skip-ingest", action="store_true",
                     help="跳过切块，直接用现有 knowledge/_processed/*.jsonl 做 embedding")
    ap.add_argument("--include-recipes", action="store_true",
                     help="透传给 ingest.py：额外收编 recipe_xiachufang.json 的样本 chunk")
    ap.add_argument("--include-sr-legacy", action="store_true",
                     help="透传给 ingest.py：额外解析 210MB 的 sr_legacy 食物库")
    ap.add_argument("--batch-size", type=int, default=16, help="透传给 embed_bge_m3.py load")
    ap.add_argument("--no-fp16", action="store_true", help="透传给 embed_bge_m3.py：关闭 fp16")
    args = ap.parse_args()

    if not args.dsn:
        print("没有连接串。传 --dsn 或 export DIET_EXPERT_PG_DSN=...", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root).resolve()
    t_start = time.time()

    if not args.skip_ingest:
        ingest_script = root / "planning" / "step1-naive-rag" / "ingest.py"
        cmd = [sys.executable, str(ingest_script), "--root", str(root)]
        if args.include_recipes:
            cmd.append("--include-recipes")
        if args.include_sr_legacy:
            cmd.append("--include-sr-legacy")
        run_step("1/2 切块 (ingest.py)", cmd)
    else:
        print("[1/2 切块] 跳过，使用现有 knowledge/_processed/*.jsonl")

    embed_script = root / "db" / "embed_bge_m3.py"
    # embed_bge_m3.py 的 --dsn/--no-fp16 是定义在子命令(load/search)之前的全局参数，
    # argparse 规则要求全局参数写在子命令 token 前面——不能拼成 "load --dsn ..."。
    # 用环境变量传给子进程更省事，也正是 embed_bge_m3.py 自己文档写的用法。
    embed_env = os.environ.copy()
    embed_env["DIET_EXPERT_PG_DSN"] = args.dsn
    cmd = [
        sys.executable, str(embed_script), "load",
        "--root", str(root),
        "--batch-size", str(args.batch_size),
    ]
    if args.no_fp16:
        cmd.append("--no-fp16")
    run_step("2/2 向量化 + 入库 (embed_bge_m3.py load)", cmd, env=embed_env)

    print(f"\n全部完成，总用时 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
