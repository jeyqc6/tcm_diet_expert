#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 knowledge/_raw/nutrition/recipe_xiachufang.json（XiaChuFang Recipe Corpus，JSONL，约 155 万行）
灌进 Postgres 的 recipes 表（见 db/schema.sql），支撑 D24："按食材查菜谱"用 SQL 精确/包含过滤，
不走 RAG 语义检索。

为什么不是逐行 INSERT：155 万行如果一行一次 INSERT + 一次网络往返，光是往返延迟就可能跑几个小时。
这里用 psycopg2.extras.execute_values 做批量插入（每批默认 2000 行拼成一条多值 INSERT），
比逐行 INSERT 快一到两个数量级；比 COPY 慢一点，但不用手写 Postgres 数组的 COPY 文本转义格式，
出错概率低很多——数据量到了百万行往上、或者需要反复重跑做 ETL，再换 COPY 也不迟。

用法：
    export DIET_EXPERT_PG_DSN="postgresql://user:pass@localhost:5432/diet_expert"
    python3 db/load_recipes.py --raw knowledge/_raw/nutrition/recipe_xiachufang.json

    # 先跑一个小批量验证没问题，再灌全量：
    python3 db/load_recipes.py --raw knowledge/_raw/nutrition/recipe_xiachufang.json --limit 2000

    # 重新导入前先清空这张表里同一个 source 的旧数据（幂等）：
    python3 db/load_recipes.py --raw knowledge/_raw/nutrition/recipe_xiachufang.json --truncate

依赖：
    pip install psycopg2-binary
"""
import argparse
import json
import os
import sys
import time

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None


def iter_records(path, limit=None):
    """逐行读 JSONL。这份文件本来就是一行一个 JSON 对象，不是 ingest.py 里
    FDC json 那种"整个文件是一个大数组"的格式，所以不需要那套流式 raw_decode 游标逻辑，
    也就不会有那个 O(n²) 问题——普通的按行迭代就是 O(n) 的。
    """
    n = 0
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            n += 1
            if limit and n >= limit:
                break
    if bad:
        print(f"  [警告] 跳过了 {bad} 行无法解析的 JSON", file=sys.stderr)


def to_row(row_idx, r):
    name = (r.get("name") or "").strip()
    dish = (r.get("dish") or "").strip() or None
    if dish == "Unknown":
        dish = None
    description = (r.get("description") or "").strip() or None
    ingredients = r.get("recipeIngredient") or []
    instructions = r.get("recipeInstructions") or []
    author = (r.get("author") or "").strip() or None
    return (row_idx, name, dish, description, ingredients, instructions, author)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="recipe_xiachufang.json 路径")
    ap.add_argument("--dsn", default=os.environ.get("DIET_EXPERT_PG_DSN"),
                     help="Postgres 连接串；不传就读环境变量 DIET_EXPERT_PG_DSN")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="只导入前 N 条，调试用")
    ap.add_argument("--truncate", action="store_true", help="导入前先清空表里 source='XiaChuFang Recipe Corpus' 的旧数据")
    args = ap.parse_args()

    if psycopg2 is None:
        print("需要 psycopg2：pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)
    if not args.dsn:
        print("没有连接串。传 --dsn，或者先 export DIET_EXPERT_PG_DSN=...", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.raw):
        print(f"找不到文件：{args.raw}", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    if args.truncate:
        cur.execute("DELETE FROM recipes WHERE source = %s", ("XiaChuFang Recipe Corpus",))
        print(f"  已清空旧数据：{cur.rowcount} 行")
        conn.commit()

    insert_sql = """
        INSERT INTO recipes (source_row, name, dish, description, ingredients, instructions, author)
        VALUES %s
    """

    batch = []
    total = 0
    t0 = time.time()
    for row_idx, r in iter_records(args.raw, limit=args.limit):
        batch.append(to_row(row_idx, r))
        if len(batch) >= args.batch_size:
            psycopg2.extras.execute_values(cur, insert_sql, batch, page_size=args.batch_size)
            conn.commit()
            total += len(batch)
            batch = []
            elapsed = time.time() - t0
            print(f"  已导入 {total} 条 ({total/elapsed:.0f} 条/秒)", end="\r", file=sys.stderr)

    if batch:
        psycopg2.extras.execute_values(cur, insert_sql, batch, page_size=args.batch_size)
        conn.commit()
        total += len(batch)

    cur.close()
    conn.close()
    print(f"\n完成。共导入 {total} 条，用时 {time.time()-t0:.1f}s")
    print("验证一下：SELECT count(*) FROM recipes; "
          "SELECT name, ingredients FROM recipes WHERE ingredients @> ARRAY['山药'] LIMIT 5;")


if __name__ == "__main__":
    main()
