#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 evals/conflict_rules.jsonl 灌进 Postgres 的 conflict_rules 表（见 db/schema.sql）。

设计依据：docs/ARCHITECTURE.md §1.2、§5.2 步骤 5（调和层按体质/目标结构化查询命中的规则）
参考：docs/ENGINEERING.md §4.1（幂等 ingest）、docs/DECISIONS.md D23（关系表建模，
不把 40 条规则整份塞进 prompt）。

JSONL 是人工编辑的源文件（diff 友好），表是查询用的派生数据——查询走表，编辑永远走 JSONL。
既然 JSONL 是唯一源头，ingest 采用**全量同步**而不是只追加：每次运行都让表的内容跟 JSONL
这一刻的内容完全一致（多的删掉、少的补上、改过的更新），而不是"只增不减"。这也是本脚本
能满足"conflict_rules 表数据和 jsonl 条数对得上"这条验收标准的原因——只做 UPSERT 不做
删除的话，一旦 JSONL 里删过某条规则（比如发现来源有问题下线），表里会留着一条不存在于源文件
的幽灵数据，条数永远对不上。

全量同步 + 全部 40 条一批写入，在一个事务里做完（要么全部生效，要么全部不生效），
不像 load_recipes.py 那样分批 commit——那边是 155 万行需要分批控制内存/单批失败范围，
这里只有几十行，没有必要引入"部分导入成功"这种中间状态。

用法：
    export DIET_EXPERT_PG_DSN="postgresql://user:pass@localhost:5432/diet_expert"
    python3 db/load_conflict_rules.py

    # 或者放进 .env，脚本会自动读（backend/env.py）：
    python3 db/load_conflict_rules.py --jsonl evals/conflict_rules.jsonl

    # 只校验 JSONL 本身格式对不对，不连数据库：
    python3 db/load_conflict_rules.py --dry-run

依赖：
    pip install psycopg2-binary python-dotenv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

# 允许脚本以 `python3 db/load_conflict_rules.py` 从仓库根目录直接运行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.env import get_pg_dsn  # noqa: E402

# 与 db/schema.sql 里 conflict_rules 表的 CHECK 约束保持一致——客户端先校验一遍，
# 是为了报错时能直接指出"哪一条规则、哪一个字段错了"，而不是让用户去看一段
# Postgres 的 CHECK violation 报错再自己倒推是第几行。校验规则不重复定义业务含义，
# 只是把 schema.sql 里已经写死的取值范围抄一遍，两边如果以后改了枚举值要同步改。
_VALID_RELATIONS = {
    "conflict", "partial_conflict", "conditional_conflict",
    "aligned", "aligned_negative", "complementary",
    "tcm_internal", "nutrition_internal",
}
_VALID_CONFIDENCE = {"high", "medium", "low", None}
_VALID_SOURCE_STATUS = {"verified", "needs_source"}

# INSERT ... ON CONFLICT (rule_id) DO UPDATE 涉及的列，顺序固定，供 execute_values 和
# DO UPDATE 子句共用，避免两处列名各写一遍、以后漏改一处。
_COLUMNS = (
    "rule_id", "line", "topic",
    "tcm_position", "tcm_source",
    "nutrition_position", "nutrition_source",
    "relation", "resolution", "resolution_rationale",
    "confidence", "evidence_level",
    "applicable_constitutions", "applicable_goals",
    "source_status",
)


def iter_records(path: Path):
    """逐行读 JSONL，每行一个规则对象。"""
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield lineno, json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {lineno} 行不是合法 JSON：{e}") from e


def validate(lineno: int, r: dict) -> list[str]:
    """返回这一条记录的错误列表；空列表表示通过。只校验"进库会炸"或
    "会破坏调和层查询前提"的字段，不做过度校验（比如不检查 resolution_rationale
    是否为空话——那是 evals/README.md §六 编辑清单该管的内容审校，不是 ingest 脚本的职责）。
    """
    errors = []
    rule_id = r.get("rule_id")
    if not rule_id:
        errors.append(f"第 {lineno} 行缺少 rule_id")
    for required in ("topic", "tcm_position", "nutrition_position", "relation", "source_status"):
        if not r.get(required):
            errors.append(f"{rule_id or f'第{lineno}行'}: 缺少必填字段 {required}")
    if r.get("relation") not in _VALID_RELATIONS:
        errors.append(f"{rule_id}: relation={r.get('relation')!r} 不在允许的 8 种关系类型内")
    if r.get("confidence") not in _VALID_CONFIDENCE:
        errors.append(f"{rule_id}: confidence={r.get('confidence')!r} 必须是 high/medium/low 或缺省")
    if r.get("source_status") not in _VALID_SOURCE_STATUS:
        errors.append(f"{rule_id}: source_status={r.get('source_status')!r} 必须是 verified/needs_source")
    for arr_field in ("applicable_constitutions", "applicable_goals"):
        val = r.get(arr_field, [])
        if not isinstance(val, list):
            errors.append(f"{rule_id}: {arr_field} 必须是数组（空数组表示不限），拿到的是 {type(val).__name__}")
    return errors


def to_row(r: dict) -> tuple:
    return (
        r["rule_id"],
        r.get("line"),
        r["topic"],
        r["tcm_position"],
        r.get("tcm_source"),
        r["nutrition_position"],
        r.get("nutrition_source"),
        r["relation"],
        r.get("resolution"),
        r.get("resolution_rationale"),
        r.get("confidence"),
        r.get("evidence_level"),
        r.get("applicable_constitutions") or [],
        r.get("applicable_goals") or [],
        r["source_status"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jsonl", default="evals/conflict_rules.jsonl", help="源文件路径")
    ap.add_argument("--dsn", default=None, help="Postgres 连接串；不传则用 backend.env.get_pg_dsn()（.env / 环境变量）")
    ap.add_argument("--dry-run", action="store_true", help="只解析 + 校验 JSONL，不连数据库、不写入")
    ap.add_argument("--no-prune", action="store_true",
                     help="只 upsert，不删除 JSONL 里已不存在的 rule_id（默认做全量同步，这个开关是为了"
                          "调试时能对比「删之前/删之后」用，正常使用不需要加）")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"找不到文件：{jsonl_path}", file=sys.stderr)
        return 1

    records = []
    all_errors = []
    seen_ids = set()
    for lineno, r in iter_records(jsonl_path):
        errors = validate(lineno, r)
        all_errors.extend(errors)
        if not errors:
            rid = r["rule_id"]
            if rid in seen_ids:
                all_errors.append(f"rule_id 重复：{rid}（第 {lineno} 行）")
            else:
                seen_ids.add(rid)
                records.append(r)

    if all_errors:
        print(f"校验失败，共 {len(all_errors)} 个问题：", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"校验通过：{jsonl_path} 共 {len(records)} 条规则。")

    if args.dry_run:
        print("（--dry-run，不连接数据库）")
        return 0

    if psycopg2 is None:
        print("需要 psycopg2：pip install psycopg2-binary", file=sys.stderr)
        return 1

    dsn = get_pg_dsn(args.dsn)
    if not dsn:
        print("没有连接串。传 --dsn，或者在 .env / 环境变量里设置 DIET_EXPERT_PG_DSN。", file=sys.stderr)
        return 1

    rows = [to_row(r) for r in records]
    current_ids = tuple(r["rule_id"] for r in records)

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        cur = conn.cursor()

        pruned = 0
        if not args.no_prune:
            cur.execute(
                "DELETE FROM conflict_rules WHERE rule_id != ALL(%s)",
                (list(current_ids),),
            )
            pruned = cur.rowcount

        update_cols = [c for c in _COLUMNS if c != "rule_id"]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        upsert_sql = f"""
            INSERT INTO conflict_rules ({", ".join(_COLUMNS)})
            VALUES %s
            ON CONFLICT (rule_id) DO UPDATE SET
                {set_clause},
                updated_at = now()
        """
        psycopg2.extras.execute_values(cur, upsert_sql, rows, page_size=len(rows) or 1)

        cur.execute("SELECT count(*) FROM conflict_rules")
        table_count = cur.fetchone()[0]

        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"完成。upsert {len(rows)} 条" + (f"，清理了 {pruned} 条 JSONL 里已不存在的旧规则" if pruned else "") + "。")

    if table_count != len(records):
        print(
            f"[警告] 表里现有 {table_count} 条，JSONL 有 {len(records)} 条，数量对不上——"
            "如果加了 --no-prune，这是预期行为；否则说明有并发写入或需要重新排查。",
            file=sys.stderr,
        )
        return 1

    print(f"验证通过：conflict_rules 表 {table_count} 条 == JSONL {len(records)} 条。")
    print("可另外手工核对：SELECT relation, count(*) FROM conflict_rules GROUP BY relation ORDER BY 2 DESC;")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
