"""
测试目标：backend/mcp_server/tools/_retrieval_common.py 的 build_filter_sql()
——混合检索(结构化预筛)的 SQL 拼接逻辑，纯函数，不需要数据库/模型。
对应实现：backend/mcp_server/tools/_retrieval_common.py
"""
from backend.mcp_server.tools._retrieval_common import build_filter_sql


def test_no_filters_returns_empty():
    sql, params = build_filter_sql(None)
    assert sql == ""
    assert params == []

    sql, params = build_filter_sql({})
    assert sql == ""
    assert params == []


def test_source_type_string():
    sql, params = build_filter_sql({"source_type": "md_table_row"})
    assert "source_type = ANY(%s)" in sql
    assert params == [["md_table_row"]]


def test_source_type_list():
    sql, params = build_filter_sql({"source_type": ["md_table_row", "pdf"]})
    assert params == [["md_table_row", "pdf"]]


def test_metadata_equality_filter():
    sql, params = build_filter_sql({"代码": "qi_xu"})
    assert "metadata @> %s::jsonb" in sql
    assert params == ['{"代码": "qi_xu"}']


def test_combined_source_type_and_metadata():
    sql, params = build_filter_sql({"source_type": "md_table_row", "代码": "qi_xu"})
    assert "source_type = ANY(%s)" in sql
    assert "metadata @> %s::jsonb" in sql
    assert sql.count(" AND ") == 2  # 一条接在 WHERE domain=%s 后面，一条连接两个子句
    assert params == [["md_table_row"], '{"代码": "qi_xu"}']


def test_multiple_metadata_keys_merge_into_one_jsonb_param():
    sql, params = build_filter_sql({"代码": "qi_xu", "体质": "气虚质"})
    assert sql.count("metadata @>") == 1
    assert len(params) == 1


def test_none_values_are_ignored():
    sql, params = build_filter_sql({"代码": None})
    assert sql == ""
    assert params == []

    sql, params = build_filter_sql({"source_type": None, "代码": "qi_xu"})
    assert "source_type" not in sql
    assert "metadata @>" in sql
