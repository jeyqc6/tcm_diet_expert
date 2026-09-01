from types import SimpleNamespace

import pytest

import backend.mcp_server.tools.query_recipes as recipes


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = None
        self.params = None
        self.closed = False

    def execute(self, sql, params):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def test_query_recipes_uses_all_ingredient_operator(monkeypatch):
    cursor = FakeCursor(
        [
            (
                1,
                "番茄炒蛋",
                "家常菜",
                "简单快手",
                ["番茄", "鸡蛋"],
                ["炒熟"],
                "作者",
                "测试来源",
            )
        ]
    )
    connection = FakeConnection(cursor)
    monkeypatch.setattr(recipes, "get_pg_dsn", lambda dsn: "test-dsn")
    monkeypatch.setattr(recipes, "psycopg2", SimpleNamespace(connect=lambda dsn: connection))

    result = recipes.query_recipes_by_ingredients(["番茄", " 鸡蛋 ", "番茄"], dsn="ignored")

    assert result[0]["name"] == "番茄炒蛋"
    assert "ingredients @>" in cursor.sql
    assert cursor.params == (["番茄", "鸡蛋"], 20)
    assert connection.closed is True
    assert cursor.closed is True


def test_query_recipes_uses_any_ingredient_operator(monkeypatch):
    cursor = FakeCursor([])
    connection = FakeConnection(cursor)
    monkeypatch.setattr(recipes, "get_pg_dsn", lambda dsn: "test-dsn")
    monkeypatch.setattr(recipes, "psycopg2", SimpleNamespace(connect=lambda dsn: connection))

    result = recipes.query_recipes_by_ingredients(["番茄"], match="any", limit=3)

    assert result == []
    assert "ingredients &&" in cursor.sql
    assert cursor.params == (["番茄"], 3)


@pytest.mark.parametrize(
    ("ingredients", "match", "limit", "message"),
    [
        ([], "all", 20, "at least one"),
        (["番茄"], "unknown", 20, "any.*all"),
        (["番茄"], "all", 0, "positive integer"),
        ([1], "all", 20, "only strings"),
    ],
)
def test_query_recipes_rejects_invalid_arguments(ingredients, match, limit, message):
    with pytest.raises(ValueError, match=message):
        recipes.query_recipes_by_ingredients(ingredients, match=match, limit=limit)
