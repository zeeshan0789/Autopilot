import sqlite3

import pytest

from app.sqlbench.database import ORDERS, PRODUCTS, TABLES, build_database, readonly_uri, schema_text
from app.sqlbench.safety import extract_sql, run_query


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_database(tmp_path_factory.mktemp("shop") / "shop.db")


def count(db, table):
    connection = sqlite3.connect(readonly_uri(db), uri=True)
    try:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        connection.close()


# ---------- The database ----------


def test_the_database_has_the_expected_data(db):
    assert (count(db, "customers"), count(db, "products"), count(db, "orders")) == (40, PRODUCTS, ORDERS)
    assert count(db, "order_items") > ORDERS
    connection = sqlite3.connect(readonly_uri(db), uri=True)
    try:
        never = connection.execute("SELECT COUNT(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)").fetchone()[0]
        assert never == 5
        names = [r[0] for r in connection.execute("SELECT name FROM customers")]
        assert len(set(names)) == len(names)
    finally:
        connection.close()


def test_the_database_is_the_same_every_time(tmp_path):
    first = build_database(tmp_path / "a.db").read_bytes()
    second = build_database(tmp_path / "b.db").read_bytes()
    assert first == second


def test_the_plain_schema_has_only_create_statements(db):
    text = schema_text(db)
    assert all(f"CREATE TABLE {t}" in text for t in TABLES)
    assert "Example rows" not in text


def test_the_rich_schema_adds_example_rows_and_values(db):
    text = schema_text(db, rich=True)
    assert "-- Example rows" in text and "-- customers: (1, " in text
    assert "-- orders.status is one of: 'cancelled', 'delivered', 'pending', 'shipped'" in text
    assert "-- products.category is one of: 'camera'" in text


# ---------- Extracting SQL from a reply ----------


@pytest.mark.parametrize(
    "reply,sql",
    [
        ("SELECT COUNT(*) FROM customers", "SELECT COUNT(*) FROM customers"),
        ("select 1;", "select 1"),
        ("```sql\nSELECT name FROM customers;\n```", "SELECT name FROM customers"),
        ("```\nWITH t AS (SELECT 1) SELECT * FROM t\n```\nThis counts them.", "WITH t AS (SELECT 1) SELECT * FROM t"),
        ("Here is the query: SELECT 1 FROM orders", "SELECT 1 FROM orders"),
        ("  SELECT *\n  FROM orders  ", "SELECT *\n  FROM orders"),
        ("DROP TABLE customers;", "DROP TABLE customers"),
    ],
)
def test_the_sql_is_found_in_a_reply(reply, sql):
    extraction = extract_sql(reply)
    assert extraction.sql == sql and extraction.declined is False


@pytest.mark.parametrize("reply", ["CANNOT_ANSWER", "cannot_answer.", "CANNOT_ANSWER, the schema has no ratings"])
def test_declining_is_recognized(reply):
    extraction = extract_sql(reply)
    assert (extraction.sql, extraction.declined) == (None, True)


@pytest.mark.parametrize("reply", ["", "   ", "I am not sure what you mean."])
def test_a_reply_without_sql_gives_none(reply):
    extraction = extract_sql(reply)
    assert (extraction.sql, extraction.declined) == (None, False)


# ---------- The safety layer ----------


def test_a_select_runs(db):
    result = run_query(db, "SELECT COUNT(*) FROM customers")
    assert result.ok and result.rows == [(40,)] and result.columns == 1 and not result.truncated


def test_a_with_query_runs(db):
    assert run_query(db, "WITH t AS (SELECT 1 AS x) SELECT * FROM t").rows == [(1,)]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE customers",
        "UPDATE products SET price = 1",
        "INSERT INTO customers VALUES (99, 'x', 'Tokyo', 'Japan', 2020)",
        "PRAGMA table_info(customers)",
        "ATTACH DATABASE 'other.db' AS other",
        "SELECT 1; DROP TABLE orders",
        "SELECT * FROM sqlite_master",
        "SELECT randomblob(10)",
    ],
)
def test_anything_that_is_not_a_plain_query_is_refused(db, sql):
    result = run_query(db, sql)
    assert not result.ok and result.kind == "blocked"


def test_refused_statements_change_nothing(db):
    for sql in ("DELETE FROM orders", "DROP TABLE customers", "SELECT 1; DELETE FROM orders"):
        run_query(db, sql)
    assert count(db, "orders") == ORDERS and count(db, "customers") == 40


def test_a_runaway_query_is_stopped(db):
    result = run_query(db, "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r", timeout_seconds=0.2)
    assert not result.ok and result.kind == "timeout"


def test_the_number_of_rows_is_limited(db):
    result = run_query(db, "SELECT * FROM order_items", max_rows=10)
    assert result.ok and len(result.rows) == 10 and result.truncated


def test_syntax_and_unknown_names_are_told_apart(db):
    assert run_query(db, "SELECT FROM customers").kind == "syntax"
    unknown = run_query(db, "SELECT nope FROM customers")
    assert unknown.kind == "execution" and "no such column" in unknown.error.lower()
    assert "no such table" in run_query(db, "SELECT * FROM suppliers").error.lower()