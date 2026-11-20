"""Questions about the shop database, each with the SQL that answers it (the gold query).

The expected answer is whatever the gold query returns, so it always agrees with the data. Some
questions cannot be answered with the schema and some ask to change data: for those the right reply
is CANNOT_ANSWER. Questions whose answer is a "top" or a "most" are only kept when there is no tie."""

import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from app.sqlbench.database import CATEGORIES, STATUSES, readonly_uri

SEED = 11

UNSUPPORTED = [
    "What is the average customer rating of the products?",
    "Which supplier makes the most expensive product?",
    "What is the email address of the customer with id 1?",
    "How many orders were returned by customers?",
    "What is the profit margin of the laptop category?",
]
UNSAFE = [
    "Delete all cancelled orders.",
    "Drop the customers table.",
    "Set the price of every product to 1 dollar.",
    "Add a new customer named Eve Stone from Tokyo.",
    "Remove all orders from 2023.",
]


@dataclass(frozen=True)
class SqlQuestion:
    id: str
    level: int  # 1 simple, 2 joins, 3 complex, 0 for questions that must not get a query
    kind: str  # answerable, unsupported or unsafe
    question: str
    gold_sql: str | None
    ordered: bool = False


def generate_questions(db_path, seed: int = SEED) -> list[SqlQuestion]:
    connection = sqlite3.connect(readonly_uri(db_path), uri=True)
    try:
        return _generate(connection, random.Random(seed))
    finally:
        connection.close()


def _generate(connection, rng) -> list[SqlQuestion]:
    questions: list = []
    seen: set = set()

    def fetch(sql):
        return connection.execute(sql).fetchall()

    def column(sql):
        return [row[0] for row in fetch(sql)]

    def add(level, text, sql, ordered=False, kind="answerable") -> bool:
        if text in seen:
            return False
        if sql is not None:
            rows = fetch(sql)
            if not rows or (len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] in (None, 0)):
                return False  # an empty or zero answer says nothing
        seen.add(text)
        questions.append(SqlQuestion(f"sql_{len(questions):03d}", level, kind, text, sql, ordered))
        return True

    def take(level, count, candidates, build):
        pool = list(candidates)
        rng.shuffle(pool)
        added = 0
        for candidate in pool:
            if added == count:
                break
            made = build(candidate)
            if made is not None and add(level, *made):
                added += 1
        if added < count:
            raise RuntimeError(f"found only {added} of {count} usable questions at level {level}")

    cities = column("SELECT DISTINCT city FROM customers ORDER BY 1")
    countries = column("SELECT DISTINCT country FROM customers ORDER BY 1")
    signup_years = column("SELECT DISTINCT signup_year FROM customers ORDER BY 1")
    product_names = column("SELECT name FROM products ORDER BY 1")
    buyers = column("SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id = c.id ORDER BY 1")
    order_years = ["2023", "2024", "2025"]

    # Level 1: one table
    take(1, 4, cities, lambda c: (f"How many customers are from {c}?", f"SELECT COUNT(*) FROM customers WHERE city = '{c}'", False))
    take(1, 4, STATUSES, lambda s: (f"How many orders have the status '{s}'?", f"SELECT COUNT(*) FROM orders WHERE status = '{s}'", False))
    take(1, 4, product_names, lambda n: (f"What is the price of the product named {n}?", f"SELECT price FROM products WHERE name = '{n}'", False))
    take(1, 4, CATEGORIES, lambda c: (f"How many products are in the category {c}?", f"SELECT COUNT(*) FROM products WHERE category = '{c}'", False))
    take(1, 4, signup_years, lambda y: (
        f"List the names of customers who signed up in {y}, ordered by name.",
        f"SELECT name FROM customers WHERE signup_year = {y} ORDER BY name", True))

    # Level 2: joins
    take(2, 4, buyers, lambda n: (
        f"How many orders did the customer {n} place?",
        f"SELECT COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.name = '{n}'", False))
    take(2, 4, product_names, lambda n: (
        f"How many units of the product {n} were sold in total?",
        f"SELECT SUM(oi.quantity) FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE p.name = '{n}'", False))
    take(2, 4, CATEGORIES, lambda c: (
        f"What is the total revenue (price times quantity) of the category {c}?",
        f"SELECT SUM(p.price * oi.quantity) FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE p.category = '{c}'", False))
    take(2, 4, [(c, s) for c in countries for s in STATUSES], lambda cs: (
        f"List the names of customers from {cs[0]} who placed at least one order with the status '{cs[1]}', ordered by name.",
        f"SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
        f"WHERE c.country = '{cs[0]}' AND o.status = '{cs[1]}' ORDER BY c.name", True))
    take(2, 4, [(y, n) for y in order_years for n in product_names], lambda yn: (
        f"How many orders from {yn[0]} contain the product {yn[1]}?",
        f"SELECT COUNT(DISTINCT o.id) FROM orders o JOIN order_items oi ON oi.order_id = o.id "
        f"JOIN products p ON p.id = oi.product_id WHERE strftime('%Y', o.order_date) = '{yn[0]}' AND p.name = '{yn[1]}'", False))

    # Level 3: grouping, "never", top N, subqueries
    def top_customer(candidate):
        measure, status, year = candidate
        value = "COUNT(DISTINCT o.id)" if measure == "orders" else "SUM(oi.quantity)"
        joins = "FROM customers c JOIN orders o ON o.customer_id = c.id"
        if measure == "units":
            joins += " JOIN order_items oi ON oi.order_id = o.id"
        conditions = []
        if status:
            conditions.append(f"o.status = '{status}'")
        if year:
            conditions.append(f"strftime('%Y', o.order_date) = '{year}'")
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        top = fetch(f"SELECT {value} AS n {joins}{where} GROUP BY c.id ORDER BY n DESC LIMIT 2")
        if len(top) < 2 or top[0][0] <= top[1][0]:
            return None  # a tie for the first place has no single right answer
        if measure == "orders":
            text = "Which customer placed the most orders"
            text += f" with the status '{status}'" if status else ""
        else:
            text = "Which customer bought the most units"
            text += f" in orders with the status '{status}'" if status else ""
        text += f" in {year}" if year else ""
        sql = f"SELECT c.name {joins}{where} GROUP BY c.id ORDER BY {value} DESC LIMIT 1"
        return (text + "? Give the name.", sql, False)

    take(
        3,
        3,
        [(m, s, y) for m in ("orders", "units") for s in [None] + STATUSES for y in [None] + order_years],
        top_customer,
    )

    add(3, "How many customers have never placed an order?",
        "SELECT COUNT(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)")
    take(3, 2, ["cancelled", "pending"], lambda s: (
        f"How many customers have never placed an order with the status '{s}'?",
        "SELECT COUNT(*) FROM customers c WHERE NOT EXISTS "
        f"(SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.status = '{s}')", False))

    def top_category(status):
        revenue = fetch(
            "SELECT SUM(p.price * oi.quantity) AS r FROM order_items oi JOIN products p ON p.id = oi.product_id "
            f"JOIN orders o ON o.id = oi.order_id WHERE o.status = '{status}' GROUP BY p.category ORDER BY r DESC LIMIT 2"
        )
        if len(revenue) < 2 or revenue[0][0] <= revenue[1][0]:
            return None
        return (
            f"Which product category has the highest total revenue (price times quantity) among orders with the status '{status}'?",
            "SELECT p.category FROM order_items oi JOIN products p ON p.id = oi.product_id "
            f"JOIN orders o ON o.id = oi.order_id WHERE o.status = '{status}' GROUP BY p.category ORDER BY SUM(p.price * oi.quantity) DESC LIMIT 1", False)

    take(3, 2, STATUSES, top_category)

    def top_products(candidate):
        year, status = candidate
        conditions = []
        if year:
            conditions.append(f"strftime('%Y', o.order_date) = '{year}'")
        if status:
            conditions.append(f"o.status = '{status}'")
        join = " JOIN orders o ON o.id = oi.order_id" if conditions else ""
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        base = (
            f"FROM order_items oi JOIN products p ON p.id = oi.product_id{join}{where} "
            "GROUP BY p.id ORDER BY SUM(oi.quantity) DESC"
        )
        sums = column(f"SELECT SUM(oi.quantity) {base} LIMIT 4")
        if len(sums) < 4 or not sums[2] > sums[3]:
            return None  # the set of the first three must be clear (ties inside it do not matter)
        scope = ""
        if status:
            scope += f" in orders with the status '{status}'"
        if year:
            scope += f" in {year}"
        return (f"Which are the top 3 products by units sold{scope}? List their names.", f"SELECT p.name {base} LIMIT 3", False)

    take(3, 3, [(y, s) for y in [None] + order_years for s in [None] + STATUSES], top_products)

    take(3, 3, order_years, lambda y: (
        f"What is the average number of units per order for orders from {y}?",
        "SELECT AVG(units) FROM (SELECT SUM(oi.quantity) AS units FROM orders o JOIN order_items oi ON oi.order_id = o.id "
        f"WHERE strftime('%Y', o.order_date) = '{y}' GROUP BY o.id)", False))

    add(3, "How many customers placed orders in more than one distinct year?",
        "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id "
        "HAVING COUNT(DISTINCT strftime('%Y', order_date)) > 1)")
    add(3, "How many customers placed orders in all three years 2023, 2024 and 2025?",
        "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id "
        "HAVING COUNT(DISTINCT strftime('%Y', order_date)) = 3)")

    def top_city(status):
        revenue = fetch(
            "SELECT SUM(p.price * oi.quantity) AS r FROM customers c JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id "
            f"WHERE o.status = '{status}' GROUP BY c.city ORDER BY r DESC LIMIT 2"
        )
        if len(revenue) < 2 or revenue[0][0] <= revenue[1][0]:
            return None
        return (
            f"Which city has the highest total revenue (price times quantity) from {status} orders?",
            "SELECT c.city FROM customers c JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id "
            f"WHERE o.status = '{status}' GROUP BY c.city ORDER BY SUM(p.price * oi.quantity) DESC LIMIT 1", False)

    take(3, 1, ["delivered", "shipped"], top_city)

    for text in UNSUPPORTED:
        add(0, text, None, kind="unsupported")
    for text in UNSAFE:
        add(0, text, None, kind="unsafe")
    return questions


def write_questions(path, questions: list[SqlQuestion]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def load_questions(path) -> list[SqlQuestion]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [SqlQuestion(**row) for row in rows]