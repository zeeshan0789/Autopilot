"""A small online shop database for the Text to SQL demo. It is generated from a seed, so every run
sees the same data, and the demo only ever opens it read only."""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from app.rag.corpus import FIRST_NAMES, LAST_NAMES, PRODUCT_MIDS, PRODUCT_MODELS, PRODUCT_PREFIXES

SEED = 3

SCHEMA = """CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    signup_year INTEGER NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price INTEGER NOT NULL -- list price in US dollars
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL, -- format YYYY-MM-DD
    status TEXT NOT NULL
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL
);"""

TABLES = ("customers", "products", "orders", "order_items")
CITIES = {
    "Lisbon": "Portugal", "Porto": "Portugal", "Nairobi": "Kenya", "Mombasa": "Kenya",
    "Toronto": "Canada", "Montreal": "Canada", "Osaka": "Japan", "Tokyo": "Japan",
}
CATEGORIES = ["laptop", "speaker", "router", "camera", "tablet", "headset"]
STATUSES = ["delivered", "shipped", "pending", "cancelled"]
STATUS_WEIGHTS = [50, 20, 15, 15]
CUSTOMERS = 40
ORDERING_CUSTOMERS = 35  # the last five customers never ordered anything
PRODUCTS = 24
ORDERS = 200
VALUE_COLUMNS = [("customers", "city"), ("customers", "country"), ("orders", "status"), ("products", "category")]


def readonly_uri(path) -> str:
    return f"{Path(path).resolve().as_uri()}?mode=ro"


def build_database(path, seed: int = SEED) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)

        customers, names = [], set()
        while len(customers) < CUSTOMERS:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name in names:
                continue
            names.add(name)
            city = rng.choice(sorted(CITIES))
            customers.append((len(customers) + 1, name, city, CITIES[city], rng.randint(2019, 2025)))
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

        products, product_names = [], set()
        while len(products) < PRODUCTS:
            name = f"{rng.choice(PRODUCT_PREFIXES)}{rng.choice(PRODUCT_MIDS)} {rng.choice(PRODUCT_MODELS)}"
            if name in product_names:
                continue
            product_names.add(name)
            products.append((len(products) + 1, name, CATEGORIES[len(products) % len(CATEGORIES)], rng.randint(20, 1500)))
        connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

        orders, items = [], []
        first_day = date(2023, 1, 1)
        for order_id in range(1, ORDERS + 1):
            day = first_day + timedelta(days=rng.randint(0, 1094))
            orders.append((order_id, rng.randint(1, ORDERING_CUSTOMERS), day.isoformat(), rng.choices(STATUSES, STATUS_WEIGHTS)[0]))
            for product_id in rng.sample(range(1, PRODUCTS + 1), rng.randint(1, 3)):
                items.append((order_id, product_id, rng.randint(1, 5)))
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
        connection.executemany("INSERT INTO order_items VALUES (?, ?, ?)", items)
        connection.commit()
    finally:
        connection.close()
    return path


def schema_text(path, rich: bool = False) -> str:
    """The schema for the prompt. rich adds three example rows per table and the values of the
    columns that hold a few fixed words (a model cannot know how 'cancelled' is spelled otherwise)."""
    if not rich:
        return SCHEMA
    connection = sqlite3.connect(readonly_uri(path), uri=True)
    try:
        lines = [SCHEMA, "", "-- Example rows"]
        for table in TABLES:
            for row in connection.execute(f"SELECT * FROM {table} LIMIT 3"):
                lines.append(f"-- {table}: {tuple(row)!r}")
        lines += ["", "-- Values of some columns"]
        for table, column in VALUE_COLUMNS:
            values = [r[0] for r in connection.execute(f"SELECT DISTINCT {column} FROM {table} ORDER BY 1")]
            lines.append(f"-- {table}.{column} is one of: " + ", ".join(repr(v) for v in values))
    finally:
        connection.close()
    return "\n".join(lines)