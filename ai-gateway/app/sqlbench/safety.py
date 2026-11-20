"""Running SQL that a model wrote, without trusting it.

Defences, all deterministic: only a query that starts with SELECT or WITH, one statement, a read only
connection, an authorizer that allows only reading the four shop tables (and no risky functions), a time
limit and a row limit. A model that is told to delete data cannot do any harm through this path."""

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from app.sqlbench.database import TABLES, readonly_uri

_SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
_READ = getattr(sqlite3, "SQLITE_READ", 20)
_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_OK = getattr(sqlite3, "SQLITE_OK", 0)
_DENY = getattr(sqlite3, "SQLITE_DENY", 1)
_DENIED_FUNCTIONS = {"load_extension", "randomblob", "zeroblob", "readfile", "writefile"}

_ONLY_QUERIES = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_STATEMENT = re.compile(
    r"^\s*(select|with|insert|update|delete|drop|alter|create|replace|pragma|attach|detach|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)
_ANY_QUERY = re.compile(r"\b(select|with)\b", re.IGNORECASE)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Extraction:
    sql: str | None  # a statement found in the reply (it may be a forbidden one: the runner refuses it)
    declined: bool  # the model answered CANNOT_ANSWER


def extract_sql(text: str) -> Extraction:
    """The SQL in a model reply: a code fence, a reply that is the query, or the first query inside prose."""
    text = (text or "").strip()
    if not text:
        return Extraction(None, False)
    fenced = _FENCE.search(text)
    if not fenced and text.upper().startswith("CANNOT_ANSWER"):
        return Extraction(None, True)
    candidate = fenced.group(1).strip() if fenced else text
    if not _STATEMENT.match(candidate):
        found = _ANY_QUERY.search(candidate)
        if found is None:
            return Extraction(None, "CANNOT_ANSWER" in text.upper())
        candidate = candidate[found.start():]
    candidate = candidate.strip().rstrip(";").strip()
    return Extraction(candidate or None, False)


@dataclass
class QueryResult:
    ok: bool
    rows: list | None = None
    columns: int | None = None
    error: str | None = None
    kind: str | None = None  # blocked, syntax, execution or timeout
    truncated: bool = False


_BLOCKED_NAME_PREFIXES = ("sqlite_", "pragma_")


def _authorizer(action, arg1, arg2, database, source):
    if action == _READ:
        # The shop has four real tables and no others. A name that is not one of them is the name of a
        # common table expression (WITH ...), which can only read those tables itself. The tables that
        # describe the database (sqlite_master) and the pragma functions are refused.
        name = (arg1 or "").lower()
        return _DENY if name.startswith(_BLOCKED_NAME_PREFIXES) else _OK
    if action == _FUNCTION:
        return _DENY if (arg2 or "").lower() in _DENIED_FUNCTIONS else _OK
    return _OK if action in (_SELECT, _RECURSIVE) else _DENY


def _failure(message: str) -> QueryResult:
    lowered = message.lower()
    if "one statement" in lowered or "not authorized" in lowered or "authorization" in lowered or "prohibited" in lowered:
        return QueryResult(False, error=message, kind="blocked")
    if "interrupted" in lowered:
        return QueryResult(False, error="the query ran too long and was stopped", kind="timeout")
    if "syntax error" in lowered or "incomplete input" in lowered:
        return QueryResult(False, error=message, kind="syntax")
    return QueryResult(False, error=message, kind="execution")


def run_query(db_path, sql: str, *, max_rows: int = 200, timeout_seconds: float = 2.0) -> QueryResult:
    statement = sql.strip().rstrip(";").strip()
    if not _ONLY_QUERIES.match(statement):
        return QueryResult(False, error="only SELECT queries are allowed", kind="blocked")

    connection = sqlite3.connect(readonly_uri(db_path), uri=True)
    try:
        connection.set_authorizer(_authorizer)
        deadline = time.monotonic() + timeout_seconds
        connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cursor = connection.execute(statement)
        fetched = cursor.fetchmany(max_rows + 1)
        return QueryResult(
            True,
            rows=[tuple(row) for row in fetched[:max_rows]],
            columns=len(cursor.description or []),
            truncated=len(fetched) > max_rows,
        )
    except (sqlite3.Error, sqlite3.Warning) as error:
        return _failure(str(error))
    finally:
        connection.close()