"""The Text to SQL app: ask a model through the gateway for one SQL query, run it safely, compare the
result with the result of the gold query. Every request goes through the gateway (traced, costed, routed)
with the cache switched off."""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.evals.cases import EvalCase
from app.evals.runner import RunConfig, run_case
from app.sqlbench.compare import compare_results
from app.sqlbench.database import schema_text
from app.sqlbench.questions import SqlQuestion
from app.sqlbench.safety import Extraction, QueryResult, extract_sql, run_query

VARIANTS = ("schema_only", "schema_values")

SYSTEM = """You translate questions about a SQLite database into one SQL query.
Rules:
- Reply with only the SQL query. No explanation, no markdown.
- Use only SELECT queries (a WITH clause is fine). Never modify data.
- Use only the tables and columns of the schema.
- If the question cannot be answered with this schema, or it asks to change data, reply exactly: CANNOT_ANSWER"""

REPAIR = "That query failed with this error: {error}\nReply with the corrected SQL query only, or CANNOT_ANSWER."


def build_messages(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]


@dataclass
class SqlConfig:
    name: str = "sql"
    model: str = "openai/gpt-oss-20b"
    variant: str = "schema_only"
    repair: bool = False  # on a syntax or execution error, show the error to the model once
    base_url: str = "http://127.0.0.1:8000"
    delay_seconds: float = 2.0
    retries: int = 3
    backoff_base: float = 2.0


@dataclass
class SqlItem:
    question_id: str
    level: int
    kind: str
    question: str
    status: str  # ok, or invalid when no usable reply came back
    answer: str | None
    sql: str | None
    declined: bool
    exec_kind: str | None  # None when no query ran, otherwise ok, blocked, syntax, execution or timeout
    error: str | None
    match_reason: str | None
    correct: bool
    calls: int
    repaired: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    served_model: str | None
    request_ids: list = field(default_factory=list)
    rows_preview: list | None = None
    gold_preview: list | None = None
    truncated: bool = False


@dataclass
class SqlRun:
    name: str
    created_at: str
    config: dict
    items: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "created_at": self.created_at, "config": self.config, "items": [asdict(i) for i in self.items]}

    @classmethod
    def from_dict(cls, data: dict) -> "SqlRun":
        return cls(data["name"], data["created_at"], data["config"], [SqlItem(**i) for i in data["items"]])


def save_run(run: SqlRun, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def load_run(path) -> SqlRun:
    return SqlRun.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def judge(question: SqlQuestion, extraction: Extraction, query: QueryResult | None, gold: QueryResult | None):
    """(correct, reason of the mismatch)."""
    if question.kind != "answerable":
        return extraction.declined and extraction.sql is None, None
    if extraction.declined or extraction.sql is None or query is None or not query.ok or gold is None:
        return False, None
    comparison = compare_results(gold.rows, query.rows, question.ordered)
    return comparison.match, comparison.reason


def _preview(rows) -> list:
    return [list(row) for row in rows[:3]]


def gold_results(questions: list[SqlQuestion], db_path) -> dict:
    results = {}
    for q in questions:
        if q.gold_sql is None:
            continue
        result = run_query(db_path, q.gold_sql, max_rows=1000, timeout_seconds=10)
        if not result.ok:
            raise RuntimeError(f"the gold query of {q.id} failed: {result.error}")
        results[q.id] = result
    return results


async def _answer(client, q, db_path, schema, gold, config: SqlConfig, run_config: RunConfig, sleep) -> SqlItem:
    messages = build_messages(schema, q.question)
    attempts = 2 if config.repair else 1
    calls = prompt_tokens = completion_tokens = latency = 0
    request_ids: list = []
    served_model = answer = None
    extraction = Extraction(None, False)
    query = None
    repaired = False
    failure = None

    for attempt in range(attempts):
        result = await run_case(client, EvalCase(q.id, q.kind, messages, []), run_config, sleep)
        calls += 1
        if result.status != "ok":
            failure = result.error or result.status
            break
        prompt_tokens += result.prompt_tokens or 0
        completion_tokens += result.completion_tokens or 0
        latency += result.latency_ms or 0
        served_model = result.served_model
        if result.request_id:
            request_ids.append(result.request_id)
        answer = result.answer or ""
        extraction = extract_sql(answer)
        query = None
        if extraction.sql is None:
            break
        query = run_query(db_path, extraction.sql)
        if query.ok or query.kind in ("blocked", "timeout") or attempt == attempts - 1:
            break
        repaired = True
        messages = messages + [
            {"role": "assistant", "content": extraction.sql},
            {"role": "user", "content": REPAIR.format(error=query.error)},
        ]

    base = dict(question_id=q.id, level=q.level, kind=q.kind, question=q.question, calls=calls, repaired=repaired,
                request_ids=request_ids)
    if failure is not None:
        return SqlItem(**base, status="invalid", answer=None, sql=None, declined=False, exec_kind=None, error=failure,
                       match_reason=None, correct=False, prompt_tokens=None, completion_tokens=None, latency_ms=None,
                       served_model=None)

    correct, reason = judge(q, extraction, query, gold.get(q.id))
    return SqlItem(
        **base,
        status="ok",
        answer=answer,
        sql=extraction.sql,
        declined=extraction.declined,
        exec_kind=None if query is None else ("ok" if query.ok else query.kind),
        error=None if query is None or query.ok else query.error,
        match_reason=reason,
        correct=correct,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency,
        served_model=served_model,
        rows_preview=_preview(query.rows) if query is not None and query.ok else None,
        gold_preview=_preview(gold[q.id].rows) if q.id in gold else None,
        truncated=bool(query is not None and query.truncated),
    )


async def run_sql_eval(
    questions: list[SqlQuestion],
    db_path,
    config: SqlConfig,
    api_key: str,
    *,
    client=None,
    sleep=asyncio.sleep,
    progress=None,
) -> SqlRun:
    if config.variant not in VARIANTS:
        raise ValueError(f"unknown variant '{config.variant}'")
    schema = schema_text(db_path, rich=config.variant == "schema_values")
    gold = gold_results(questions, db_path)
    run_config = RunConfig(
        name=config.name, base_url=config.base_url, model=config.model, temperature=0.0,
        delay_seconds=config.delay_seconds, retries=config.retries, backoff_base=config.backoff_base,
    )
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            base_url=config.base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=180.0, trust_env=False
        )
    run = SqlRun(config.name, datetime.now(timezone.utc).isoformat(timespec="seconds"), asdict(config))
    try:
        for number, q in enumerate(questions, 1):
            item = await _answer(client, q, db_path, schema, gold, config, run_config, sleep)
            run.items.append(item)
            if progress:
                progress(number, len(questions), item)
            if number < len(questions) and config.delay_seconds > 0:
                await sleep(config.delay_seconds)
    finally:
        if owns_client:
            await client.aclose()
    return run