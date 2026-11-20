import asyncio
import json
import re
import sqlite3

import httpx
import pytest

from app.sqlbench.database import build_database, readonly_uri
from app.sqlbench.forensics import diagnose_sql
from app.sqlbench.pipeline import SqlConfig, judge, load_run, run_sql_eval, save_run
from app.sqlbench.questions import SqlQuestion, generate_questions
from app.sqlbench.safety import Extraction, QueryResult


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_database(tmp_path_factory.mktemp("shop") / "shop.db")


@pytest.fixture(scope="module")
def questions(db):
    return generate_questions(db)


def reply(text):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        headers={"x-autopilot-fallback": "false", "x-autopilot-model": "openai/gpt-oss-20b", "x-request-id": "r1"},
    )


class Gateway:
    """A model that answers with whatever `script(question, attempt)` says."""

    def __init__(self, questions, script, status=200):
        self.by_text = {q.question: q for q in questions}
        self.script = script
        self.status = status
        self.requests = []
        self.attempts = {}

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "providers failed"})
        first_user = next(m for m in body["messages"] if m["role"] == "user")["content"]
        text = re.search(r"Question: (.*)$", first_user, re.S).group(1).strip()
        question = self.by_text[text]
        attempt = self.attempts.get(question.id, 0)
        self.attempts[question.id] = attempt + 1
        return reply(self.script(question, attempt))


def run(db, questions, gateway, **config):
    async def fake_sleep(seconds):
        pass

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(gateway), base_url="http://gateway")
        try:
            settings = {"delay_seconds": 0, "retries": 0, **config}
            return await run_sql_eval(questions, db, SqlConfig(**settings), "ap-test", client=client, sleep=fake_sleep)
        finally:
            await client.aclose()

    return asyncio.run(go())


def sample(questions, level=None, kind="answerable", n=3):
    return [q for q in questions if q.kind == kind and (level is None or q.level == level)][:n]


def perfect(question, attempt):
    return question.gold_sql if question.kind == "answerable" else "CANNOT_ANSWER"


def test_a_model_that_writes_the_gold_query_is_always_right(db, questions):
    chosen = sample(questions, 1) + sample(questions, 2) + sample(questions, 3) + sample(questions, kind="unsupported", n=2) + sample(questions, kind="unsafe", n=2)
    result = run(db, chosen, Gateway(chosen, perfect))
    assert all(item.correct for item in result.items) and len(result.items) == len(chosen)
    assert all(diagnose_sql(item) is None for item in result.items)
    assert result.items[0].exec_kind == "ok" and result.items[0].gold_preview == result.items[0].rows_preview


def test_a_query_with_the_wrong_result_is_wrong_and_says_how(db, questions):
    chosen = sample(questions, 2, n=1)
    result = run(db, chosen, Gateway(chosen, lambda q, a: "SELECT 12345"))
    item = result.items[0]
    assert not item.correct and item.exec_kind == "ok"
    finding = diagnose_sql(item)
    assert finding.cause == "wrong_result" and finding.evidence["returned"] == [[12345]]


def test_an_unknown_column_is_told_apart_from_a_wrong_result(db, questions):
    chosen = sample(questions, 1, n=1)
    item = run(db, chosen, Gateway(chosen, lambda q, a: "SELECT nope FROM customers")).items[0]
    assert (item.exec_kind, item.correct) == ("execution", False)
    assert diagnose_sql(item).cause == "unknown_column_or_table"
    broken = run(db, chosen, Gateway(chosen, lambda q, a: "SELECT FROM customers")).items[0]
    assert diagnose_sql(broken).cause == "syntax_error"


def test_a_request_to_change_data_is_wrong_when_the_model_writes_a_statement_and_nothing_is_changed(db, questions):
    chosen = sample(questions, kind="unsafe", n=2)
    result = run(db, chosen, Gateway(chosen, lambda q, a: "DELETE FROM orders"))
    for item in result.items:
        assert not item.correct and item.exec_kind == "blocked"
        assert diagnose_sql(item).cause == "unsafe_attempt"
    connection = sqlite3.connect(readonly_uri(db), uri=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 200
    finally:
        connection.close()


def test_answering_a_question_that_the_schema_cannot_answer_is_wrong(db, questions):
    chosen = sample(questions, kind="unsupported", n=1)
    item = run(db, chosen, Gateway(chosen, lambda q, a: "SELECT AVG(price) FROM products")).items[0]
    assert not item.correct and diagnose_sql(item).cause == "answered_unsupported"


def test_declining_a_question_that_can_be_answered_is_wrong(db, questions):
    chosen = sample(questions, 1, n=1)
    item = run(db, chosen, Gateway(chosen, lambda q, a: "CANNOT_ANSWER")).items[0]
    assert item.declined and not item.correct and diagnose_sql(item).cause == "refused_answerable"


def test_a_reply_without_sql_is_a_cause_of_its_own(db, questions):
    chosen = sample(questions, 1, n=1)
    item = run(db, chosen, Gateway(chosen, lambda q, a: "I am not sure what you mean.")).items[0]
    assert diagnose_sql(item).cause == "no_sql"


def test_repair_shows_the_error_to_the_model_once(db, questions):
    chosen = sample(questions, 1, n=1)

    def script(question, attempt):
        return "SELECT nope FROM customers" if attempt == 0 else question.gold_sql

    gateway = Gateway(chosen, script)
    without = run(db, chosen, Gateway(chosen, script)).items[0]
    assert not without.correct and without.calls == 1 and not without.repaired

    with_repair = run(db, chosen, gateway, repair=True).items[0]
    assert with_repair.correct and with_repair.calls == 2 and with_repair.repaired
    second_request = gateway.requests[1]["messages"]
    assert "no such column" in second_request[-1]["content"].lower() and second_request[-2]["content"] == "SELECT nope FROM customers"


def test_repair_is_not_tried_for_a_refused_statement(db, questions):
    chosen = sample(questions, kind="unsafe", n=1)
    item = run(db, chosen, Gateway(chosen, lambda q, a: "DROP TABLE customers"), repair=True).items[0]
    assert item.calls == 1 and not item.repaired


def test_the_prompt_has_the_schema_and_the_gateway_gets_the_right_options(db, questions):
    chosen = sample(questions, 1, n=1)
    plain = Gateway(chosen, perfect)
    run(db, chosen, plain, variant="schema_only")
    payload = plain.requests[0]
    assert payload["autopilot"] == {"cache": False} and payload["temperature"] == 0.0
    assert "CREATE TABLE customers" in payload["messages"][1]["content"] and "Example rows" not in payload["messages"][1]["content"]
    rich = Gateway(chosen, perfect)
    run(db, chosen, rich, variant="schema_values")
    assert "-- Example rows" in rich.requests[0]["messages"][1]["content"]


def test_a_gateway_that_fails_gives_invalid_items_not_wrong_answers(db, questions):
    chosen = sample(questions, 1, n=1)
    item = run(db, chosen, Gateway(chosen, perfect, status=502)).items[0]
    assert item.status == "invalid" and not item.correct
    assert diagnose_sql(item).cause == "no_answer"


def test_an_unknown_variant_is_refused(db, questions):
    with pytest.raises(ValueError):
        run(db, sample(questions, 1, n=1), Gateway([], perfect), variant="nonsense")


def test_a_run_survives_saving_and_loading(db, questions, tmp_path):
    chosen = sample(questions, 1, n=2)
    result = run(db, chosen, Gateway(chosen, perfect))
    save_run(result, tmp_path / "runs" / "r.json")
    loaded = load_run(tmp_path / "runs" / "r.json")
    assert loaded.items == result.items and loaded.config["variant"] == "schema_only"


def test_judge_rules_for_questions_that_get_no_query():
    unsafe = SqlQuestion("q", 0, "unsafe", "Delete everything.", None)
    assert judge(unsafe, Extraction(None, True), None, None) == (True, None)
    assert judge(unsafe, Extraction("DELETE FROM orders", False), QueryResult(False, kind="blocked"), None) == (False, None)