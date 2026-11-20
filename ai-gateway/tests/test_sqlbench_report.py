import pytest

from app.sqlbench.pipeline import SqlItem
from app.sqlbench.forensics import diagnose_sql
from app.sqlbench.report import failure_counts, markdown_section, summarize, summary_lines


def item(kind="answerable", level=1, correct=True, status="ok", sql="SELECT 1", declined=False, exec_kind="ok", error=None,
         match_reason=None, calls=1, tokens=(100, 20), model="openai/gpt-oss-20b", **extra):
    values = dict(
        question_id="q", level=level, kind=kind, question="?", status=status, answer=sql, sql=sql, declined=declined,
        exec_kind=exec_kind, error=error, match_reason=match_reason, correct=correct, calls=calls, repaired=False,
        prompt_tokens=tokens[0], completion_tokens=tokens[1], latency_ms=100, served_model=model,
    )
    values.update(extra)
    return SqlItem(**values)


ITEMS = [
    item(level=1), item(level=1, correct=False, match_reason="values"),
    item(level=2), item(level=3, correct=False, exec_kind="execution", error="no such column: x"),
    item(kind="unsupported", level=0, correct=True, sql=None, declined=True, exec_kind=None),
    item(kind="unsafe", level=0, correct=False, sql="DELETE FROM orders", exec_kind="blocked", error="only SELECT queries are allowed"),
    item(kind="unsafe", level=0, correct=True, sql=None, declined=True, exec_kind=None),
    item(level=1, status="invalid", correct=False, answer=None, sql=None, exec_kind=None, error="http 502", prompt_tokens=None, completion_tokens=None),
]


def test_the_summary_counts_by_level_and_kind():
    s = summarize(ITEMS)
    assert (s["questions"], s["invalid"]) == (8, 1)
    assert (s["levels"][1]["n"], s["levels"][1]["passed"]) == (2, 1)
    assert (s["levels"][2]["passed"], s["levels"][3]["passed"]) == (1, 0)
    assert (s["answerable"]["n"], s["answerable"]["passed"]) == (4, 2)
    assert s["unsupported"]["rate"] == 1.0 and s["unsafe"]["rate"] == 0.5
    assert s["executed"] == 0.75
    assert (s["write_attempts"], s["blocked"]) == (1, 1)
    assert s["avg_calls"] == 1 and s["avg_tokens"] == 120
    assert s["avg_cost_usd"] == pytest.approx((100 * 0.10 + 20 * 0.50) / 1_000_000)


def test_failure_counts_name_the_causes():
    counts = failure_counts(ITEMS)
    assert dict(counts) == {"wrong_result": 1, "unknown_column_or_table": 1, "unsafe_attempt": 1, "no_answer": 1}


def test_every_failure_has_a_cause_and_a_correct_answer_has_none():
    assert diagnose_sql(ITEMS[0]) is None
    assert all(diagnose_sql(i) is not None for i in ITEMS if not i.correct)


def test_the_console_summary_mentions_the_safety_numbers():
    text = "\n".join(summary_lines("run", summarize(ITEMS)))
    assert "the model wrote a statement 1 times, the safety layer blocked 1 queries" in text
    assert "simple" in text and "joins" in text and "complex" in text


def test_the_markdown_has_a_row_per_run_and_the_limits():
    summaries = {"plain": summarize(ITEMS), "values": summarize(ITEMS[:4])}
    failures = {name: failure_counts(items) for name, items in (("plain", ITEMS), ("values", ITEMS[:4]))}
    info = {"rows": "40 customers", "questions": 67, "answerable": 57, "simple": 20, "joins": 20, "complex": 17, "unsupported": 5, "unsafe": 5}
    text = markdown_section(summaries, failures, info)
    assert "| plain | 1 of 2 (50.0%)" in text and "| values |" in text
    assert "67 questions" in text and "safety layer" in text and "template" in text
    assert "`plain`: 4 questions without a right answer" in text

def test_a_reply_that_is_neither_a_query_nor_a_decline_is_not_called_a_statement():
    reply = item(kind="unsafe", level=0, correct=False, sql=None, declined=False, exec_kind=None, answer="I will think about it")
    assert diagnose_sql(reply).cause == "no_decline"