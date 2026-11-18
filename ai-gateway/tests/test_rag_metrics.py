import pytest

from app.rag.forensics import diagnose_rag
from app.rag.metrics import (
    abstention_sweep,
    failure_counts,
    recommend_min_score,
    retrieval_summary,
    summarize_items,
)
from app.rag.pipeline import RagItem
from app.rag.report import markdown_section, summary_lines


def item(kind="answerable", mode="rag", passed=False, gold_rank=1, k=3, abstained=False, answer="wrong", status="ok", top=0.8, **extra):
    values = dict(
        question_id="q", kind=kind, mode=mode, question="?", gold_doc="d1", k=k,
        retrieved=[{"chunk": "d1#0", "doc": "d1", "title": "T", "score": 0.8}],
        gold_rank=gold_rank, top_score=top, abstained=abstained, status=status, answer=answer,
        passed=passed, failed_checks=["contains: missing 'x'"],
    )
    values.update(extra)
    return RagItem(**values)


# ---------- Retrieval numbers ----------


def test_retrieval_summary():
    summary = retrieval_summary([1, 2, None, 4, 1])
    assert summary["n"] == 5
    assert summary["hit@1"] == pytest.approx(2 / 5)
    assert summary["hit@3"] == pytest.approx(3 / 5)
    assert summary["hit@5"] == pytest.approx(4 / 5)
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25 + 1) / 5)
    assert retrieval_summary([])["mrr"] is None


def test_the_abstention_sweep_and_its_recommendation():
    sweep = abstention_sweep([0.9, 0.8, 0.7, 0.6], [0.3, 0.4, 0.65], thresholds=[0.5, 0.65, 0.75, 0.95])
    by_t = {row["threshold"]: row for row in sweep}
    assert by_t[0.5]["false_abstain"] == 0 and by_t[0.5]["correct_abstain"] == pytest.approx(2 / 3)
    assert by_t[0.65]["false_abstain"] == 0.25 and by_t[0.65]["correct_abstain"] == pytest.approx(2 / 3)
    assert by_t[0.95]["false_abstain"] == 1.0 and by_t[0.95]["correct_abstain"] == 1.0
    assert recommend_min_score(sweep, max_false_abstain=0.0) == 0.5
    assert recommend_min_score(sweep, max_false_abstain=0.3) == 0.65
    assert recommend_min_score(abstention_sweep([0.1], [0.2], thresholds=[0.5])) is None


# ---------- The summary of a run ----------


def test_a_summary_counts_by_kind_and_keeps_invalid_answers_apart():
    items = [
        item("answerable", passed=True, latency_ms=100, served_model="openai/gpt-oss-20b", prompt_tokens=100, completion_tokens=10),
        item("answerable", passed=False, latency_ms=300, served_model="openai/gpt-oss-20b", prompt_tokens=100, completion_tokens=30),
        item("answerable", status="invalid", answer=None, error="http 502"),
        item("unanswerable", passed=True, abstained=True, answer="I cannot find this"),
    ]
    summary = summarize_items(items)
    answerable = summary["by_kind"]["answerable"]
    assert (answerable["n"], answerable["valid"], answerable["passed"]) == (3, 2, 1)
    assert answerable["rate"] == 0.5
    assert summary["by_kind"]["unanswerable"]["rate"] == 1.0
    assert summary["by_kind"]["injection"]["rate"] is None
    assert (summary["questions"], summary["invalid"], summary["abstained"], summary["llm_calls"]) == (4, 1, 1, 2)
    assert summary["avg_tokens"] == pytest.approx(120) and summary["avg_latency_ms"] == 200
    assert summary["avg_cost_usd"] == pytest.approx(((100 * 0.10 + 10 * 0.50) + (100 * 0.10 + 30 * 0.50)) / 2 / 1_000_000)


# ---------- Forensics ----------


def cause(**kwargs):
    finding = diagnose_rag(item(**kwargs))
    return None if finding is None else finding.cause


def test_a_right_answer_needs_no_diagnosis():
    assert cause(passed=True) is None


def test_every_kind_of_failure_gets_its_own_cause():
    assert cause(status="invalid", answer=None) == "no_answer"
    assert cause(kind="injection", answer="PWNED") == "followed_injected_instruction"
    assert cause(mode="closed_book", gold_rank=None) == "no_context"
    assert cause(kind="unanswerable", answer="It costs 5 dollars", gold_rank=None) == "hallucinated_answer"
    assert cause(gold_rank=7) == "retrieval_miss"
    assert cause(gold_rank=None) == "retrieval_miss"
    assert cause(abstained=True, answer="I cannot find this in the documents.") == "false_abstain"
    assert cause(answer="   ") == "empty_answer"
    assert cause(answer="I cannot answer that question.") == "refused_despite_context"
    assert cause(answer="the answer is 7, honestly") == "generation_failure"


def test_a_missing_document_wins_over_what_the_model_said():
    finding = diagnose_rag(item(gold_rank=5, k=3, answer="I cannot find this in the documents."))
    assert finding.cause == "retrieval_miss"
    assert "ranked 5" in finding.message and finding.evidence["k"] == 3 and finding.evidence["top_score"] == 0.8


def test_the_oracle_mode_blames_the_generation_not_the_retrieval():
    assert cause(mode="oracle", gold_rank=1) == "generation_failure"


def test_the_evidence_of_a_hallucination_has_the_retrieval_score():
    finding = diagnose_rag(item(kind="unanswerable", answer="It costs 5", top=0.41, gold_rank=None))
    assert finding.evidence["top_score"] == 0.41 and finding.evidence["retrieved"] == ["d1"]


def test_failure_counts():
    items = [item(gold_rank=9), item(gold_rank=9), item(answer="the answer is 7"), item(passed=True)]
    assert dict(failure_counts(items)) == {"retrieval_miss": 2, "generation_failure": 1}


# ---------- The report ----------


def test_the_report_has_the_numbers_and_the_limits():
    rag = [item("answerable", passed=True, latency_ms=100, served_model="openai/gpt-oss-20b", prompt_tokens=100, completion_tokens=10),
           item("answerable", gold_rank=9, latency_ms=100, served_model="openai/gpt-oss-20b", prompt_tokens=100, completion_tokens=10)]
    summaries = {"rag": summarize_items(rag)}
    retrieval = retrieval_summary([1, 9])
    info = {"docs": 34, "chunks": 40, "questions": 94, "answerable": 77, "unanswerable": 14, "injection": 3, "model": "m", "k": 3}
    text = markdown_section(summaries, retrieval, failure_counts(rag), info)
    assert "34 documents (40 chunks)" in text and "| RAG (retrieved chunks) | 1 of 2 (50.0%)" in text
    assert "retrieval_miss 1" in text and "in the first 1 chunks for 50.0%" in text
    assert "templates" in text and "variation of the model" in text
    assert any("answerable" in line and "1 of" in line for line in summary_lines("rag", summaries["rag"]))