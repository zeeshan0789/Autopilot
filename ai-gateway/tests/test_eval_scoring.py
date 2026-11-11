import pytest

from app.evals.cases import EvalCase
from app.evals.runner import CaseResult, EvalRun
from app.evals.scoring import score_case, score_run, summary_lines

MATH = EvalCase(
    id="m1",
    category="math",
    messages=[{"role": "user", "content": "2 + 2?"}],
    checks=[{"type": "numeric_answer", "value": 4}, {"type": "max_words", "value": 3}],
)
FACT = EvalCase(
    id="f1",
    category="factual",
    messages=[{"role": "user", "content": "Capital of France?"}],
    checks=[{"type": "contains", "value": "Paris"}],
)


def ok(case_id, answer, latency=100, tokens=10):
    return CaseResult(id=case_id, status="ok", answer=answer, latency_ms=latency, total_tokens=tokens)


def run_of(*results):
    return EvalRun(name="r", created_at="now", config={}, results=list(results))


def test_a_case_score_is_the_share_of_passed_checks():
    assert score_case(MATH, ok("m1", "4")).score == 1.0
    partial = score_case(MATH, ok("m1", "The answer is 4 for sure"))  # right number, too many words
    assert partial.score == 0.5 and not partial.passed
    assert score_case(MATH, ok("m1", "It is probably five I think")).score == 0.0


def test_a_case_without_an_answer_is_invalid_not_zero():
    invalid = CaseResult(id="m1", status="invalid", error="answered by a fallback model")
    scored = score_case(MATH, invalid)
    assert not scored.valid and scored.score is None
    assert "fallback" in scored.reason


def test_run_totals_ignore_invalid_cases():
    run = run_of(ok("m1", "4"), ok("f1", "London"), CaseResult(id="x", status="invalid", error="boom"))
    cases = [MATH, FACT, EvalCase(id="x", category="factual", messages=FACT.messages, checks=FACT.checks)]
    score = score_run(cases, run)
    assert len(score.cases) == 3 and score.invalid_count == 1
    assert score.mean_score == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert score.pass_rate == pytest.approx(0.5)
    assert score.by_category() == {"factual": (1, 0.0, 0.0), "math": (1, 1.0, 1.0)}


def test_latency_and_tokens_are_averaged_over_scored_cases():
    run = run_of(ok("m1", "4", latency=100, tokens=10), ok("f1", "Paris", latency=300, tokens=30))
    score = score_run([MATH, FACT], run)
    assert score.avg_latency_ms == 200 and score.avg_tokens == 20


def test_an_unknown_case_id_is_an_error():
    with pytest.raises(ValueError, match="not in the cases file"):
        score_run([MATH], run_of(ok("ghost", "x")))


def test_an_empty_run_has_no_averages():
    score = score_run([MATH], run_of())
    assert score.mean_score is None and score.pass_rate is None


def test_summary_mentions_the_numbers_and_invalid_cases():
    run = run_of(ok("m1", "4"), CaseResult(id="f1", status="invalid", error="fallback"))
    text = "\n".join(summary_lines(score_run([MATH, FACT], run)))
    assert "2 cases, 1 scored, 1 invalid" in text
    assert "100.0%" in text
    assert "invalid f1: fallback" in text