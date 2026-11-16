from app.evals.cases import EvalCase
from app.evals.runner import CaseResult
from app.forensics.diagnose import Observation, cause_counts, diagnose

CASE = EvalCase(
    id="c1",
    category="x",
    messages=[{"role": "user", "content": "Capital of France?"}],
    checks=[{"type": "contains", "value": "Paris"}],
)


def result(answer="Paris", **fields):
    values = dict(status="ok", answer=answer, served_model="m1", latency_ms=1000, completion_tokens=20, attempts=1)
    values.update(fields)
    return CaseResult(id="c1", **values)


def obs(answer="Paris", config=None, trace=None, **fields):
    return Observation(result(answer, **fields), config or {"model": "m1"}, trace)


def kinds(diagnosis, role=None):
    return [f.kind for f in diagnosis.findings if role is None or f.role == role]


def test_a_passing_answer_needs_no_diagnosis():
    d = diagnose(CASE, obs("Paris"))
    assert not d.failed and d.findings == []
    assert d.conclusion() == "the answer passed every check"


def test_a_wrong_answer_without_any_context_has_no_cause_and_says_what_was_not_checked():
    d = diagnose(CASE, obs("Lyon"))
    assert d.failed and kinds(d) == ["failed_checks"] and d.causes == []
    text = d.conclusion()
    assert text.startswith("no deterministic cause found")
    assert "Not checked" in text and "baseline" in text and "cache" in text
    assert "Paris" in d.findings[0].message


def test_a_refusal_is_a_cause():
    d = diagnose(CASE, obs("I\u2019m sorry, but I can\u2019t help with that."))
    assert "refusal" in kinds(d, "cause")


def test_an_empty_answer_is_a_cause_and_not_also_a_refusal():
    d = diagnose(CASE, obs("   "))
    assert kinds(d, "cause") == ["empty_answer"]


def test_a_missing_answer_is_a_cause_with_its_error():
    d = diagnose(CASE, Observation(CaseResult(id="c1", status="invalid", error="boom"), {"model": "m1"}))
    assert kinds(d, "cause") == ["no_answer"] and "boom" in d.findings[0].message
    assert "failed_checks" not in kinds(d)


def test_a_fallback_answer_is_a_cause_that_names_both_models():
    d = diagnose(CASE, obs("Lyon", config={"model": "gpt"}, served_model="qwen", fallback_used=True))
    finding = next(f for f in d.findings if f.kind == "fallback")
    assert finding.role == "cause" and "qwen" in finding.message and "gpt" in finding.message


def test_retries_are_context_not_a_cause():
    d = diagnose(CASE, obs("Lyon", attempts=3))
    assert kinds(d, "context") == ["retries"] and d.causes == []


# ---------- Compared with a baseline ----------


def test_a_different_model_than_a_passing_baseline_is_a_cause():
    candidate = obs("Lyon", config={"model": "small"}, served_model="small")
    baseline = obs("Paris", config={"model": "big"}, served_model="big")
    d = diagnose(CASE, candidate, baseline)
    finding = next(f for f in d.findings if f.kind == "model")
    assert finding.role == "cause" and "small" in finding.message and "big" in finding.message
    assert "same_setup" not in kinds(d)


def test_a_different_model_than_a_baseline_that_failed_too_is_only_context():
    candidate = obs("Lyon", config={"model": "small"}, served_model="small")
    baseline = obs("Nice", config={"model": "big"}, served_model="big")
    d = diagnose(CASE, candidate, baseline)
    assert next(f for f in d.findings if f.kind == "model").role == "context"
    assert d.causes == []


def test_a_different_prompt_version_is_a_cause():
    candidate = obs("Lyon", config={"model": "m1", "prompt": "qa", "version": 2})
    baseline = obs("Paris", config={"model": "m1", "prompt": "qa", "version": 1})
    finding = next(f for f in diagnose(CASE, candidate, baseline).findings if f.kind == "prompt_version")
    assert finding.role == "cause" and "qa:2" in finding.message and "qa:1" in finding.message


def test_an_identical_setup_is_reported_as_such():
    d = diagnose(CASE, obs("Lyon"), obs("Paris"))
    assert "same_setup" in kinds(d, "info") and d.causes == []
    assert d.conclusion().startswith("no deterministic cause found")


def test_latency_spikes_are_context():
    slow = diagnose(CASE, obs("Lyon", latency_ms=5000), obs("Paris", latency_ms=1000))
    assert "latency" in kinds(slow, "context")
    normal = diagnose(CASE, obs("Lyon", latency_ms=1500), obs("Paris", latency_ms=1000))
    assert "latency" not in kinds(normal)


def test_a_latency_far_above_the_baseline_p95_is_flagged():
    d = diagnose(CASE, obs("Lyon", latency_ms=3000), obs("Paris", latency_ms=2500), baseline_latency_p95=1200)
    assert "latency" in kinds(d, "context")


def test_a_much_longer_or_shorter_answer_is_context():
    longer = diagnose(CASE, obs("Lyon", completion_tokens=400), obs("Paris", completion_tokens=50))
    assert "length" in kinds(longer, "context") and "longer" in next(f for f in longer.findings if f.kind == "length").message
    shorter = diagnose(CASE, obs("Lyon", completion_tokens=5), obs("Paris", completion_tokens=400))
    assert "shorter" in next(f for f in shorter.findings if f.kind == "length").message
    similar = diagnose(CASE, obs("Lyon", completion_tokens=60), obs("Paris", completion_tokens=50))
    assert "length" not in kinds(similar)


# ---------- The gateway's trace ----------


def test_a_cache_hit_is_a_cause():
    trace = {"cache_status": "hit", "cache_similarity": 0.93, "status": "success"}
    d = diagnose(CASE, obs("Lyon", trace=trace))
    finding = next(f for f in d.findings if f.kind == "cache")
    assert finding.role == "cause" and "0.93" in finding.message


def test_a_gateway_error_is_a_cause():
    trace = {"status": "error", "http_status": 502, "error": "all providers failed"}
    finding = next(f for f in diagnose(CASE, obs("Lyon", trace=trace)).findings if f.kind == "gateway_error")
    assert finding.role == "cause" and "502" in finding.message and "all providers failed" in finding.message


def test_a_routing_decision_that_differs_from_a_passing_baseline_is_a_cause_with_the_router_evidence():
    trace = {
        "status": "success", "policy_rule": "ml_router", "routed_model": "small",
        "ml_status": "router", "ml_pick": "small", "ml_prob": 0.98, "ml_similarity": 0.9,
    }
    candidate = obs("Lyon", config={"model": "smart"}, served_model="small", trace=trace)
    baseline = obs("Paris", config={"model": "big"}, served_model="big")
    finding = next(f for f in diagnose(CASE, candidate, baseline).findings if f.kind == "routing")
    assert finding.role == "cause" and "ml_router" in finding.message
    assert finding.evidence["ml_prob"] == 0.98 and finding.evidence["ml_similarity"] == 0.9


def test_an_explicit_model_is_not_a_routing_finding():
    d = diagnose(CASE, obs("Lyon", trace={"status": "success", "policy_rule": "explicit", "routed_model": "m1"}))
    assert "routing" not in kinds(d)
    assert "cache" not in d.unchecked  # the trace was there


def test_the_conclusion_lists_every_cause():
    trace = {"cache_status": "hit", "cache_similarity": 0.9, "status": "success"}
    d = diagnose(CASE, obs("I cannot help with that.", served_model="qwen", fallback_used=True, trace=trace))
    text = d.conclusion()
    assert text.startswith("likely cause:")
    assert "refused" in text and "fallback" in text and "semantic cache" in text


def test_cause_counts():
    first = diagnose(CASE, obs("Lyon", served_model="qwen", fallback_used=True))
    second = diagnose(CASE, obs("Lyon"))  # no cause found
    third = diagnose(CASE, obs("Paris"))  # passed
    counts = cause_counts([first, second, third])
    assert counts == {"fallback": 1, "no_cause_found": 1}