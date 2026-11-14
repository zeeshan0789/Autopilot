import asyncio
import json

import httpx
import pytest

from app.evals.cases import EvalCase
from app.evals.runner import CaseResult, RunConfig
from app.mlrouter.collect import collect_outcomes
from app.mlrouter.data import (
    append_outcome,
    family_matrix,
    load_outcomes,
    model_stats,
    oracle,
    outcome_cost,
    pending_cases,
    split_of,
)

CHEAP, MID, STRONG = "qwen2.5:0.5b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"
MODELS = [CHEAP, MID, STRONG]


def case(case_id):
    return EvalCase(id=case_id, category="x", messages=[{"role": "user", "content": "q"}], checks=[{"type": "contains", "value": "ok"}])


def result(case_id, answer="ok", **fields):
    values = dict(status="ok", answer=answer, prompt_tokens=100, completion_tokens=50, total_tokens=150, latency_ms=200)
    values.update(fields)
    return CaseResult(id=case_id, **values)


# ---------- Storage ----------


def test_outcomes_roundtrip_and_a_later_line_wins(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    append_outcome(path, MID, result("a", "first"))
    append_outcome(path, STRONG, result("a", "other model"))
    append_outcome(path, MID, result("a", "second"))
    outcomes = load_outcomes(path)
    assert set(outcomes) == {("a", MID), ("a", STRONG)}
    assert outcomes[("a", MID)].answer == "second"


def test_unusable_lines_and_unknown_fields_are_ignored(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    rows = [
        {"model": MID, "id": "a", "status": "invalid", "answer": None},
        {"model": MID, "id": "b", "status": "ok", "answer": "ok", "a_future_field": 1},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    assert set(load_outcomes(path)) == {("b", MID)}


def test_a_missing_file_means_no_outcomes(tmp_path):
    assert load_outcomes(tmp_path / "none.jsonl") == {}


def test_pending_cases_skip_finished_ones():
    outcomes = {("a", MID): result("a")}
    cases = [case("a"), case("b")]
    assert [c.id for c in pending_cases(cases, outcomes, MID)] == ["b"]
    assert [c.id for c in pending_cases(cases, outcomes, STRONG)] == ["a", "b"]


def test_the_split_of_a_generated_id_is_train_or_test():
    assert {split_of(f"gen_arith_d1_{i:03d}") for i in range(50)} == {"train", "test"}
    shares = sum(1 for i in range(400) if split_of(f"gen_arith_d1_{i:03d}") == "test") / 400
    assert 0.15 <= shares <= 0.35


# ---------- Numbers ----------


def test_cost_uses_the_list_price_of_the_model():
    assert outcome_cost(MID, result("a")) == pytest.approx((100 * 0.10 + 50 * 0.50) / 1_000_000)
    assert outcome_cost(MID, result("a", prompt_tokens=None)) is None
    assert outcome_cost("not-a-priced-model", result("a")) is None


def test_model_stats():
    cases = [case("a"), case("b")]
    outcomes = {("a", MID): result("a"), ("b", MID): result("b", "no")}
    stats = model_stats(cases, outcomes, [MID, STRONG])
    assert (stats[0].cases, stats[0].pass_rate, stats[0].mean_score) == (2, 0.5, 0.5)
    assert stats[0].avg_tokens == 150 and stats[0].avg_latency_ms == 200
    assert stats[0].avg_cost_usd == pytest.approx(3.5e-5)
    assert stats[1].cases == 0 and stats[1].pass_rate is None


def test_family_matrix_groups_by_family_and_level():
    cases = [case("gen_arith_d1_000"), case("gen_arith_d1_001"), case("fact_01")]
    outcomes = {
        ("gen_arith_d1_000", MID): result("gen_arith_d1_000"),
        ("gen_arith_d1_001", MID): result("gen_arith_d1_001", "no"),
        ("fact_01", MID): result("fact_01"),
    }
    matrix = family_matrix(cases, outcomes, [MID, STRONG])
    assert matrix["arith_d1"][MID] == (2, 0.5)
    assert matrix["arith_d1"][STRONG] == (0, None)
    assert matrix["handwritten"][MID] == (1, 1.0)


def test_the_oracle_picks_the_cheapest_passing_model():
    cases = [case("a"), case("b"), case("c"), case("d")]
    outcomes = {}
    for case_id, answers in {"a": ("ok", "ok", "ok"), "b": ("no", "ok", "ok"), "c": ("no", "no", "no")}.items():
        for model, answer in zip(MODELS, answers):
            outcomes[(case_id, model)] = result(case_id, answer)
    outcomes[("d", CHEAP)] = result("d")  # only one model answered this case: left out

    found = oracle(cases, outcomes, MODELS)
    assert found.cases == 3
    assert found.chosen == {CHEAP: 1, MID: 1, STRONG: 1}  # the unsolved case goes to the strongest
    assert found.oracle_pass_rate == pytest.approx(2 / 3)
    assert found.strongest_pass_rate == pytest.approx(2 / 3)
    assert found.cheapest_pass_rate == pytest.approx(1 / 3)
    cheap_cost, mid_cost, strong_cost = 7.5e-7, 3.5e-5, 5.25e-5
    assert found.oracle_avg_cost_usd == pytest.approx((cheap_cost + mid_cost + strong_cost) / 3)
    assert found.strongest_avg_cost_usd == pytest.approx(strong_cost)


def test_the_oracle_without_complete_cases():
    found = oracle([case("a")], {("a", CHEAP): result("a")}, MODELS)
    assert found.cases == 0 and found.oracle_pass_rate is None


# ---------- Collecting ----------


def good_reply():
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
        headers={"x-autopilot-fallback": "false", "x-autopilot-model": MID, "x-request-id": "req-1"},
    )


def collect(cases, handler, path, **kwargs):
    async def fake_sleep(seconds):
        pass

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gateway")
        try:
            config = RunConfig(model=MID, delay_seconds=0, retries=0)
            return await collect_outcomes(cases, config, "ap-test", path, client=client, sleep=fake_sleep, **kwargs)
        finally:
            await client.aclose()

    return asyncio.run(go())


def test_usable_answers_are_saved_with_their_tokens_and_request_id(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    summary = collect([case("a"), case("b")], lambda request: good_reply(), path)
    assert (summary.saved, summary.failed, summary.stopped_early) == (2, 0, False)
    outcomes = load_outcomes(path)
    saved = outcomes[("a", MID)]
    assert (saved.answer, saved.prompt_tokens, saved.completion_tokens) == ("ok", 7, 3)
    assert saved.request_id == "req-1"


def test_failures_are_not_saved_and_a_run_of_them_stops_early(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    cases = [case(f"c{i}") for i in range(6)]
    summary = collect(cases, lambda request: httpx.Response(502, json={}), path, max_consecutive_failures=3)
    assert (summary.saved, summary.failed, summary.stopped_early) == (0, 3, True)
    assert "later" in summary.reason
    assert not path.exists()


def test_a_success_resets_the_failure_streak(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    replies = iter([502, 502, 200, 502, 502, 200])

    def handler(request):
        return good_reply() if next(replies) == 200 else httpx.Response(502, json={})

    summary = collect([case(f"c{i}") for i in range(6)], handler, path, max_consecutive_failures=3)
    assert (summary.saved, summary.failed, summary.stopped_early) == (2, 4, False)