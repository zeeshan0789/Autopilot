import pytest

from app.bench.results import ItemResult, classify_wrong_hits, markdown_section, paired_difference, percentile, summarize_mode


def item(i, kind="unique", passed=True, status="ok", latency=100.0, cost=0.0, cache=None, fallback=False, outage=False):
    return ItemResult(f"w{i:03d}", kind, f"case{i}", outage, status, passed and status == "ok", latency,
                      cost_usd=cost, cache_status=cache, fallback_used=fallback)


ITEMS = [
    item(1, latency=1000, cost=1e-4, cache="miss"),
    item(2, "repeat", latency=20, cache="hit"),
    item(3, "repeat", passed=False, latency=25, cache="hit"),  # a wrong answer from the cache
    item(4, status="error", latency=50, outage=True),
    item(5, latency=3000, cost=1e-6, fallback=True, outage=True),
    item(6, "paraphrase", passed=False, latency=900, cost=1e-4, cache="miss"),
]


def test_percentiles():
    assert percentile([20, 25, 900, 1000, 3000], 0.5) == 900
    assert percentile([20, 25, 900, 1000, 3000], 0.95) == 3000
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([], 0.5) is None


def test_summary_counts():
    s = summarize_mode("autopilot", ITEMS)
    assert (s.requests, s.answered, s.errors) == (6, 5, 1)
    assert (s.errors_in_outage, s.errors_outside_outage) == (1, 0)
    assert s.correct == 3 and s.correct_rate == 0.5
    assert s.correct_outside_outage == 0.5 and s.correct_in_outage == 0.5
    assert (s.cache_hits, s.wrong_hits) == (2, 1)
    assert (s.fallbacks, s.fallbacks_in_outage) == (1, 1)
    assert (s.latency_p50, s.latency_p95) == (900, 3000)  # errors have no latency in the percentiles
    assert s.total_cost == pytest.approx(2.01e-4) and s.avg_cost == pytest.approx(3.35e-5)
    low, high = s.correct_interval
    assert low < 0.5 < high


def test_an_empty_run_has_no_rates():
    s = summarize_mode("direct", [])
    assert s.requests == 0 and s.correct_rate is None and s.latency_p50 is None and s.avg_cost is None


def test_paired_difference():
    direct = [item(i, passed=(i % 2 == 0)) for i in range(40)]  # half correct
    better = [item(i, passed=True) for i in range(40)]
    result = paired_difference(direct, better)
    assert result["mean"] == pytest.approx(0.5)
    assert 0.3 < result["low"] <= result["mean"] <= result["high"] < 0.7
    same = paired_difference(direct, direct)
    assert same["mean"] == 0 and same["low"] == 0 and same["high"] == 0


def test_runs_of_different_requests_cannot_be_compared():
    with pytest.raises(ValueError):
        paired_difference([item(1)], [item(2)])


def test_the_markdown_report_has_the_numbers_and_the_caveats():
    direct = [item(i, passed=(i % 4 != 0), latency=1000, cost=1e-4, cache="miss") for i in range(8)]
    autopilot = [item(i, passed=True, latency=100, cost=0.0, cache="hit") for i in range(8)]
    summaries = {"direct": summarize_mode("direct", direct), "autopilot": summarize_mode("autopilot", autopilot)}
    config = {"requests": 8, "kinds": "x", "model": "m", "outage_start": 2, "outage_end": 4}
    text = markdown_section(summaries, {"autopilot": paired_difference(direct, autopilot)}, config)
    assert "| | Direct to the provider | Autopilot |" in text
    assert "Cost against direct | +0% | -100%" in text
    assert "requests 3 to 4" in text
    assert "Autopilot against direct, correct answers: +25.0 points" in text
    assert "synthetic" in text and "simulated" in text

def entry(i, base, answer, passed, cache="miss", kind="unique", outage=False):
    return ItemResult(f"w{i:03d}", kind, base, outage, "ok", passed, 10.0, cache_status=cache, answer=answer)


def test_wrong_cache_answers_are_told_apart():
    items = [
        entry(1, "A", "x", False),  # the provider answered A wrongly (and the cache kept it)
        entry(2, "A", "x", False, cache="hit", kind="repeat"),  # the same prompt again: a repeat of that wrong answer
        entry(3, "B", "y", True),
        entry(4, "C", "y", False, cache="hit"),  # prompt C got the answer written for B
        entry(5, "D", "zzz", False, cache="hit"),  # nobody ever gave this answer
        entry(6, "A", "x", True, cache="hit", kind="repeat"),  # a correct hit is not listed
    ]
    prompts = {f"w{i:03d}": f"prompt {i}" for i in range(1, 7)}
    wrong = classify_wrong_hits(items, prompts)
    assert [(w.item_id, w.kind) for w in wrong] == [("w002", "replay"), ("w004", "cross_prompt"), ("w005", "unknown")]
    assert wrong[1].source_id == "w003" and wrong[1].source_prompt == "prompt 3" and wrong[1].prompt == "prompt 4"
    assert wrong[2].source_id is None


def test_the_difference_can_be_limited_to_the_outage_or_to_the_rest():
    direct = [entry(i, f"c{i}", "a", i % 2 == 0, outage=(i >= 20)) for i in range(40)]
    other = [entry(i, f"c{i}", "a", True, outage=(i >= 20)) for i in range(40)]
    everything = paired_difference(direct, other)
    outside = paired_difference(direct, other, outage=False)
    during = paired_difference(direct, other, outage=True)
    assert everything["n"] == 40 and outside["n"] == 20 and during["n"] == 20
    assert outside["mean"] == pytest.approx(0.5) and during["mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        paired_difference([entry(1, "a", "x", True, outage=True)], [entry(1, "a", "x", True, outage=True)], outage=False)


def test_the_markdown_report_shows_the_parts_and_the_kinds_of_wrong_answers():
    from collections import Counter

    direct = [entry(i, f"c{i}", "a", i % 4 != 0, outage=(i >= 4)) for i in range(8)]
    autopilot = [entry(i, f"c{i}", "a", True, outage=(i >= 4)) for i in range(8)]
    summaries = {"direct": summarize_mode("direct", direct), "autopilot": summarize_mode("autopilot", autopilot)}
    config = {"requests": 8, "kinds": "x", "model": "m", "outage_start": 2, "outage_end": 4}
    text = markdown_section(
        summaries,
        {"autopilot": paired_difference(direct, autopilot)},
        config,
        part_diffs={"autopilot": {"normal": paired_difference(direct, autopilot, outage=False), "outage": paired_difference(direct, autopilot, outage=True)}},
        wrong_kinds={"autopilot": Counter({"replay": 2, "cross_prompt": 3})},
    )
    assert "  - outside the outage:" in text and "  - during the outage:" in text
    assert "of which repeats of an earlier wrong answer | 0 | 2 |" in text
    assert "of which the answer written for another prompt | 0 | 3 |" in text