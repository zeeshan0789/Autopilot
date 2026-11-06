from decimal import Decimal

import numpy as np
import pytest

from app.cache.guard import check_guard
from app.metrics.cache_eval import (
    LabeledPair,
    category_stats,
    default_thresholds,
    evaluate,
    false_hit_pairs,
    load_pairs,
    lowest_safe_threshold,
    missed_pairs,
    score_pairs,
    sweep,
    wilson_interval,
)
from app.metrics.cache_report import summarize
from app.settings import BASE_DIR

# Hand made 4 dimensional embeddings with known similarities between the pairs below
VECTORS = {
    "What is the capital of France?": [1, 0, 0, 0],
    "Which city is the capital of France?": [0.95, 0.31225, 0, 0],  # 0.95
    "What was Apple revenue in 2024?": [0, 0, 1, 0],
    "What was Apple revenue in 2025?": [0, 0, 0.97, 0.2431],  # 0.97
    "how to bake bread": [0, 0, 0, 1],
    "why is the sky blue": [0.9798, 0, 0, 0.2],  # 0.20
    "what is the capital of france": [1, 0, 0, 0],
    "what is the population of france": [0.88, 0.47497, 0, 0],  # 0.88
}

PAIRS = [
    LabeledPair("Which city is the capital of France?", "What is the capital of France?", True, "paraphrase"),
    LabeledPair("What was Apple revenue in 2025?", "What was Apple revenue in 2024?", False, "year_changed"),
    LabeledPair("why is the sky blue", "how to bake bread", False, "unrelated"),
    LabeledPair("what is the population of france", "what is the capital of france", False, "topic_different"),
]


class FakeEmbedder:
    def encode(self, text: str) -> np.ndarray:
        vector = np.asarray(VECTORS[text], dtype=np.float32)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def scored():
    return score_pairs(PAIRS, FakeEmbedder())


# ---------- Scoring and outcomes ----------


def test_scoring(scored):
    assert [round(s.similarity, 2) for s in scored] == [0.95, 0.97, 0.20, 0.88]
    assert [s.guard_ok for s in scored] == [True, False, True, True]
    assert scored[1].guard_reason == "number_mismatch"


def test_guard_removes_the_false_hit_that_the_threshold_lets_through(scored):
    with_guard = evaluate(scored, 0.90, use_guard=True)
    assert (with_guard.hits, with_guard.false_hits) == (1, 0)
    assert with_guard.false_hit_rate == 0.0
    assert with_guard.recall == 1.0

    without_guard = evaluate(scored, 0.90, use_guard=False)
    assert (without_guard.hits, without_guard.false_hits) == (2, 1)
    assert without_guard.false_hit_rate == 0.5


def test_lower_threshold_lets_the_topic_pair_in(scored):
    outcome = evaluate(scored, 0.85, use_guard=True)
    assert (outcome.hits, outcome.false_hits) == (2, 1)


def test_no_hits_means_no_false_hit_rate(scored):
    outcome = evaluate(scored, 0.99, use_guard=True)
    assert outcome.hits == 0
    assert outcome.false_hit_rate is None
    assert outcome.recall == 0.0


def test_lowest_safe_threshold(scored):
    outcomes = sweep(scored, [0.5, 0.8, 0.9, 0.99], use_guard=True)
    assert lowest_safe_threshold(outcomes).threshold == 0.9
    assert lowest_safe_threshold(sweep(scored, [0.5], use_guard=True)) is None


def test_false_hit_and_missed_pairs(scored):
    assert [s.pair.category for s in false_hit_pairs(scored, 0.85)] == ["topic_different"]
    assert [s.pair.category for s in missed_pairs(scored, 0.99)] == ["paraphrase"]


def test_category_stats(scored):
    stats = {c.category: c for c in category_stats(scored)}
    assert stats["year_changed"].guard_blocked == 1
    assert stats["paraphrase"].reusable == 1
    assert stats["year_changed"].max_similarity == pytest.approx(0.97, abs=1e-3)


def test_default_thresholds():
    thresholds = default_thresholds()
    assert thresholds[0] == 0.5 and thresholds[-1] == 0.99 and len(thresholds) == 50
    assert thresholds == sorted(thresholds)


# ---------- Confidence interval ----------


def test_wilson_interval():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=0.001)

    low, high = wilson_interval(5, 10)
    assert low + high == pytest.approx(1.0)
    assert wilson_interval(0, 0) == (0.0, 1.0)


# ---------- The shipped dataset ----------


def shipped_pairs():
    return load_pairs(BASE_DIR / "benchmarks" / "cache_pairs.jsonl")


def test_shipped_dataset_is_valid():
    pairs = shipped_pairs()
    assert len(pairs) >= 50
    assert any(p.reusable for p in pairs) and any(not p.reusable for p in pairs)
    keys = [(p.new_prompt, p.cached_prompt) for p in pairs]
    assert len(keys) == len(set(keys)), "duplicate pairs"


def test_guard_blocks_changed_numbers_years_and_names_in_the_shipped_dataset():
    for p in shipped_pairs():
        if p.category in ("number_changed", "year_changed", "entity_changed"):
            assert not check_guard(p.new_prompt, p.cached_prompt).ok, p


def test_load_pairs_rejects_a_bad_label(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"new_prompt": "a", "cached_prompt": "b", "reusable": "yes", "category": "x"}\n')
    with pytest.raises(ValueError):
        load_pairs(path)


# ---------- Traffic report ----------


def row(status, reason=None, saved_tokens=None, saved_cost=None, cost="0", ms=100, outcome="success"):
    return {
        "cache_status": status,
        "cache_reason": reason,
        "saved_tokens": saved_tokens,
        "saved_cost_usd": None if saved_cost is None else Decimal(saved_cost),
        "estimated_cost_usd": Decimal(cost),
        "latency_ms": ms,
        "status": outcome,
    }


def test_summarize():
    rows = [
        row("hit", "hit", 100, "0.0001", "0", 20),
        row("hit", "hit", 50, "0.00005", "0", 30),
        row("miss", "no_candidate", cost="0.0001", ms=1000),
        row("blocked", "number_mismatch", cost="0.0002", ms=800),
        row("bypass", "time_sensitive", cost="0.0003", ms=900),
        row(None, cost="0.0009", ms=500),  # written before the cache existed: ignored
    ]
    s = summarize(rows)
    assert (s.total, s.hits, s.misses, s.blocked, s.bypass) == (5, 2, 1, 1, 1)
    assert s.consulted == 4
    assert s.hit_rate == 0.5
    assert s.bypass_reasons == {"time_sensitive": 1}
    assert s.tokens_saved == 150
    assert s.cost_saved_usd == Decimal("0.00015")
    assert s.cost_spent_usd == Decimal("0.0006")
    assert s.saving_pct == pytest.approx(0.2)
    assert s.median_ms_hit == 25
    assert s.median_ms_not_hit == 900


def test_summarize_edge_cases():
    empty = summarize([])
    assert empty.total == 0 and empty.hit_rate is None and empty.saving_pct is None

    s = summarize([
        row("hit", "hit", 10, None, "0", 5),  # a hit whose original call had no price
        row("miss", "no_candidate", ms=5000, outcome="error"),  # failed requests skip latency
    ])
    assert s.unpriced_hits == 1
    assert s.median_ms_not_hit is None