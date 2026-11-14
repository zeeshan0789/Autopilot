import types

import numpy as np
import pytest

from app.evals.cases import EvalCase
from app.evals.runner import CaseResult
from app.mlrouter.evaluate import (
    Dataset,
    bootstrap_mean_ci,
    brier,
    build_dataset,
    calibration_table,
    choose_models,
    make_folds,
    oracle_picks,
    out_of_fold,
    replay,
    roc_auc,
    rule_policy_picks,
    select_tau,
)
from app.mlrouter.features import FEATURE_NAMES

ORDER = ["cheap", "strong"]


def make_cases(n, text=None):
    return [
        EvalCase(
            id=f"c{i}",
            category="x",
            messages=[{"role": "user", "content": text(i) if text else f"q{i}"}],
            checks=[{"type": "contains", "value": "ok"}],
        )
        for i in range(n)
    ]


def make_dataset(passed, costs, X=None):
    n = len(next(iter(passed.values())))
    return Dataset(
        make_cases(n),
        np.zeros((n, 2)) if X is None else X,
        {m: np.array(v, dtype=int) for m, v in passed.items()},
        {m: np.array(v, dtype=float) for m, v in costs.items()},
    )


# ---------- Building the dataset ----------


def result(case_id, answer):
    return CaseResult(id=case_id, status="ok", answer=answer, prompt_tokens=10, completion_tokens=5)


def test_only_cases_answered_by_every_model_are_used():
    qwen, mid = "qwen2.5:0.5b", "openai/gpt-oss-20b"
    cases = make_cases(3)
    outcomes = {
        ("c0", qwen): result("c0", "ok"), ("c0", mid): result("c0", "ok"),
        ("c1", qwen): result("c1", "no"), ("c1", mid): result("c1", "ok"),
        ("c2", qwen): result("c2", "ok"),  # the other model has no answer for c2
    }
    dataset = build_dataset(cases, outcomes, [qwen, mid])
    assert [c.id for c in dataset.cases] == ["c0", "c1"]
    assert dataset.X.shape == (2, len(FEATURE_NAMES))
    assert list(dataset.passed[qwen]) == [1, 0] and list(dataset.passed[mid]) == [1, 1]
    assert dataset.costs[mid][0] == pytest.approx((10 * 0.10 + 5 * 0.50) / 1_000_000)
    assert dataset.costs[qwen][0] == pytest.approx((10 * 0.005 + 5 * 0.005) / 1_000_000)


def test_a_subset_keeps_everything_aligned():
    dataset = make_dataset({"cheap": [1, 0, 1], "strong": [1, 1, 1]}, {"cheap": [0, 0, 0], "strong": [1, 2, 3]})
    sub = dataset.subset([2, 0])
    assert [c.id for c in sub.cases] == ["c2", "c0"]
    assert list(sub.passed["cheap"]) == [1, 1] and list(sub.costs["strong"]) == [3.0, 1.0]


# ---------- Cross validation ----------


def test_folds_cover_every_row_and_keep_both_classes_in_each_fold():
    labels = np.array([1] * 40 + [0] * 20)
    folds = make_folds(labels, k=5, seed=3)
    assert set(folds) == {0, 1, 2, 3, 4}
    for fold in range(5):
        assert {0, 1} <= set(labels[folds == fold])
    assert np.array_equal(folds, make_folds(labels, k=5, seed=3))


def test_out_of_fold_predictions_come_from_models_that_never_saw_the_row():
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    passed = {"cheap": (X[:, 0] > 0.5).astype(int), "strong": np.ones(200, dtype=int)}
    dataset = make_dataset(passed, {"cheap": np.zeros(200), "strong": np.ones(200)}, X=X)
    oof = out_of_fold(dataset, ORDER, k=5, seed=1)
    assert len(oof["cheap"]) == 200 and (oof["cheap"] > 0).all()
    assert roc_auc(passed["cheap"], oof["cheap"]) > 0.9


# ---------- Policies ----------


def test_the_cheapest_model_above_the_threshold_is_chosen():
    probs = {"cheap": np.array([0.9, 0.4, 0.7]), "strong": np.array([0.99, 0.99, 0.99])}
    assert choose_models(probs, ORDER, 0.6) == ["cheap", "strong", "cheap"]
    assert choose_models(probs, ORDER, 0.95) == ["strong", "strong", "strong"]
    nobody = {"cheap": np.array([0.1]), "strong": np.array([0.2])}
    assert choose_models(nobody, ORDER, 0.9) == ["strong"]  # the strongest when nothing qualifies


def test_replay_uses_the_outcome_and_cost_of_the_picked_model():
    dataset = make_dataset({"cheap": [1, 0], "strong": [1, 1]}, {"cheap": [0.0, 0.0], "strong": [1.0, 2.0]})
    r = replay(["cheap", "strong"], dataset)
    assert r.pass_rate == 1.0 and r.avg_cost == pytest.approx(1.0)
    assert r.shares(ORDER) == {"cheap": 0.5, "strong": 0.5}
    assert replay(["cheap", "cheap"], dataset).pass_rate == 0.5


def test_the_perfect_router_picks_the_cheapest_passing_model():
    dataset = make_dataset({"cheap": [1, 0, 0], "strong": [1, 1, 0]}, {"cheap": [0] * 3, "strong": [1] * 3})
    assert oracle_picks(dataset, ORDER) == ["cheap", "strong", "strong"]


def test_the_rule_policy_is_replayed_through_its_own_decide():
    class FakeRouter:
        settings = types.SimpleNamespace(alias="auto")

        def decide(self, request):
            return types.SimpleNamespace(model="strong" if "hard" in request.messages[-1].content else "cheap")

    cases = make_cases(3, text=lambda i: "hard question" if i == 1 else "easy")
    assert rule_policy_picks(cases, FakeRouter()) == ["cheap", "strong", "cheap"]


# ---------- Choosing the threshold ----------


def test_the_cheapest_threshold_within_the_quality_budget_is_chosen():
    dataset = make_dataset(
        {"cheap": [1, 1, 1, 1, 0, 0], "strong": [1] * 6},
        {"cheap": [0.0] * 6, "strong": [1.0] * 6},
    )
    probs = {"cheap": np.array([0.9, 0.9, 0.9, 0.9, 0.2, 0.2]), "strong": np.ones(6)}
    # Up to 0.9 the cheap model serves the four cases it passes, above it everything goes to the strong one
    assert select_tau(probs, dataset, ORDER, "strong", 0.0) == 0.9


def test_a_looser_budget_can_accept_a_cheaper_but_riskier_threshold():
    dataset = make_dataset(
        {"cheap": [1, 1, 1, 1, 0, 0], "strong": [1] * 6},
        {"cheap": [0.0] * 6, "strong": [1.0] * 6},
    )
    probs = {"cheap": np.array([0.9, 0.9, 0.9, 0.9, 0.6, 0.6]), "strong": np.ones(6)}
    assert select_tau(probs, dataset, ORDER, "strong", 0.0) == 0.9
    assert select_tau(probs, dataset, ORDER, "strong", 0.4) == 0.6  # may lose two of six, so use cheap more


def test_no_threshold_in_the_grid_can_be_none():
    dataset = make_dataset({"cheap": [0, 0, 0], "strong": [1, 1, 1]}, {"cheap": [0.0] * 3, "strong": [1.0] * 3})
    probs = {"cheap": np.array([0.9, 0.9, 0.9]), "strong": np.ones(3)}
    assert select_tau(probs, dataset, ORDER, "strong", 0.0, grid=[0.5]) is None


# ---------- Numbers ----------


def test_auc():
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert roc_auc([1, 0], [0.5, 0.5]) == 0.5
    assert roc_auc([1, 1], [0.5, 0.6]) is None


def test_brier():
    assert brier([1, 0], [1.0, 0.0]) == 0.0
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_calibration_table():
    rows = calibration_table([1, 1, 0, 0], [0.9, 0.95, 0.1, 0.15])
    by_low = {low: (count, actual) for low, _, count, actual in rows}
    assert by_low[0.8] == (2, 1.0)
    assert by_low[0.0] == (2, 0.0)
    assert by_low[0.4] == (0, None)


def test_bootstrap_interval_is_tight_for_constant_values_and_wide_for_noisy_ones():
    assert bootstrap_mean_ci([1.0] * 50) == (1.0, 1.0)
    low, high = bootstrap_mean_ci([1.0] * 10 + [0.0] * 10)
    assert 0.2 < low < 0.5 < high < 0.8
    assert bootstrap_mean_ci([1, 0, 1, 1, 0, 1], seed=4) == bootstrap_mean_ci([1, 0, 1, 1, 0, 1], seed=4)