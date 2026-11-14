import numpy as np
import pytest

from app.mlrouter.model import RouterModel

MODELS = ["org/cheap", "strong"]


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    passed = {"org/cheap": (X[:, 0] > 0.5).astype(int), "strong": np.ones(n, dtype=int)}
    return X, passed


def test_only_a_learnable_model_gets_a_classifier():
    X, passed = data()
    router = RouterModel(MODELS).fit(X, passed)
    assert set(router.boosters) == {"org/cheap"}
    assert router.priors["strong"] == pytest.approx(201 / 202)


def test_predictions_separate_the_classes_and_generalize():
    X, passed = data()
    router = RouterModel(MODELS).fit(X, passed)
    p = router.predict(X)["org/cheap"]
    assert p[passed["org/cheap"] == 1].mean() > 0.8
    assert p[passed["org/cheap"] == 0].mean() < 0.2

    X_new, passed_new = data(n=300, seed=1)
    p_new = router.predict(X_new)["org/cheap"]
    assert ((p_new > 0.5).astype(int) == passed_new["org/cheap"]).mean() > 0.9


def test_a_model_without_enough_failures_gets_only_its_pass_rate():
    X, passed = data()
    y = np.ones(len(X), dtype=int)
    y[:5] = 0  # 5 failures are too few to learn from
    router = RouterModel(MODELS).fit(X, {"org/cheap": y, "strong": passed["strong"]})
    assert router.boosters == {}
    assert router.predict(X)["org/cheap"] == pytest.approx(np.full(len(X), 196 / 202))


def test_the_prior_of_a_model_that_always_passes_is_below_one():
    X, passed = data()
    prior = RouterModel(MODELS).fit(X, passed).predict(X)["strong"]
    assert 0.99 < prior[0] < 1.0


def test_training_is_deterministic():
    X, passed = data()
    first = RouterModel(MODELS).fit(X, passed).predict(X)["org/cheap"]
    second = RouterModel(MODELS).fit(X, passed).predict(X)["org/cheap"]
    assert np.array_equal(first, second)


def test_a_single_row_can_be_predicted():
    X, passed = data()
    predicted = RouterModel(MODELS).fit(X, passed).predict(X[0])
    assert len(predicted["org/cheap"]) == 1 and len(predicted["strong"]) == 1


def test_a_saved_model_predicts_the_same(tmp_path):
    X, passed = data()
    router = RouterModel(MODELS).fit(X, passed, feature_names=["a", "b", "c"])
    router.save(tmp_path / "router", {"tau": 0.9})

    assert (tmp_path / "router" / "org_cheap.txt").exists()  # the "/" of the model name is not a folder
    loaded, meta = RouterModel.load(tmp_path / "router")
    assert meta["tau"] == 0.9 and meta["feature_names"] == ["a", "b", "c"]
    assert loaded.models == MODELS
    before, after = router.predict(X), loaded.predict(X)
    for model in MODELS:
        assert np.allclose(before[model], after[model])