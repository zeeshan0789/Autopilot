"""Training data, cross validation, routing policies replayed on recorded outcomes, and the
numbers of the router evaluation.

Every case was answered by every model, so any policy can be replayed exactly: the case gets
the outcome (pass or fail) and the cost of the model the policy would have picked.
"""

from dataclasses import dataclass

import numpy as np

from app.evals.cases import EvalCase
from app.evals.scoring import score_case
from app.mlrouter.data import outcome_cost
from app.mlrouter.features import extract_features
from app.mlrouter.model import RouterModel
from app.schemas.openai import ChatCompletionRequest

TAU_GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 up to 0.99


@dataclass
class Dataset:
    cases: list
    X: np.ndarray
    passed: dict  # model -> 0/1 per case
    costs: dict  # model -> estimated cost per case

    def __len__(self) -> int:
        return len(self.cases)

    def subset(self, indices) -> "Dataset":
        indices = np.asarray(indices, dtype=int)
        return Dataset(
            [self.cases[i] for i in indices],
            self.X[indices],
            {m: v[indices] for m, v in self.passed.items()},
            {m: v[indices] for m, v in self.costs.items()},
        )


def build_dataset(cases: list[EvalCase], outcomes: dict, models: list[str]) -> Dataset:
    """Only cases that every model answered can be used."""
    complete = [c for c in cases if all((c.id, m) in outcomes for m in models)]
    X = np.array([extract_features(c.messages) for c in complete], dtype=float).reshape(len(complete), -1)
    passed = {
        m: np.array([1 if score_case(c, outcomes[(c.id, m)]).passed else 0 for c in complete], dtype=int)
        for m in models
    }
    costs = {
        m: np.array([outcome_cost(m, outcomes[(c.id, m)]) or 0.0 for c in complete], dtype=float)
        for m in models
    }
    return Dataset(complete, X, passed, costs)


# ---------- Cross validation ----------


def make_folds(labels, k: int = 5, seed: int = 1) -> np.ndarray:
    """Fold number of every row, with the two classes spread evenly over the folds."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(labels), dtype=int)
    for value in np.unique(labels):
        indices = rng.permutation(np.where(labels == value)[0])
        folds[indices] = np.arange(len(indices)) % k
    return folds


def out_of_fold(dataset: Dataset, models: list[str], k: int = 5, seed: int = 1, rounds=None, params=None) -> dict:
    """Predictions for every row from a model that never saw that row."""
    folds = make_folds(dataset.passed[models[0]], k, seed)
    oof = {m: np.zeros(len(dataset)) for m in models}
    options = {} if rounds is None else {"rounds": rounds}
    for fold in range(k):
        train_rows = np.where(folds != fold)[0]
        test_rows = np.where(folds == fold)[0]
        if len(test_rows) == 0:
            continue
        train = dataset.subset(train_rows)
        router = RouterModel(models, params=params, seed=seed, **options).fit(train.X, train.passed)
        predicted = router.predict(dataset.X[test_rows])
        for m in models:
            oof[m][test_rows] = predicted[m]
    return oof


# ---------- Policies ----------


def choose_models(probs: dict, order: list[str], tau: float) -> list[str]:
    """For every row the cheapest model whose predicted pass probability is at least tau,
    or the strongest model when none is."""
    n = len(probs[order[0]])
    return [next((m for m in order if probs[m][i] >= tau), order[-1]) for i in range(n)]


def rule_policy_picks(cases: list[EvalCase], router) -> list[str]:
    """What the rule based policy (model "auto") would pick for every case."""
    picks = []
    for case in cases:
        request = ChatCompletionRequest(model=router.settings.alias, messages=case.messages)
        picks.append(router.decide(request).model)
    return picks


def oracle_picks(dataset: Dataset, order: list[str]) -> list[str]:
    return [
        next((m for m in order if dataset.passed[m][i]), order[-1]) for i in range(len(dataset))
    ]


@dataclass
class Replay:
    picks: list
    passed: np.ndarray
    cost: np.ndarray

    @property
    def pass_rate(self) -> float:
        return float(self.passed.mean())

    @property
    def avg_cost(self) -> float:
        return float(self.cost.mean())

    def shares(self, order: list[str]) -> dict:
        return {m: sum(1 for p in self.picks if p == m) / len(self.picks) for m in order}


def replay(picks: list[str], dataset: Dataset) -> Replay:
    passed = np.array([dataset.passed[m][i] for i, m in enumerate(picks)], dtype=float)
    cost = np.array([dataset.costs[m][i] for i, m in enumerate(picks)], dtype=float)
    return Replay(list(picks), passed, cost)


def select_tau(probs: dict, dataset: Dataset, order: list[str], baseline_model: str, tolerance: float, grid=None):
    """The threshold with the lowest cost whose pass rate stays within `tolerance` of the
    baseline model. A tie goes to the higher (safer) threshold. None when no threshold fits."""
    floor = float(dataset.passed[baseline_model].mean()) - tolerance
    best = None
    for tau in grid or TAU_GRID:
        outcome = replay(choose_models(probs, order, tau), dataset)
        if outcome.pass_rate < floor - 1e-12:
            continue
        if best is None or outcome.avg_cost < best[1] - 1e-15 or (abs(outcome.avg_cost - best[1]) <= 1e-15 and tau > best[0]):
            best = (tau, outcome.avg_cost)
    return None if best is None else best[0]


# ---------- Numbers ----------


def roc_auc(labels, scores):
    """Chance that a random passing case gets a higher score than a random failing one."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def brier(labels, probabilities) -> float:
    labels, probabilities = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - labels) ** 2))


def calibration_table(labels, probabilities, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)) -> list:
    """(low, high, cases, share that really passed) for each bin of predicted probability."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        rows.append((low, min(high, 1.0), int(mask.sum()), float(labels[mask].mean()) if mask.any() else None))
    return rows


def bootstrap_mean_ci(values, iterations: int = 2000, seed: int = 1) -> tuple:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))