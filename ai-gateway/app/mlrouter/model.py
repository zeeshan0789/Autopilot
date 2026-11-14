"""The learned router: one LightGBM classifier per model that predicts "this model's answer
will pass every check". A model without enough examples of both outcomes (at least
min_minority of each) gets no classifier, only its smoothed overall pass rate, because
nothing can be learned from a handful of failures."""

import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_data_in_leaf": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
DEFAULT_ROUNDS = 150
MIN_MINORITY = 15


def _file_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model) + ".txt"


class RouterModel:
    def __init__(self, models, params=None, rounds: int = DEFAULT_ROUNDS, min_minority: int = MIN_MINORITY, seed: int = 1):
        self.models = list(models)  # cheapest first
        self.params = {**DEFAULT_PARAMS, "seed": seed, **(params or {})}
        self.rounds = rounds
        self.min_minority = min_minority
        self.boosters: dict = {}
        self.priors: dict = {}
        self.feature_names: list | None = None

    def fit(self, X, passed: dict, feature_names=None) -> "RouterModel":
        X = np.asarray(X, dtype=float)
        self.feature_names = list(feature_names) if feature_names else None
        self.boosters, self.priors = {}, {}
        for model in self.models:
            y = np.asarray(passed[model], dtype=int)
            positives = int(y.sum())
            negatives = len(y) - positives
            self.priors[model] = (positives + 1) / (len(y) + 2)
            if min(positives, negatives) >= self.min_minority:
                dataset = lgb.Dataset(X, label=y, feature_name=self.feature_names or "auto")
                self.boosters[model] = lgb.train(self.params, dataset, num_boost_round=self.rounds)
        return self

    def predict(self, X) -> dict:
        """model -> probability that its answer passes, one value per row."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        result = {}
        for model in self.models:
            booster = self.boosters.get(model)
            result[model] = booster.predict(X) if booster is not None else np.full(len(X), self.priors[model])
        return result

    def save(self, directory, meta: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for model, booster in self.boosters.items():
            files[model] = _file_name(model)
            booster.save_model(str(directory / files[model]))
        info = {
            "models": self.models,
            "priors": {m: float(p) for m, p in self.priors.items()},
            "boosters": files,
            "feature_names": self.feature_names,
            **(meta or {}),
        }
        (directory / "meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        """(model, meta dict)."""
        directory = Path(directory)
        info = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        router = cls(info["models"])
        router.priors = dict(info["priors"])
        router.feature_names = info.get("feature_names")
        router.boosters = {
            model: lgb.Booster(model_file=str(directory / name)) for model, name in info["boosters"].items()
        }
        return router, info