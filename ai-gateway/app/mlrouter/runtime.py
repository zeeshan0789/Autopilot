"""The learned router inside the gateway.

Model "smart" is routed by the LightGBM router of scripts/router_train.py, but only for prompts
that look like the prompts it was trained on (the novelty guard). For every other prompt, and
whenever anything goes wrong, the rule policy decides, as for model "auto". Requests for "auto"
can also be scored silently (shadow mode): the router's pick is only recorded in the trace.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.cache.runtime import cache_service
from app.mlrouter.evaluate import choose_models
from app.mlrouter.features import FEATURE_NAMES, extract_features, prompt_text
from app.mlrouter.model import RouterModel
from app.settings import BASE_DIR, load_router_config

logger = logging.getLogger("autopilot.mlrouter")


@dataclass(frozen=True)
class RouterSettings:
    enabled: bool = True
    alias: str = "smart"
    model_dir: Path = BASE_DIR / "models" / "router"
    shadow: bool = True

    def __post_init__(self):
        if not self.alias or not self.alias.strip():
            raise ValueError("alias must not be empty")

    @classmethod
    def from_config(cls, config: dict) -> "RouterSettings":
        directory = Path(config.get("model_dir", "models/router"))
        if not directory.is_absolute():
            directory = BASE_DIR / directory
        return cls(
            enabled=bool(config.get("enabled", True)),
            alias=config.get("alias", "smart"),
            model_dir=directory,
            shadow=bool(config.get("shadow", True)),
        )


@dataclass(frozen=True)
class MlDecision:
    model: str | None  # the router's pick, None when it declines to pick
    status: str  # router, novel (unfamiliar prompt), error, unavailable, disabled
    probability: float | None = None  # predicted chance that the cheapest model passes
    similarity: float | None = None  # similarity to the nearest training prompt


class MlRouter:
    def __init__(self, router: RouterModel, tau: float, sim_threshold: float, train_embeddings, embedder):
        self.router = router
        self.tau = tau
        self.sim_threshold = sim_threshold
        self.train_embeddings = np.asarray(train_embeddings, dtype=float)
        self.embedder = embedder

    @classmethod
    def load(cls, directory, embedder) -> "MlRouter":
        router, meta = RouterModel.load(directory)
        if router.feature_names != list(FEATURE_NAMES):
            raise ValueError("the saved router was trained with different features, train it again")
        tau, threshold = meta.get("tau"), meta.get("sim_threshold")
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0 < tau <= 1:
            raise ValueError("meta.json has no valid tau, train the router again")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("meta.json has no sim_threshold, train the router again")
        embeddings = np.load(Path(directory) / "train_embeddings.npy")
        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise ValueError("train_embeddings.npy is empty or has the wrong shape")
        return cls(router, float(tau), float(threshold), embeddings, embedder)

    def decide(self, messages: list[dict]) -> MlDecision:
        order = self.router.models
        probabilities = self.router.predict(np.array(extract_features(messages), dtype=float))
        p_cheapest = float(probabilities[order[0]][0])
        embedding = np.asarray(self.embedder.encode(prompt_text(messages)), dtype=float)
        similarity = float((self.train_embeddings @ embedding).max())
        if similarity < self.sim_threshold:
            return MlDecision(None, "novel", p_cheapest, similarity)
        pick = choose_models(probabilities, order, self.tau)[0]
        return MlDecision(pick, "router", p_cheapest, similarity)


class MlRuntime:
    """Loads the router once, on the first use or at startup, and never raises."""

    def __init__(self, settings: RouterSettings, embedder_factory):
        self.settings = settings
        self._embedder_factory = embedder_factory
        self._router: MlRouter | None = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._router is not None

    def warm_up(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self.settings.enabled:
                return
            try:
                self._router = MlRouter.load(self.settings.model_dir, self._embedder_factory())
            except Exception as e:
                logger.warning(
                    "Learned router not available (%s): model '%s' falls back to the rule policy",
                    e,
                    self.settings.alias,
                )

    def decide(self, messages: list[dict]) -> MlDecision:
        if not self.settings.enabled:
            return MlDecision(None, "disabled")
        self._ensure_loaded()
        if self._router is None:
            return MlDecision(None, "unavailable")
        try:
            return self._router.decide(messages)
        except Exception as e:
            logger.warning("Learned router failed (%s), using the rule policy", e)
            return MlDecision(None, "error")


# The embedding model is shared with the semantic cache
ml_runtime = MlRuntime(RouterSettings.from_config(load_router_config()), lambda: cache_service.embedder)