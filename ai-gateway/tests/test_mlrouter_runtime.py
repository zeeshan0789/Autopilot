import numpy as np
import pytest

from app.mlrouter.features import FEATURE_NAMES, extract_features, prompt_text
from app.mlrouter.model import RouterModel
from app.mlrouter.runtime import MlRouter, MlRuntime, RouterSettings
from app.settings import BASE_DIR

MODELS = ["org/cheap", "strong"]


def easy(i):
    return [{"role": "user", "content": f"What is {i + 10} + {i + 20}? Reply with only the number."}]


def hard(i):
    return [{"role": "user", "content": f"Reverse the letters of the word 'abcdefgh{'xyz'[i % 3]}{i}'. Reply with only the reversed word."}]


class FakeEmbedder:
    """Arithmetic prompts point one way, reversal prompts another, anything else a third way."""

    def encode(self, text):
        if "What is" in text:
            return np.array([1.0, 0.0, 0.0])
        if "Reverse" in text:
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])


class BrokenEmbedder:
    def encode(self, text):
        raise RuntimeError("embedding failed")


def train_into(directory, tau=0.9, sim_threshold=0.8, feature_names=None, meta=None):
    prompts = [easy(i) for i in range(60)] + [hard(i) for i in range(60)]
    X = np.array([extract_features(m) for m in prompts])
    passed = {"org/cheap": np.array([1] * 60 + [0] * 60), "strong": np.ones(120, dtype=int)}
    router = RouterModel(MODELS).fit(X, passed, feature_names or FEATURE_NAMES)
    info = {"tau": tau, "sim_threshold": sim_threshold} if meta is None else meta
    router.save(directory, info)
    np.save(directory / "train_embeddings.npy", np.array([FakeEmbedder().encode(prompt_text(m)) for m in prompts]))
    return directory


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    return train_into(tmp_path_factory.mktemp("router"))


def ask(router, text):
    return router.decide([{"role": "user", "content": text}])


def test_a_familiar_easy_prompt_goes_to_the_cheap_model(trained):
    decision = ask(MlRouter.load(trained, FakeEmbedder()), "What is 31 + 47? Reply with only the number.")
    assert (decision.model, decision.status) == ("org/cheap", "router")
    assert decision.probability > 0.9 and decision.similarity == pytest.approx(1.0)


def test_a_familiar_hard_prompt_goes_to_the_strong_model(trained):
    decision = ask(MlRouter.load(trained, FakeEmbedder()), "Reverse the letters of the word 'abcdefghz99'. Reply with only the reversed word.")
    assert (decision.model, decision.status) == ("strong", "router")
    assert decision.probability < 0.2


def test_an_unfamiliar_prompt_is_declined_but_still_scored(trained):
    decision = ask(MlRouter.load(trained, FakeEmbedder()), "Write a poem about the sea.")
    assert (decision.model, decision.status) == (None, "novel")
    assert decision.similarity == pytest.approx(0.0)
    assert decision.probability is not None


def test_a_router_trained_with_other_features_is_refused(tmp_path):
    directory = tmp_path / "old"
    prompts = [easy(i) for i in range(60)] + [hard(i) for i in range(60)]
    X = np.array([extract_features(m) for m in prompts])
    passed = {"org/cheap": np.array([1] * 60 + [0] * 60), "strong": np.ones(120, dtype=int)}
    other_names = [f"old_feature_{i}" for i in range(X.shape[1])]  # as many names as columns, but not the current ones
    RouterModel(MODELS).fit(X, passed, other_names).save(directory, {"tau": 0.9, "sim_threshold": 0.8})
    np.save(directory / "train_embeddings.npy", np.ones((2, 3)))
    with pytest.raises(ValueError, match="different features"):
        MlRouter.load(directory, FakeEmbedder())


@pytest.mark.parametrize("meta", [{}, {"tau": 0.9}, {"tau": 2.0, "sim_threshold": 0.8}, {"tau": True, "sim_threshold": 0.8}])
def test_a_meta_file_without_valid_thresholds_is_refused(tmp_path, meta):
    train_into(tmp_path / "r", meta=meta)
    with pytest.raises(ValueError):
        MlRouter.load(tmp_path / "r", FakeEmbedder())


# ---------- The runtime ----------


def settings(directory, **fields):
    return RouterSettings(model_dir=directory, **fields)


def test_the_runtime_loads_once_and_decides(trained):
    created = []

    def factory():
        created.append(1)
        return FakeEmbedder()

    runtime = MlRuntime(settings(trained), factory)
    assert not runtime.available
    first = runtime.decide([{"role": "user", "content": "What is 31 + 47? Reply with only the number."}])
    second = runtime.decide([{"role": "user", "content": "Write a poem."}])
    assert (first.status, second.status) == ("router", "novel")
    assert runtime.available and len(created) == 1


def test_a_missing_router_means_unavailable_not_a_crash(tmp_path):
    runtime = MlRuntime(settings(tmp_path / "nothing"), FakeEmbedder)
    runtime.warm_up()
    assert not runtime.available
    assert runtime.decide([{"role": "user", "content": "hi"}]).status == "unavailable"


def test_a_disabled_runtime_never_loads_anything(trained):
    def factory():
        raise AssertionError("must not be called")

    runtime = MlRuntime(settings(trained, enabled=False), factory)
    runtime.warm_up()
    assert runtime.decide([{"role": "user", "content": "hi"}]).status == "disabled"


def test_a_failure_while_deciding_is_reported_as_an_error(trained):
    runtime = MlRuntime(settings(trained), BrokenEmbedder)
    decision = runtime.decide([{"role": "user", "content": "What is 31 + 47?"}])
    assert (decision.model, decision.status) == (None, "error")


def test_settings_from_config():
    defaults = RouterSettings.from_config({})
    assert defaults.alias == "smart" and defaults.enabled and defaults.shadow
    assert defaults.model_dir == BASE_DIR / "models" / "router"
    custom = RouterSettings.from_config({"alias": "x", "enabled": False, "shadow": False, "model_dir": "somewhere"})
    assert (custom.alias, custom.enabled, custom.shadow) == ("x", False, False)
    assert custom.model_dir == BASE_DIR / "somewhere"
    with pytest.raises(ValueError):
        RouterSettings(alias=" ")


def test_the_shipped_router_files_match_the_current_features():
    """If the features change, the saved router must be trained again (this test says so)."""
    directory = BASE_DIR / "models" / "router"
    if not directory.exists():
        pytest.skip("no saved router in this checkout")
    router = MlRouter.load(directory, FakeEmbedder())
    assert router.router.feature_names == list(FEATURE_NAMES)


def test_prompt_text_is_the_system_prompt_and_the_last_user_message():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert prompt_text(messages) == "Be brief.\nsecond"
    assert prompt_text([{"role": "user", "content": "only"}]) == "only"