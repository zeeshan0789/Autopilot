import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cache.service import CacheService, CacheSettings
from app.core.policy import PolicyRouter, PolicySettings, Rule
from app.main import app

client = TestClient(app)

QUESTION = "What is the capital of France?"
HARD = "Please analyze the trade offs of two database designs."

VECTORS = {QUESTION: [1.0, 0.0, 0.0, 0.0], HARD: [0.0, 1.0, 0.0, 0.0]}

CACHE_SETTINGS = CacheSettings(
    similarity_threshold=0.90,
    ttl_seconds=3600,
    max_entries_per_scope=100,
    max_prompt_chars=500,
    max_temperature=0.7,
)


class FakeEmbedder:
    dimension = 4

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(VECTORS[text], dtype=np.float32)


def make_policy(enabled=True) -> PolicyRouter:
    """Simple prompts go to 'mock'. Prompts with the word analyze go to 'mock-down',
    which always fails, so they end up on the fallback ('mock') and the difference
    between the routed model and the served model becomes visible."""
    return PolicyRouter(
        PolicySettings(
            enabled=enabled,
            tiers={"simple": "mock", "complex": "mock-down"},
            default_tier="simple",
            rules=(Rule("analysis", "complex", {"any_keywords": ["analyze"]}),),
        )
    )


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    router = make_policy()
    monkeypatch.setattr("app.api.chat.policy_router", router)
    return router


@pytest.fixture
def executor_calls(monkeypatch):
    from app.api import chat as chat_module

    real = chat_module.execute_with_fallback
    calls = []

    async def counting(chain, request, *args, **kwargs):
        calls.append(request.model)
        return await real(chain, request, *args, **kwargs)

    monkeypatch.setattr("app.api.chat.execute_with_fallback", counting)
    return calls


def post(key: str, content: str, model: str = "auto", **extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    )


# ---------- Routing ----------


def test_auto_is_routed_to_the_default_tier(api_key, traces):
    r = post(api_key, QUESTION)
    assert r.status_code == 200
    assert r.headers["x-autopilot-route"] == "default"
    assert r.headers["x-autopilot-routed-model"] == "mock"
    assert r.headers["x-autopilot-model"] == "mock"
    t = traces[0]
    assert t.requested_model == "auto"  # what the client asked for
    assert t.routed_model == "mock"  # what the policy chose
    assert t.served_model == "mock"  # who answered
    assert t.policy_rule == "default"


def test_a_matching_rule_routes_to_another_tier(api_key, traces):
    r = post(api_key, HARD)
    assert r.status_code == 200
    assert r.headers["x-autopilot-route"] == "analysis"
    assert r.headers["x-autopilot-routed-model"] == "mock-down"
    assert traces[0].policy_rule == "analysis"
    assert traces[0].routed_model == "mock-down"


def test_routed_model_that_fails_still_falls_back(api_key, traces):
    r = post(api_key, HARD)
    assert r.headers["x-autopilot-model"] == "mock"  # the fallback answered
    assert r.headers["x-autopilot-fallback"] == "true"
    t = traces[0]
    assert t.routed_model == "mock-down"
    assert t.served_model == "mock"
    assert t.fallback_used is True


def test_the_provider_receives_the_concrete_model_not_the_alias(api_key, executor_calls):
    post(api_key, QUESTION)
    assert executor_calls == ["mock"]


def test_explicit_model_is_not_routed(api_key, traces):
    r = post(api_key, HARD, model="mock")
    assert r.status_code == 200
    assert "x-autopilot-route" not in r.headers
    t = traces[0]
    assert t.requested_model == "mock"
    assert t.routed_model == "mock"
    assert t.policy_rule == "explicit"


def test_unknown_explicit_model_is_traced_with_the_explicit_rule(api_key, traces):
    assert post(api_key, QUESTION, model="no-such-model").status_code == 404
    assert traces[0].policy_rule == "explicit"
    assert traces[0].routed_model == "no-such-model"


def test_disabled_policy_makes_auto_an_unknown_model(api_key, monkeypatch, traces):
    monkeypatch.setattr("app.api.chat.policy_router", make_policy(enabled=False))
    assert post(api_key, QUESTION).status_code == 404
    assert traces[0].policy_rule == "explicit"


def test_a_tier_pointing_at_an_unknown_model_gives_404(api_key, monkeypatch):
    broken = PolicyRouter(
        PolicySettings(tiers={"simple": "not-configured"}, default_tier="simple")
    )
    monkeypatch.setattr("app.api.chat.policy_router", broken)
    assert post(api_key, QUESTION).status_code == 404


def test_a_policy_bug_does_not_fail_the_request(api_key, monkeypatch, traces):
    def boom(request):
        raise RuntimeError("bug")

    monkeypatch.setattr("app.core.policy.extract_features", boom)
    r = post(api_key, QUESTION)
    assert r.status_code == 200
    assert traces[0].policy_rule == "policy_error"
    assert r.headers["x-autopilot-route"] == "policy_error"


def test_stream_request_is_rejected_before_routing(api_key, traces):
    assert post(api_key, QUESTION, stream=True).status_code == 400
    assert traces[0].policy_rule is None


# ---------- Routing and the cache ----------


@pytest.fixture
def cache(monkeypatch):
    service = CacheService(CACHE_SETTINGS, FakeEmbedder())
    monkeypatch.setattr("app.api.chat.cache_service", service)
    return service


def test_cache_is_used_for_routed_requests(api_key, cache, executor_calls, traces):
    first = post(api_key, QUESTION)
    second = post(api_key, QUESTION)
    assert first.headers["x-autopilot-cache"] == "miss"
    assert second.headers["x-autopilot-cache"] == "hit"
    assert second.headers["x-autopilot-route"] == "default"
    assert second.headers["x-autopilot-routed-model"] == "mock"
    assert executor_calls == ["mock"]
    assert traces[1].policy_rule == "default"
    assert traces[1].routed_model == "mock"


def test_cache_is_keyed_by_the_resolved_model_not_by_the_alias(api_key, cache, executor_calls):
    """auto resolved to 'mock' and an explicit 'mock' are the same model, so they share answers."""
    post(api_key, QUESTION, model="auto")
    again = post(api_key, QUESTION, model="mock")
    assert again.headers["x-autopilot-cache"] == "hit"
    assert executor_calls == ["mock"]


def test_an_answer_from_one_model_is_never_served_for_another_model(
    api_key, cache, monkeypatch, executor_calls
):
    """If the policy now sends the same prompt to a different model, the old answer is not reused."""
    post(api_key, QUESTION)  # auto -> mock, answer cached for model "mock"

    other = PolicyRouter(
        PolicySettings(tiers={"simple": "mock-429"}, default_tier="simple")
    )
    monkeypatch.setattr("app.api.chat.policy_router", other)
    r = post(api_key, QUESTION)
    assert r.headers["x-autopilot-cache"] == "miss"  # different resolved model, different scope
    assert r.headers["x-autopilot-routed-model"] == "mock-429"