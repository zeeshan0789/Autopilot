import types

import pytest
from fastapi.testclient import TestClient

from app.core.policy import PolicyRouter, PolicySettings, Rule
from app.main import app
from app.mlrouter.runtime import MlDecision

client = TestClient(app)

QUESTION = "What is the capital of France?"
HARD = "Please analyze the trade offs of two database designs."


class FakeRuntime:
    def __init__(self, decision, alias="smart", shadow=True, enabled=True):
        self.settings = types.SimpleNamespace(enabled=enabled, alias=alias, shadow=shadow)
        self.decision = decision
        self.calls = []

    def decide(self, messages):
        self.calls.append(messages)
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


@pytest.fixture(autouse=True)
def rule_policy(monkeypatch):
    """Simple prompts go to 'mock'. Prompts with the word analyze go to 'mock-down' (which fails over to 'mock')."""
    router = PolicyRouter(
        PolicySettings(
            tiers={"simple": "mock", "complex": "mock-down"},
            default_tier="simple",
            rules=(Rule("analysis", "complex", {"any_keywords": ["analyze"]}),),
        )
    )
    monkeypatch.setattr("app.api.chat.policy_router", router)


def use(monkeypatch, runtime):
    monkeypatch.setattr("app.api.chat.ml_runtime", runtime)
    return runtime


def post(key, content, model="smart"):
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    return client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})


def test_smart_uses_the_pick_of_the_learned_router(api_key, traces, monkeypatch):
    use(monkeypatch, FakeRuntime(MlDecision("mock", "router", 0.99, 0.9)))
    r = post(api_key, QUESTION)
    assert r.status_code == 200
    assert r.headers["x-autopilot-route"] == "ml_router"
    assert r.headers["x-autopilot-routed-model"] == "mock"
    t = traces[0]
    assert (t.requested_model, t.routed_model, t.policy_rule) == ("smart", "mock", "ml_router")
    assert (t.ml_pick, t.ml_status, t.ml_prob, t.ml_similarity) == ("mock", "router", 0.99, 0.9)


def test_the_messages_are_given_to_the_router_as_plain_dictionaries(api_key, monkeypatch):
    runtime = use(monkeypatch, FakeRuntime(MlDecision("mock", "router")))
    post(api_key, QUESTION)
    assert runtime.calls == [[{"role": "user", "content": QUESTION}]]


def test_an_unfamiliar_prompt_is_decided_by_the_rule_policy(api_key, traces, monkeypatch):
    use(monkeypatch, FakeRuntime(MlDecision(None, "novel", 0.5, 0.2)))
    r = post(api_key, QUESTION)
    assert r.headers["x-autopilot-route"] == "ml_novel>default"
    assert r.headers["x-autopilot-routed-model"] == "mock"
    assert traces[0].ml_status == "novel" and traces[0].ml_pick is None


def test_the_rule_policy_still_applies_its_own_rules_after_a_decline(api_key, traces, monkeypatch):
    use(monkeypatch, FakeRuntime(MlDecision(None, "novel", 0.5, 0.2)))
    r = post(api_key, HARD)
    assert r.headers["x-autopilot-route"] == "ml_novel>analysis"
    assert r.headers["x-autopilot-routed-model"] == "mock-down"
    assert r.status_code == 200 and traces[0].fallback_used is True  # mock-down failed over to mock


def test_a_crashing_router_does_not_fail_the_request(api_key, traces, monkeypatch):
    use(monkeypatch, FakeRuntime(RuntimeError("boom")))
    r = post(api_key, QUESTION)
    assert r.status_code == 200
    assert r.headers["x-autopilot-route"] == "ml_error>default"
    assert traces[0].ml_status == "error"


def test_a_pick_without_a_provider_falls_back_to_the_rule_policy(api_key, monkeypatch):
    use(monkeypatch, FakeRuntime(MlDecision("model-nobody-serves", "router", 0.99, 0.9)))
    r = post(api_key, QUESTION)
    assert r.status_code == 200
    assert r.headers["x-autopilot-route"] == "ml_router>default"
    assert r.headers["x-autopilot-routed-model"] == "mock"


def test_unavailable_router_means_the_rule_policy_decides(api_key, traces, monkeypatch):
    use(monkeypatch, FakeRuntime(MlDecision(None, "unavailable")))
    assert post(api_key, QUESTION).headers["x-autopilot-route"] == "ml_unavailable>default"
    assert traces[0].ml_prob is None


def test_auto_is_scored_in_shadow_mode_without_changing_the_route(api_key, traces, monkeypatch):
    runtime = use(monkeypatch, FakeRuntime(MlDecision("strong-model", "router", 0.1, 0.9)))
    r = post(api_key, QUESTION, model="auto")
    assert r.headers["x-autopilot-route"] == "default"  # the rule policy decided
    assert r.headers["x-autopilot-routed-model"] == "mock"
    assert len(runtime.calls) == 1
    t = traces[0]
    assert (t.policy_rule, t.routed_model) == ("default", "mock")
    assert (t.ml_pick, t.ml_status) == ("strong-model", "router")  # what it would have picked


def test_shadow_mode_can_be_switched_off(api_key, traces, monkeypatch):
    runtime = use(monkeypatch, FakeRuntime(MlDecision("mock", "router"), shadow=False))
    post(api_key, QUESTION, model="auto")
    assert runtime.calls == [] and traces[0].ml_status is None


def test_an_explicit_model_never_reaches_the_router(api_key, traces, monkeypatch):
    runtime = use(monkeypatch, FakeRuntime(MlDecision("mock", "router")))
    r = post(api_key, QUESTION, model="mock")
    assert r.status_code == 200 and runtime.calls == []
    assert traces[0].ml_status is None and traces[0].policy_rule == "explicit"


def test_with_the_router_disabled_smart_is_an_unknown_model(api_key, monkeypatch):
    runtime = use(monkeypatch, FakeRuntime(MlDecision("mock", "router"), enabled=False))
    assert post(api_key, QUESTION).status_code == 404
    assert runtime.calls == []