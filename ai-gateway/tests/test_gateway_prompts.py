import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cache.service import CacheService, CacheSettings
from app.core.prompts import PromptStore, PromptVersionData
from app.main import app

client = TestClient(app)

QUESTION = "What is the capital of France?"
TEMPLATED = "Please answer: " + QUESTION


class FakeStore(PromptStore):
    def __init__(self, data):
        super().__init__()
        self.data = data

    async def _load(self, name):
        return self.data.get(name)


class DownStore(PromptStore):
    async def _load(self, name):
        raise OSError("database is down")


def version(number, weight=1, system=None, template=None):
    return PromptVersionData(number, system or f"You are version {number}.", template, weight)


@pytest.fixture(autouse=True)
def store(monkeypatch):
    fake = FakeStore(
        {
            "qa": [version(1, 50), version(2, 50, template="Please answer: {{input}}")],
            "only_v2": [version(1, 0), version(2, 100)],
            "greeter": [version(1, 100, system="You greet {{name}} warmly.")],
        }
    )
    monkeypatch.setattr("app.api.chat.prompt_store", fake)
    return fake


@pytest.fixture
def seen(monkeypatch):
    """The messages that really reach the providers."""
    from app.api import chat as chat_module

    real = chat_module.execute_with_fallback
    captured = []

    async def capturing(chain, request, *args, **kwargs):
        captured.append(request.messages)
        return await real(chain, request, *args, **kwargs)

    monkeypatch.setattr("app.api.chat.execute_with_fallback", capturing)
    return captured


@pytest.fixture
def cache(monkeypatch):
    class FourDimensions:
        dimension = 4

        def encode(self, text):
            return np.asarray({QUESTION: [1, 0, 0, 0], TEMPLATED: [1, 0, 0, 0]}[text], dtype=np.float32)

    settings = CacheSettings(
        similarity_threshold=0.90,
        ttl_seconds=3600,
        max_entries_per_scope=100,
        max_prompt_chars=500,
        max_temperature=0.7,
    )
    monkeypatch.setattr("app.api.chat.cache_service", CacheService(settings, FourDimensions()))


def post(key, content, model="mock", **extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})


def answer(response):
    return response.json()["choices"][0]["message"]["content"]


def test_a_request_without_a_prompt_is_unchanged(api_key, traces):
    r = post(api_key, QUESTION)
    assert r.status_code == 200
    assert "x-autopilot-prompt" not in r.headers
    assert traces[0].prompt_name is None and traces[0].prompt_version is None


def test_a_forced_version_is_applied_and_traced(api_key, traces):
    r = post(api_key, QUESTION, autopilot={"prompt": "qa", "version": 2})
    assert r.status_code == 200
    assert answer(r) == f"Mock reply: {TEMPLATED}"  # the user template was applied
    assert r.headers["x-autopilot-prompt"] == "qa:2"
    assert (traces[0].prompt_name, traces[0].prompt_version) == ("qa", 2)


def test_the_prompt_system_message_is_sent_to_the_provider(api_key, seen):
    post(api_key, "hi", autopilot={"prompt": "greeter", "variables": {"name": "Sara"}})
    messages = seen[0]
    assert (messages[0].role, messages[0].content) == ("system", "You greet Sara warmly.")
    assert (messages[1].role, messages[1].content) == ("user", "hi")


def test_the_callers_own_system_message_is_kept_after_the_prompt(api_key, seen):
    body = {
        "model": "mock",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ],
        "autopilot": {"prompt": "qa", "version": 1},
    }
    client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {api_key}"})
    assert [m.content for m in seen[0]] == ["You are version 1.", "Be brief.", "hi"]


def test_a_version_with_weight_zero_is_never_served(api_key):
    served = {post(api_key, "hi", autopilot={"prompt": "only_v2"}).headers["x-autopilot-prompt"] for _ in range(10)}
    assert served == {"only_v2:2"}


def test_the_split_serves_both_versions(api_key):
    served = {post(api_key, "hi", autopilot={"prompt": "qa"}).headers["x-autopilot-prompt"] for _ in range(40)}
    assert served == {"qa:1", "qa:2"}


def test_a_user_always_sees_the_same_version(api_key):
    served = {
        post(api_key, "hi", autopilot={"prompt": "qa"}, user="sara").headers["x-autopilot-prompt"]
        for _ in range(15)
    }
    assert len(served) == 1


def test_missing_variable_gives_400(api_key, traces):
    r = post(api_key, "hi", autopilot={"prompt": "greeter"})
    assert r.status_code == 400
    assert "missing variable 'name'" in r.json()["detail"]
    assert traces[0].status == "error"


def test_unknown_prompt_and_unknown_version_give_404(api_key):
    assert post(api_key, "hi", autopilot={"prompt": "nope"}).status_code == 404
    assert post(api_key, "hi", autopilot={"prompt": "qa", "version": 9}).status_code == 404


def test_unavailable_prompt_store_gives_503(api_key, monkeypatch):
    monkeypatch.setattr("app.api.chat.prompt_store", DownStore())
    assert post(api_key, "hi", autopilot={"prompt": "qa"}).status_code == 503


def test_cache_never_mixes_answers_of_different_prompt_versions(api_key, cache):
    first = post(api_key, QUESTION, autopilot={"prompt": "qa", "version": 1})
    again = post(api_key, QUESTION, autopilot={"prompt": "qa", "version": 1})
    other = post(api_key, QUESTION, autopilot={"prompt": "qa", "version": 2})
    assert first.headers["x-autopilot-cache"] == "miss"
    assert again.headers["x-autopilot-cache"] == "hit"
    assert other.headers["x-autopilot-cache"] == "miss"  # same words, different version