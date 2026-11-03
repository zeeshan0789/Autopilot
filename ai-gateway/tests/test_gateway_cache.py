from decimal import Decimal

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cache.service import CacheLookup, CacheService, CacheSettings
from app.core.auth import KeyStore
from app.main import app

client = TestClient(app)

FRANCE = "What is the capital of France?"
FRANCE_PARAPHRASE = "Which city is the capital of France?"
APPLE_2024 = "What was Apple revenue in 2024?"
APPLE_2025 = "What was Apple revenue in 2025?"

# Hand made 4 dimensional embeddings with known similarities (see test_cache_service.py)
VECTORS = {
    FRANCE: [1.0, 0.0, 0.0, 0.0],
    FRANCE_PARAPHRASE: [0.95, 0.3122499, 0.0, 0.0],  # similarity 0.95 to FRANCE
    APPLE_2024: [0.0, 0.0, 1.0, 0.0],
    APPLE_2025: [0.0, 0.0, 0.96, 0.28],  # similarity 0.96 to APPLE_2024
}

SETTINGS = CacheSettings(
    enabled=True,
    similarity_threshold=0.90,
    ttl_seconds=3600,
    max_entries_per_scope=100,
    max_prompt_chars=500,
    max_temperature=0.7,
)


class FakeEmbedder:
    dimension = 4

    def __init__(self):
        self.calls = 0

    def encode(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.asarray(VECTORS[text], dtype=np.float32)


class BrokenEmbedder:
    dimension = 4

    def encode(self, text: str) -> np.ndarray:
        raise RuntimeError("model could not be loaded")


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture(autouse=True)
def cache(cache_off_by_default, monkeypatch, embedder):
    """Install a working cache (the shared conftest fixture runs first, this one wins)."""
    service = CacheService(SETTINGS, embedder)
    monkeypatch.setattr("app.api.chat.cache_service", service)
    return service


@pytest.fixture
def executor_calls(monkeypatch):
    """Records every time the gateway actually goes to a provider."""
    from app.api import chat as chat_module

    real = chat_module.execute_with_fallback
    calls = []

    async def counting(chain, request, *args, **kwargs):
        calls.append(request.model)
        return await real(chain, request, *args, **kwargs)

    monkeypatch.setattr("app.api.chat.execute_with_fallback", counting)
    return calls


def post(key: str, content: str, model: str = "mock", **extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    )


def answer(response) -> str:
    return response.json()["choices"][0]["message"]["content"]


# ---------- The basic cycle: miss, then hit ----------


def test_second_identical_request_is_served_from_the_cache(api_key, executor_calls):
    first = post(api_key, FRANCE)
    second = post(api_key, FRANCE)

    assert first.status_code == second.status_code == 200
    assert first.headers["x-autopilot-cache"] == "miss"
    assert second.headers["x-autopilot-cache"] == "hit"
    assert answer(first) == answer(second)
    assert first.json()["id"] != second.json()["id"]
    assert executor_calls == ["mock"]  # the provider was only called once


def test_hit_headers(api_key):
    post(api_key, FRANCE)
    hit = post(api_key, FRANCE)
    assert hit.headers["x-autopilot-provider"] == "cache"
    assert hit.headers["x-autopilot-model"] == "mock"
    assert hit.headers["x-autopilot-fallback"] == "false"
    assert hit.headers["x-request-id"]


def test_hit_returns_the_original_usage(api_key):
    first = post(api_key, FRANCE)
    second = post(api_key, FRANCE)
    assert second.json()["usage"] == first.json()["usage"]


def test_miss_trace(api_key, traces):
    post(api_key, FRANCE)
    t = traces[0]
    assert t.cache_status == "miss"
    assert t.cache_reason == "no_candidate"
    assert t.cache_similarity is None
    assert t.saved_tokens is None
    assert t.saved_cost_usd is None
    assert t.provider == "mock"


def test_hit_trace_records_what_was_saved(api_key, traces):
    first = post(api_key, FRANCE)
    post(api_key, FRANCE)
    miss, hit = traces

    assert hit.status == "success"
    assert hit.cache_status == "hit"
    assert hit.cache_reason == "hit"
    assert hit.cache_similarity == pytest.approx(1.0, abs=1e-4)
    assert hit.provider == "cache"
    assert hit.served_model == "mock"
    # What this request used: nothing
    assert hit.prompt_tokens == hit.completion_tokens == hit.total_tokens == 0
    assert hit.estimated_cost_usd == Decimal(0)
    # What the cache avoided: the original call
    assert hit.saved_tokens == first.json()["usage"]["total_tokens"]
    assert hit.saved_cost_usd == miss.estimated_cost_usd
    assert hit.cache_entry_id
    assert hit.request_id != miss.request_id


def test_paraphrase_is_served_the_cached_answer(api_key, executor_calls):
    first = post(api_key, FRANCE)
    second = post(api_key, FRANCE_PARAPHRASE)

    assert second.headers["x-autopilot-cache"] == "hit"
    # The answer is the cached one (the mock provider would have echoed the paraphrase)
    assert answer(second) == answer(first) == f"Mock reply: {FRANCE}"
    assert executor_calls == ["mock"]


# ---------- The guard protects against wrong answers ----------


def test_changed_year_is_not_served_from_the_cache(api_key, traces, executor_calls):
    post(api_key, APPLE_2024)
    second = post(api_key, APPLE_2025)

    assert second.headers["x-autopilot-cache"] == "blocked"
    assert answer(second) == f"Mock reply: {APPLE_2025}"  # a fresh answer, not 2024's
    assert executor_calls == ["mock", "mock"]
    assert traces[1].cache_status == "blocked"
    assert traces[1].cache_reason == "number_mismatch"
    assert traces[1].cache_similarity == pytest.approx(0.96, abs=1e-3)


def test_blocked_request_is_remembered_for_next_time(api_key, executor_calls):
    post(api_key, APPLE_2024)
    post(api_key, APPLE_2025)
    third = post(api_key, APPLE_2025)

    assert third.headers["x-autopilot-cache"] == "hit"
    assert answer(third) == f"Mock reply: {APPLE_2025}"
    assert executor_calls == ["mock", "mock"]


# ---------- Isolation ----------


def test_cache_is_not_shared_between_api_keys(tmp_path, monkeypatch, executor_calls):
    store = KeyStore(tmp_path / "two_keys.json")
    key_a = store.create("a")
    key_b = store.create("b")
    monkeypatch.setattr("app.core.auth.key_store", store)

    assert post(key_a, FRANCE).headers["x-autopilot-cache"] == "miss"
    assert post(key_b, FRANCE).headers["x-autopilot-cache"] == "miss"
    assert post(key_a, FRANCE).headers["x-autopilot-cache"] == "hit"
    assert executor_calls == ["mock", "mock"]


# ---------- What is never cached ----------


def test_time_sensitive_prompts_bypass_the_cache(api_key, traces, embedder, executor_calls):
    text = "What is the weather today?"
    first = post(api_key, text)
    second = post(api_key, text)

    assert first.headers["x-autopilot-cache"] == "bypass"
    assert second.headers["x-autopilot-cache"] == "bypass"
    assert executor_calls == ["mock", "mock"]
    assert traces[0].cache_reason == "time_sensitive"
    assert embedder.calls == 0


def test_high_temperature_bypasses_the_cache(api_key, traces):
    post(api_key, FRANCE, temperature=1.0)
    again = post(api_key, FRANCE, temperature=1.0)
    assert again.headers["x-autopilot-cache"] == "bypass"
    assert traces[1].cache_reason == "high_temperature"


def test_fallback_answers_are_not_cached(api_key, executor_calls):
    first = post(api_key, FRANCE, model="mock-down")
    second = post(api_key, FRANCE, model="mock-down")

    assert first.headers["x-autopilot-fallback"] == "true"
    assert second.headers["x-autopilot-cache"] == "miss"  # nothing was stored
    assert second.headers["x-autopilot-fallback"] == "true"
    assert executor_calls == ["mock-down", "mock-down"]

def test_refusals_are_not_cached(api_key, traces, monkeypatch):
    """A model that says "I can't help" once may answer properly next time."""
    from app.core.executor import ExecutionResult
    from app.schemas.openai import ChatCompletionResponse, Choice, Message, Usage

    calls = []

    async def refusing(chain, request, *args, **kwargs):
        calls.append(request.model)
        return ExecutionResult(
            response=ChatCompletionResponse(
                model="mock",
                choices=[
                    Choice(
                        message=Message(
                            role="assistant",
                            content="I\u2019m sorry, but I can\u2019t provide that.",
                        )
                    )
                ],
                usage=Usage(prompt_tokens=5, completion_tokens=8, total_tokens=13),
            ),
            provider="mock",
            model="mock",
            fallback_used=False,
        )

    monkeypatch.setattr("app.api.chat.execute_with_fallback", refusing)

    first = post(api_key, FRANCE)
    second = post(api_key, FRANCE)

    assert first.status_code == second.status_code == 200
    assert first.headers["x-autopilot-cache"] == "miss"
    assert second.headers["x-autopilot-cache"] == "miss"  # the refusal was not remembered
    assert len(calls) == 2

# ---------- Errors and special requests ----------


def test_failed_request_is_traced_as_a_miss_and_not_stored(api_key, traces):
    r = post(api_key, FRANCE, model="mock-dead")
    assert r.status_code == 502
    assert traces[0].status == "error"
    assert traces[0].cache_status == "miss"

    again = post(api_key, FRANCE, model="mock-dead")
    assert again.status_code == 502  # still not served from the cache


def test_unknown_model_never_reaches_the_cache(api_key, traces, embedder):
    r = post(api_key, FRANCE, model="no-such-model")
    assert r.status_code == 404
    assert traces[0].cache_status is None
    assert embedder.calls == 0


def test_stream_request_never_reaches_the_cache(api_key, traces, embedder):
    r = post(api_key, FRANCE, stream=True)
    assert r.status_code == 400
    assert traces[0].cache_status is None
    assert embedder.calls == 0


# ---------- The cache must never break a request ----------


def test_embedding_failure_does_not_fail_the_request(api_key, traces, monkeypatch):
    broken = CacheService(SETTINGS, BrokenEmbedder())
    monkeypatch.setattr("app.api.chat.cache_service", broken)

    r = post(api_key, FRANCE)
    assert r.status_code == 200
    assert r.headers["x-autopilot-cache"] == "bypass"
    assert traces[0].cache_reason == "embedding_error"


def test_a_crashing_lookup_does_not_fail_the_request(api_key, traces, monkeypatch):
    class ExplodingLookup(CacheService):
        def lookup(self, request, namespace):
            raise RuntimeError("unexpected bug in the cache")

    monkeypatch.setattr(
        "app.api.chat.cache_service", ExplodingLookup(SETTINGS, FakeEmbedder())
    )
    r = post(api_key, FRANCE)
    assert r.status_code == 200
    assert traces[0].cache_status == "bypass"
    assert traces[0].cache_reason == "cache_error"


def test_a_crashing_store_does_not_fail_the_request(api_key, traces, monkeypatch):
    class ExplodingStore(CacheService):
        def store(self, *args, **kwargs):
            raise RuntimeError("unexpected bug while storing")

    monkeypatch.setattr(
        "app.api.chat.cache_service", ExplodingStore(SETTINGS, FakeEmbedder())
    )
    r = post(api_key, FRANCE)
    assert r.status_code == 200
    assert traces[0].status == "success"
    assert traces[0].cache_status == "miss"


def test_disabled_cache_serves_every_request_from_the_provider(
    api_key, traces, monkeypatch, executor_calls
):
    off = CacheService(CacheSettings(enabled=False), FakeEmbedder())
    monkeypatch.setattr("app.api.chat.cache_service", off)

    post(api_key, FRANCE)
    second = post(api_key, FRANCE)
    assert second.headers["x-autopilot-cache"] == "bypass"
    assert traces[1].cache_reason == "disabled"
    assert executor_calls == ["mock", "mock"]