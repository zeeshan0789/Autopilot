from decimal import Decimal

import numpy as np
import pytest

from app.cache.service import (
    CacheService,
    CacheSettings,
    conversation_text,
    looks_like_refusal,
    system_hash,
    temperature_bucket,
)
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)
from app.settings import load_cache_config

FRANCE = "What is the capital of France?"
FRANCE_PARAPHRASE = "Which city is the capital of France?"
APPLE_2024 = "What was Apple revenue in 2024?"
APPLE_2025 = "What was Apple revenue in 2025?"
APPLE_2025_SHORT = "Apple revenue for 2025?"

# Hand made 4 dimensional "embeddings". The dot product of two unit vectors is their
# cosine similarity, so each pair below has a known similarity.
VECTORS = {
    FRANCE: [1.0, 0.0, 0.0, 0.0],
    FRANCE_PARAPHRASE: [0.95, 0.3122499, 0.0, 0.0],  # similarity 0.95 to FRANCE
    "Unrelated question": [0.5, 0.8660254, 0.0, 0.0],  # similarity 0.50 to FRANCE
    APPLE_2024: [0.0, 0.0, 1.0, 0.0],
    APPLE_2025: [0.0, 0.0, 0.96, 0.28],  # similarity 0.96 to APPLE_2024
    APPLE_2025_SHORT: [0.0, 0.0, 0.99, 0.141],  # slightly closer to 2024 than to 2025
}


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


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


SETTINGS = CacheSettings(
    enabled=True,
    similarity_threshold=0.90,
    ttl_seconds=60,
    max_entries_per_scope=100,
    max_prompt_chars=200,
    max_temperature=0.7,
)


def make_service(settings=SETTINGS, embedder=None, clock=None):
    clock = clock or Clock()
    embedder = embedder or FakeEmbedder()
    return CacheService(settings, embedder, clock=clock), embedder, clock


def make_request(content=FRANCE, model="m", **fields) -> ChatCompletionRequest:
    messages = fields.pop("messages", None) or [{"role": "user", "content": content}]
    return ChatCompletionRequest(model=model, messages=messages, **fields)


def make_response(answer="Paris", model="m") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=Message(role="assistant", content=answer))],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def learn(service, request, answer="Paris", namespace="key1", **store_fields):
    """Look up (a miss), then store an answer, like the gateway will do."""
    lookup = service.lookup(request, namespace)
    stored = service.store(
        lookup,
        make_response(answer, model=request.model),
        **{
            "provider": "mock",
            "estimated_cost_usd": Decimal("0.001"),
            "fallback_used": False,
            **store_fields,
        },
    )
    return lookup, stored


# ---------- Hit, miss, blocked ----------


def test_empty_cache_is_a_miss_and_can_store():
    service, _, _ = make_service()
    lookup = service.lookup(make_request(), "key1")
    assert lookup.status == "miss"
    assert lookup.reason == "no_candidate"
    assert lookup.can_store


def test_identical_prompt_hits_after_store():
    service, _, _ = make_service()
    learn(service, make_request(), answer="Paris")

    lookup = service.lookup(make_request(), "key1")
    assert lookup.status == "hit"
    assert lookup.similarity == pytest.approx(1.0, abs=1e-4)
    assert lookup.entry.response["choices"][0]["message"]["content"] == "Paris"
    assert not lookup.can_store


def test_paraphrase_above_threshold_hits():
    service, _, _ = make_service()
    learn(service, make_request(FRANCE))

    lookup = service.lookup(make_request(FRANCE_PARAPHRASE), "key1")
    assert lookup.status == "hit"
    assert lookup.similarity == pytest.approx(0.95, abs=1e-3)


def test_below_threshold_is_a_miss():
    service, _, _ = make_service()
    learn(service, make_request(FRANCE))

    lookup = service.lookup(make_request("Unrelated question"), "key1")
    assert lookup.status == "miss"
    assert lookup.reason == "below_threshold"
    assert lookup.similarity == pytest.approx(0.5, abs=1e-3)
    assert lookup.can_store


def test_trap_pair_is_blocked_by_the_guard():
    """Embeddings say 0.96 similar, but the year differs."""
    service, _, _ = make_service()
    learn(service, make_request(APPLE_2024), answer="2024 answer")

    lookup = service.lookup(make_request(APPLE_2025), "key1")
    assert lookup.status == "blocked"
    assert lookup.reason == "number_mismatch"
    assert lookup.entry is None
    assert lookup.similarity == pytest.approx(0.96, abs=1e-3)
    assert lookup.can_store  # the new answer should be remembered too


def test_next_candidate_is_used_when_the_closest_is_blocked():
    service, _, _ = make_service()
    learn(service, make_request(APPLE_2024), answer="2024 answer")
    learn(service, make_request(APPLE_2025), answer="2025 answer")

    # Closest to the 2024 entry by similarity, but only the 2025 entry passes the guard
    lookup = service.lookup(make_request(APPLE_2025_SHORT), "key1")
    assert lookup.status == "hit"
    assert lookup.entry.response["choices"][0]["message"]["content"] == "2025 answer"


# ---------- Scopes ----------


@pytest.mark.parametrize(
    "other",
    [
        dict(namespace="key2"),
        dict(model="other-model"),
        dict(system="You are a pirate."),
        dict(temperature=0.5),  # "mid" bucket instead of "default"
    ],
)
def test_a_different_scope_never_hits(other):
    service, _, _ = make_service()
    learn(service, make_request())

    namespace = other.get("namespace", "key1")
    fields = {}
    if "model" in other:
        fields["model"] = other["model"]
    if "system" in other:
        fields["messages"] = [
            {"role": "system", "content": other["system"]},
            {"role": "user", "content": FRANCE},
        ]
    if "temperature" in other:
        fields["temperature"] = other["temperature"]

    lookup = service.lookup(make_request(**fields), namespace)
    assert lookup.status == "miss"


def test_same_system_prompt_shares_the_cache():
    service, _, _ = make_service()
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": FRANCE},
    ]
    learn(service, make_request(messages=messages))
    assert service.lookup(make_request(messages=messages), "key1").status == "hit"


# ---------- Bypass (the cache is not consulted at all) ----------


@pytest.mark.parametrize(
    "fields,content,reason",
    [
        (dict(), "What is the weather today?", "time_sensitive"),
        (dict(), "Who is the current CEO of Netflix?", "time_sensitive"),
        (dict(temperature=1.0), FRANCE, "high_temperature"),
        (dict(max_tokens=50), FRANCE, "max_tokens_set"),
        (dict(stream=True), FRANCE, "stream"),
        (dict(autopilot={"cache": False}), FRANCE, "opt_out"),
        (dict(), "x" * 201, "prompt_too_long"),
        (dict(), "   ", "empty_prompt"),
    ],
)
def test_bypass_reasons(fields, content, reason):
    service, embedder, _ = make_service()
    lookup = service.lookup(make_request(content, **fields), "key1")
    assert lookup.status == "bypass"
    assert lookup.reason == reason
    assert not lookup.can_store
    assert embedder.calls == 0  # bypass is decided before any embedding work


def test_disabled_cache_bypasses_everything():
    settings = CacheSettings(**{**SETTINGS.__dict__, "enabled": False})
    service, _, _ = make_service(settings=settings)
    assert service.lookup(make_request(), "key1").reason == "disabled"


def test_temperature_at_the_limit_is_still_cached():
    service, _, _ = make_service()
    lookup = service.lookup(make_request(temperature=0.7), "key1")
    assert lookup.status == "miss"


# ---------- Storing rules ----------


def test_fallback_answers_are_not_stored():
    service, _, _ = make_service()
    _, stored = learn(service, make_request(), fallback_used=True)
    assert stored is False
    assert service.size() == 0


def test_empty_answers_are_not_stored():
    service, _, _ = make_service()
    _, stored = learn(service, make_request(), answer="   ")
    assert stored is False
    assert service.size() == 0


def test_cannot_store_after_a_bypass():
    service, _, _ = make_service()
    lookup = service.lookup(make_request("What is the weather today?"), "key1")
    assert service.store(
        lookup,
        make_response(),
        provider="mock",
        estimated_cost_usd=None,
        fallback_used=False,
    ) is False


def test_cannot_store_after_a_hit():
    service, _, _ = make_service()
    learn(service, make_request())
    lookup = service.lookup(make_request(), "key1")
    assert lookup.status == "hit"
    assert service.store(
        lookup,
        make_response(),
        provider="mock",
        estimated_cost_usd=None,
        fallback_used=False,
    ) is False
    assert service.size() == 1


def test_entry_remembers_what_the_original_call_used():
    service, _, _ = make_service()
    learn(service, make_request(), provider="groq", estimated_cost_usd=Decimal("0.002"))
    entry = service.lookup(make_request(), "key1").entry
    assert entry.provider == "groq"
    assert entry.prompt_tokens == 10
    assert entry.completion_tokens == 5
    assert entry.estimated_cost_usd == Decimal("0.002")
    assert entry.served_model == "m"


def test_expired_entries_are_not_served():
    service, _, clock = make_service()
    learn(service, make_request())
    clock.now += 61
    assert service.lookup(make_request(), "key1").status == "miss"

# ---------- Refusals are not cached ----------


@pytest.mark.parametrize(
    "refusal",
    [
        "I'm sorry, but I can't provide that.",
        "I\u2019m sorry, but I can\u2019t help with that.",  # curly apostrophes
        "I cannot answer that question.",
        "**Sorry**, I don't know.",
        '"I can\'t help with that."',
        "Unfortunately, I don't have that information.",
        "As an AI, I can't browse the internet.",
        "I don't have access to that information.",
        "\n  Sorry, no.",
    ],
)
def test_refusals_are_not_stored(refusal):
    service, _, _ = make_service()
    _, stored = learn(service, make_request(), answer=refusal)
    assert stored is False
    assert service.size() == 0


@pytest.mark.parametrize(
    "answer",
    [
        "Paris",
        "The capital of France is **Paris**.",
        "The answer is 42.",
        "It can't be negative, so the answer is 5.",  # contains "can't" but does not start with it
        pytest.param(
            "Sorry " + "this is a long and useful answer. " * 20,
            id="long answer that starts with Sorry is real content",
        ),
    ],
)
def test_real_answers_are_stored(answer):
    service, _, _ = make_service()
    _, stored = learn(service, make_request(), answer=answer)
    assert stored is True
    assert service.size() == 1


def test_looks_like_refusal_ignores_text_length_only_above_the_limit():
    assert looks_like_refusal("Sorry, no.")
    assert not looks_like_refusal("Sorry. " + "x" * 400)

# ---------- Failing open ----------


def test_embedding_failure_bypasses_instead_of_raising():
    service, _, _ = make_service(embedder=BrokenEmbedder())
    lookup = service.lookup(make_request(), "key1")
    assert lookup.status == "bypass"
    assert lookup.reason == "embedding_error"


# ---------- Response rebuilding ----------


def test_response_from_entry_has_fresh_id_and_same_content():
    service, _, _ = make_service()
    original = make_response("Paris")
    lookup = service.lookup(make_request(), "key1")
    service.store(
        lookup, original, provider="mock", estimated_cost_usd=None, fallback_used=False
    )

    entry = service.lookup(make_request(), "key1").entry
    rebuilt = service.response_from(entry)
    assert rebuilt.choices[0].message.content == "Paris"
    assert rebuilt.usage == original.usage
    assert rebuilt.model == original.model
    assert rebuilt.id != original.id


# ---------- Building the key ----------


def test_single_user_message_is_embedded_as_plain_text():
    assert conversation_text([Message(role="user", content="  hello ")]) == "hello"


def test_conversation_with_history_keeps_roles_and_skips_system():
    messages = [
        Message(role="system", content="Be brief."),
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello"),
        Message(role="user", content="Capital of France?"),
    ]
    assert conversation_text(messages) == (
        "user: Hi\nassistant: Hello\nuser: Capital of France?"
    )


def test_system_hash():
    assert system_hash([Message(role="user", content="hi")]) == "none"
    a = system_hash([Message(role="system", content="Be brief.")])
    b = system_hash([Message(role="system", content="  Be brief. ")])
    c = system_hash([Message(role="system", content="Be long.")])
    assert a == b != c


@pytest.mark.parametrize(
    "value,bucket",
    [(None, "default"), (0.0, "low"), (0.3, "low"), (0.5, "mid"), (0.7, "mid"), (0.9, "high")],
)
def test_temperature_bucket(value, bucket):
    assert temperature_bucket(value) == bucket


# ---------- Settings ----------


def test_cache_yaml_builds_valid_settings():
    settings = CacheSettings(**load_cache_config())
    assert 0 < settings.similarity_threshold <= 1


@pytest.mark.parametrize(
    "bad",
    [
        dict(similarity_threshold=0),
        dict(similarity_threshold=1.5),
        dict(ttl_seconds=0),
        dict(max_entries_per_scope=0),
        dict(max_prompt_chars=0),
        dict(max_temperature=-1),
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        CacheSettings(**bad)