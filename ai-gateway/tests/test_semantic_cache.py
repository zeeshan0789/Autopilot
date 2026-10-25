import numpy as np
import pytest

from app.cache.semantic_cache import CacheEntry, CacheScope, SemanticCache

DIM = 4
E1 = [1, 0, 0, 0]
E2 = [0, 1, 0, 0]
E3 = [0, 0, 1, 0]


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_scope(**overrides) -> CacheScope:
    fields = dict(
        namespace="key1",
        model="m",
        system_hash="s",
        temperature_bucket="low",
        prompt_version="none",
    )
    fields.update(overrides)
    return CacheScope(**fields)


def make_entry(clock: Clock, text: str = "hello", ttl: float = 60.0) -> CacheEntry:
    return CacheEntry(
        prompt_text=text,
        response={"answer": text},
        served_model="m",
        provider="mock",
        prompt_tokens=1,
        completion_tokens=3,
        estimated_cost_usd=None,
        created_at=clock(),
        ttl_seconds=ttl,
    )


def make_cache(clock: Clock, **kwargs) -> SemanticCache:
    return SemanticCache(dimension=DIM, clock=clock, **kwargs)


def test_identical_vector_is_found():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "first"))

    found = cache.lookup(make_scope(), E1)
    assert found is not None
    assert found.entry.prompt_text == "first"
    assert found.similarity == pytest.approx(1.0, abs=1e-5)


def test_unknown_scope_returns_none():
    clock = Clock()
    cache = make_cache(clock)
    assert cache.lookup(make_scope(), E1) is None


def test_similarity_is_cosine():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock))

    found = cache.lookup(make_scope(), [1, 1, 0, 0])
    assert found.similarity == pytest.approx(0.7071, abs=1e-3)


def test_vector_length_does_not_matter():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), [5, 0, 0, 0], make_entry(clock))

    found = cache.lookup(make_scope(), [0.1, 0, 0, 0])
    assert found.similarity == pytest.approx(1.0, abs=1e-5)


def test_closest_entry_wins():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "one"))
    cache.add(make_scope(), E2, make_entry(clock, "two"))

    found = cache.lookup(make_scope(), [0.1, 1, 0, 0])
    assert found.entry.prompt_text == "two"


def test_low_similarity_still_returns_a_candidate():
    """The cache only finds the closest entry. Applying a threshold is the caller's job."""
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock))

    found = cache.lookup(make_scope(), E2)
    assert found is not None
    assert found.similarity == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("namespace", "key2"),
        ("model", "other-model"),
        ("system_hash", "other-system"),
        ("temperature_bucket", "high"),
        ("prompt_version", "v2"),
    ],
)
def test_scopes_are_isolated(field, value):
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock))

    assert cache.lookup(make_scope(**{field: value}), E1) is None


def test_expired_entry_is_not_returned():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, ttl=60))

    clock.now += 61
    assert cache.lookup(make_scope(), E1) is None
    assert cache.size() == 0


def test_entry_is_valid_until_exactly_its_ttl():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, ttl=60))

    clock.now += 59.9
    assert cache.lookup(make_scope(), E1) is not None
    clock.now += 0.1
    assert cache.lookup(make_scope(), E1) is None


def test_expired_entry_does_not_hide_a_fresh_one():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "old", ttl=10))

    clock.now += 100
    cache.add(make_scope(), E1, make_entry(clock, "new", ttl=10))

    found = cache.lookup(make_scope(), E1)
    assert found.entry.prompt_text == "new"


def test_oldest_entries_are_dropped_when_full():
    clock = Clock()
    cache = make_cache(clock, max_entries_per_scope=2)
    cache.add(make_scope(), E1, make_entry(clock, "first"))
    cache.add(make_scope(), E2, make_entry(clock, "second"))
    cache.add(make_scope(), E3, make_entry(clock, "third"))

    assert cache.size() == 2
    newest = cache.lookup(make_scope(), E3)
    assert newest.entry.prompt_text == "third"
    assert newest.similarity == pytest.approx(1.0, abs=1e-5)

    # "first" is gone: its direction now matches nothing closely
    oldest = cache.lookup(make_scope(), E1)
    assert oldest.similarity == pytest.approx(0.0, abs=1e-5)


def test_zero_vector_is_rejected():
    clock = Clock()
    cache = make_cache(clock)
    with pytest.raises(ValueError):
        cache.add(make_scope(), [0, 0, 0, 0], make_entry(clock))


def test_wrong_dimension_is_rejected():
    clock = Clock()
    cache = make_cache(clock)
    with pytest.raises(ValueError):
        cache.lookup(make_scope(), [1, 0, 0])

        # ---------- lookup_many (added in step P2.4b) ----------


def test_lookup_many_returns_best_first():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "one"))
    cache.add(make_scope(), [1, 1, 0, 0], make_entry(clock, "diagonal"))
    cache.add(make_scope(), E2, make_entry(clock, "two"))

    found = cache.lookup_many(make_scope(), [1, 0.2, 0, 0], k=3)
    assert [c.entry.prompt_text for c in found] == ["one", "diagonal", "two"]
    similarities = [c.similarity for c in found]
    assert similarities == sorted(similarities, reverse=True)


def test_lookup_many_respects_k():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "one"))
    cache.add(make_scope(), E2, make_entry(clock, "two"))
    cache.add(make_scope(), E3, make_entry(clock, "three"))

    assert len(cache.lookup_many(make_scope(), E1, k=2)) == 2


def test_lookup_many_with_fewer_entries_than_k():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "only"))

    found = cache.lookup_many(make_scope(), E1, k=5)
    assert [c.entry.prompt_text for c in found] == ["only"]


def test_lookup_many_on_unknown_scope_or_empty_cache_is_empty():
    clock = Clock()
    cache = make_cache(clock)
    assert cache.lookup_many(make_scope(), E1) == []

    cache.add(make_scope(), E1, make_entry(clock))
    assert cache.lookup_many(make_scope(namespace="other"), E1) == []


def test_lookup_many_skips_expired_entries():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "old", ttl=10))
    clock.now += 11
    cache.add(make_scope(), E2, make_entry(clock, "new", ttl=10))

    found = cache.lookup_many(make_scope(), E1, k=5)
    assert [c.entry.prompt_text for c in found] == ["new"]


def test_lookup_returns_the_same_first_result_as_lookup_many():
    clock = Clock()
    cache = make_cache(clock)
    cache.add(make_scope(), E1, make_entry(clock, "one"))
    cache.add(make_scope(), E2, make_entry(clock, "two"))

    single = cache.lookup(make_scope(), [0.2, 1, 0, 0])
    many = cache.lookup_many(make_scope(), [0.2, 1, 0, 0], k=2)
    assert single.entry.prompt_text == many[0].entry.prompt_text == "two"
    assert single.similarity == pytest.approx(many[0].similarity)