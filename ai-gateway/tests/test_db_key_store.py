import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.auth import ApiKeyInfo, AuthUnavailable, DbKeyStore, KeyStore, hash_key
from app.main import app

client = TestClient(app)

INFO = ApiKeyInfo(key_id="abcd1234", name="demo", rate_limit_per_min=30)
BODY = {"model": "mock", "messages": [{"role": "user", "content": "hello"}]}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeDbStore(DbKeyStore):
    """A DbKeyStore that reads from a dictionary instead of Postgres."""

    def __init__(self, rows=None, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows if rows is not None else {}
        self.loads = 0
        self.fail = False

    async def _load(self, key_hash):
        self.loads += 1
        if self.fail:
            raise OSError("database is down")
        return self.rows.get(key_hash)


def authenticate(store, key):
    return asyncio.run(store.authenticate(key))


def test_hash_matches_the_known_sha256_vector():
    """The Node.js web backend hashes keys the same way, so this must never change."""
    assert hash_key("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_a_stored_key_is_accepted():
    store = FakeDbStore({hash_key("ap-good"): INFO})
    assert authenticate(store, "ap-good") == INFO


def test_an_unknown_key_is_rejected():
    assert authenticate(FakeDbStore(), "ap-nope") is None


def test_answers_are_remembered_for_the_ttl():
    clock = Clock()
    store = FakeDbStore({hash_key("ap-good"): INFO}, ttl_seconds=10, clock=clock)
    authenticate(store, "ap-good")
    authenticate(store, "ap-good")
    assert store.loads == 1
    clock.now += 11
    authenticate(store, "ap-good")
    assert store.loads == 2


def test_a_revoked_key_stops_working_after_the_ttl():
    clock = Clock()
    store = FakeDbStore({hash_key("ap-good"): INFO}, ttl_seconds=10, clock=clock)
    assert authenticate(store, "ap-good") == INFO
    store.rows.clear()  # revoked in the database
    assert authenticate(store, "ap-good") == INFO  # still remembered
    clock.now += 11
    assert authenticate(store, "ap-good") is None


def test_a_known_key_keeps_working_when_the_database_is_down():
    clock = Clock()
    store = FakeDbStore({hash_key("ap-good"): INFO}, ttl_seconds=10, clock=clock)
    authenticate(store, "ap-good")
    clock.now += 11
    store.fail = True
    assert authenticate(store, "ap-good") == INFO


def test_an_unknown_key_with_the_database_down_is_unavailable_not_allowed():
    store = FakeDbStore({hash_key("ap-good"): INFO})
    store.fail = True
    with pytest.raises(AuthUnavailable):
        authenticate(store, "ap-good")  # never seen before, so nothing is remembered
    with pytest.raises(AuthUnavailable):
        authenticate(store, "ap-other")


def test_the_memory_of_the_store_is_bounded():
    store = FakeDbStore(max_cached=3)
    for i in range(10):
        authenticate(store, f"ap-key-{i}")
    assert len(store._cache) <= 3


def test_the_file_key_store_has_the_same_interface(tmp_path):
    file_store = KeyStore(tmp_path / "keys.json")
    key = file_store.create("old", rate_limit_per_min=5)
    info = asyncio.run(file_store.authenticate(key))
    assert (info.name, info.rate_limit_per_min) == ("old", 5)
    assert asyncio.run(file_store.authenticate("ap-nope")) is None


# ---------- Through the gateway ----------


def test_gateway_accepts_a_key_from_the_database_store(monkeypatch):
    store = FakeDbStore({hash_key("ap-good"): INFO})
    monkeypatch.setattr("app.core.auth.key_store", store)
    r = client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer ap-good"})
    assert r.status_code == 200


def test_gateway_rejects_a_key_the_database_does_not_know(monkeypatch):
    monkeypatch.setattr("app.core.auth.key_store", FakeDbStore())
    r = client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer ap-nope"})
    assert r.status_code == 401


def test_gateway_answers_503_when_keys_cannot_be_checked(monkeypatch):
    store = FakeDbStore()
    store.fail = True
    monkeypatch.setattr("app.core.auth.key_store", store)
    r = client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer ap-anything"})
    assert r.status_code == 503
    assert r.json()["detail"] == "authentication service unavailable"