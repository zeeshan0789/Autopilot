import json

from fastapi.testclient import TestClient

from app.core.auth import KeyStore
from app.main import app

client = TestClient(app)

BODY = {"model": "mock", "messages": [{"role": "user", "content": "hello"}]}


# ---------- KeyStore (seedha) ----------


def test_plaintext_key_is_not_stored(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("x")
    assert key not in store.path.read_text()  # file mein sirf hash


def test_verify_valid_and_invalid_key(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("alice", rate_limit_per_min=7)
    info = store.verify(key)
    assert info is not None
    assert info.name == "alice"
    assert info.rate_limit_per_min == 7
    assert store.verify("ap-galat") is None


# ---------- HTTP (guard) ----------


def test_health_needs_no_key():
    assert client.get("/health").status_code == 200


def test_missing_key_gives_401(api_key):
    r = client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 401


def test_wrong_key_gives_401(api_key):
    r = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={"Authorization": "Bearer ap-galat-key"},
    )
    assert r.status_code == 401


def test_non_bearer_scheme_gives_401(api_key):
    r = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={"Authorization": f"Basic {api_key}"},
    )
    assert r.status_code == 401


def test_revoked_key_gives_401(tmp_path, monkeypatch):
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("old")
    data = json.loads(store.path.read_text())
    for entry in data.values():
        entry["active"] = False
    store.path.write_text(json.dumps(data))
    monkeypatch.setattr("app.core.auth.key_store", store)

    r = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 401


def test_valid_key_works(api_key):
    r = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 200