import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.auth import ApiKeyInfo
from app.core.prompts import PromptStore, PromptVersionData
from app.core.quota import QuotaResult
from app.main import app

client = TestClient(app)

A = ApiKeyInfo(key_id="aaaaaaaa", name="a", rate_limit_per_min=60, user_id=1, daily_request_limit=2, max_tokens_cap=100)
B = ApiKeyInfo(key_id="bbbbbbbb", name="b", rate_limit_per_min=60, user_id=2, daily_request_limit=2, max_tokens_cap=100)
LEGACY = ApiKeyInfo(key_id="cccccccc", name="old", rate_limit_per_min=60)  # no owner, no quota, no cap


class FakeKeys:
    def __init__(self, infos):
        self.infos = infos

    async def authenticate(self, key):
        return self.infos.get(key)


class FakeQuota:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.calls = []

    async def consume(self, user_id, limit):
        self.calls.append((user_id, limit))
        return QuotaResult(self.allowed, limit, 1 if self.allowed else 0, 3600)


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setattr("app.core.auth.key_store", FakeKeys({"ap-a": A, "ap-b": B, "ap-old": LEGACY}))


@pytest.fixture
def quota(monkeypatch):
    fake = FakeQuota()
    monkeypatch.setattr("app.core.rate_limit.quota_limiter", fake)
    return fake


def post(key, text="hello", **extra):
    body = {"model": "mock", "messages": [{"role": "user", "content": text}], **extra}
    return client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})


# ---------- Whose request it was ----------


def test_the_trace_records_the_owner_of_the_key(quota, traces):
    assert post("ap-a").status_code == 200
    assert post("ap-b").status_code == 200
    assert post("ap-old").status_code == 200
    assert [t.user_id for t in traces] == [1, 2, None]


# ---------- The daily quota ----------


def test_a_request_within_the_quota_shows_what_is_left(quota):
    response = post("ap-a")
    assert response.status_code == 200
    assert (response.headers["x-quota-limit"], response.headers["x-quota-remaining"]) == ("2", "1")
    assert quota.calls == [(1, 2)]


def test_a_request_over_the_quota_gets_429_with_the_time_to_wait(monkeypatch, traces):
    monkeypatch.setattr("app.core.rate_limit.quota_limiter", FakeQuota(allowed=False))
    response = post("ap-a")
    assert response.status_code == 429
    assert "daily request limit" in response.json()["detail"]
    assert response.headers["retry-after"] == "3600" and response.headers["x-quota-remaining"] == "0"
    assert traces == []  # refused before the handler, like the per minute limit


def test_a_key_without_an_owner_has_no_quota(quota):
    assert post("ap-old").status_code == 200
    assert quota.calls == []


# ---------- The cap on max_tokens ----------


def test_max_tokens_above_the_cap_of_the_account_is_refused(quota, traces):
    refused = post("ap-a", max_tokens=101)
    assert refused.status_code == 400 and "100" in refused.json()["detail"]
    assert traces[0].status == "error" and traces[0].http_status == 400
    assert post("ap-a", max_tokens=100).status_code == 200
    assert post("ap-old", max_tokens=100000).status_code == 200  # a key without an owner has no cap


# ---------- A prompt belongs to its owner ----------


class OwnedStore(PromptStore):
    def __init__(self, data):
        super().__init__()
        self.data = data

    async def _load(self, name, user_id=None):
        return self.data.get((user_id, name))


def version(number, text):
    return PromptVersionData(number, text, None, 1)


def test_a_user_cannot_use_the_prompt_of_another_user(monkeypatch, quota):
    monkeypatch.setattr("app.api.chat.prompt_store", OwnedStore({(1, "qa"): [version(1, "You belong to user one.")]}))
    mine = post("ap-a", autopilot={"prompt": "qa"})
    assert mine.status_code == 200 and mine.headers["x-autopilot-prompt"] == "qa:1"
    assert post("ap-b", autopilot={"prompt": "qa"}).status_code == 404  # the same name, another owner
    assert post("ap-old", autopilot={"prompt": "qa"}).status_code == 404  # a key without an owner either


def test_two_users_can_have_a_prompt_with_the_same_name(monkeypatch, quota):
    data = {(1, "qa"): [version(1, "one")], (2, "qa"): [version(5, "two")]}
    monkeypatch.setattr("app.api.chat.prompt_store", OwnedStore(data))
    assert post("ap-a", autopilot={"prompt": "qa"}).headers["x-autopilot-prompt"] == "qa:1"
    assert post("ap-b", autopilot={"prompt": "qa"}).headers["x-autopilot-prompt"] == "qa:5"


def test_the_prompt_cache_does_not_mix_owners():
    store = OwnedStore({(1, "qa"): [version(1, "one")], (2, "qa"): [version(1, "two")]})
    assert asyncio.run(store.get_versions("qa", 1))[0].system_prompt == "one"
    assert asyncio.run(store.get_versions("qa", 2))[0].system_prompt == "two"
    assert asyncio.run(store.get_versions("qa", 3)) is None
    assert asyncio.run(store.get_versions("qa", None)) is None


def test_the_prompt_cache_has_a_size_limit():
    store = OwnedStore({})
    store.MAX_REMEMBERED = 5
    for i in range(20):
        asyncio.run(store.get_versions(f"name{i}", 1))
    assert len(store._cache) <= 5