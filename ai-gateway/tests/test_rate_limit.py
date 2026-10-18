import asyncio

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.auth import KeyStore
from app.core.rate_limit import RedisRateLimiter
from app.main import app

client = TestClient(app)

BODY = {"model": "mock", "messages": [{"role": "user", "content": "hello"}]}

# 1000020 ek minute ki bilkul shuruaat hai (1000020 % 60 == 0)
WINDOW_START = 1000020


# ---------- Nakli Redis aur nakli ghari ----------


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def incr(self, key):
        self.ops.append(("incr", key))

    def expire(self, key, seconds):
        self.ops.append(("expire", key, seconds))

    async def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "incr":
                self.redis.store[op[1]] = self.redis.store.get(op[1], 0) + 1
                results.append(self.redis.store[op[1]])
            else:
                results.append(True)
        return results


class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self, transaction=True):
        return FakePipeline(self)


class BrokenRedis:
    """Redis band hone ka natak."""

    def pipeline(self, transaction=True):
        raise RedisConnectionError("Redis band hai")


class Clock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now


def make_limiter(monkeypatch, now=WINDOW_START + 10, redis=None):
    clock = Clock(now)
    monkeypatch.setattr("app.core.rate_limit.time", clock)
    limiter = RedisRateLimiter("redis://127.0.0.1:6379/0")
    limiter.client = redis if redis is not None else FakeRedis()
    return limiter, clock


def check(limiter, key_id, limit):
    return asyncio.run(limiter.check(key_id, limit))


# ---------- Limiter ka asli logic ----------


def test_allows_up_to_limit_then_blocks(monkeypatch):
    limiter, _ = make_limiter(monkeypatch)
    results = [check(limiter, "keyA", 3) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]


def test_keys_are_independent(monkeypatch):
    limiter, _ = make_limiter(monkeypatch)
    for _ in range(3):
        check(limiter, "keyA", 3)
    assert check(limiter, "keyA", 3).allowed is False
    assert check(limiter, "keyB", 3).allowed is True  # doosri key par asar nahi


def test_counter_resets_in_next_minute(monkeypatch):
    limiter, clock = make_limiter(monkeypatch)
    for _ in range(4):
        check(limiter, "keyA", 3)
    assert check(limiter, "keyA", 3).allowed is False

    clock.now = WINDOW_START + 61  # agla minute shuru
    assert check(limiter, "keyA", 3).allowed is True


def test_retry_after_counts_down_to_next_window(monkeypatch):
    limiter, clock = make_limiter(monkeypatch, now=WINDOW_START + 10)
    assert check(limiter, "keyA", 3).retry_after == 50
    clock.now = WINDOW_START + 55
    assert check(limiter, "keyA", 3).retry_after == 5


def test_fail_open_when_redis_is_down(monkeypatch):
    limiter, _ = make_limiter(monkeypatch, redis=BrokenRedis())
    result = check(limiter, "keyA", 3)
    assert result.allowed is True  # Redis girne se gateway band nahi hona chahiye


# ---------- HTTP level: 429, headers, aur auth pehle ----------


def make_key(tmp_path, monkeypatch, limit):
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("limited", rate_limit_per_min=limit)
    monkeypatch.setattr("app.core.auth.key_store", store)
    return key


def test_http_429_after_limit_with_headers(tmp_path, monkeypatch):
    key = make_key(tmp_path, monkeypatch, limit=3)
    limiter, _ = make_limiter(monkeypatch)
    monkeypatch.setattr("app.core.rate_limit.limiter", limiter)
    headers = {"Authorization": f"Bearer {key}"}

    responses = [
        client.post("/v1/chat/completions", json=BODY, headers=headers)
        for _ in range(4)
    ]

    assert [r.status_code for r in responses] == [200, 200, 200, 429]
    assert responses[0].headers["x-ratelimit-limit"] == "3"
    assert responses[0].headers["x-ratelimit-remaining"] == "2"
    assert responses[2].headers["x-ratelimit-remaining"] == "0"
    assert responses[3].headers["retry-after"] == "50"


def test_invalid_key_gets_401_and_uses_no_quota(tmp_path, monkeypatch):
    make_key(tmp_path, monkeypatch, limit=3)
    fake = FakeRedis()
    limiter, _ = make_limiter(monkeypatch, redis=fake)
    monkeypatch.setattr("app.core.rate_limit.limiter", limiter)

    r = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={"Authorization": "Bearer ap-galat-key"},
    )
    assert r.status_code == 401
    assert fake.store == {}  # auth pehle fail hui, to Redis tak baat gayi hi nahi