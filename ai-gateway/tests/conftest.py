import pytest

from app.core.auth import KeyStore
from app.core.rate_limit import RateLimitResult


class AllowAllLimiter:
    async def check(self, key_id: str, limit: int) -> RateLimitResult:
        return RateLimitResult(True, limit, limit, 0)


@pytest.fixture(autouse=True)
def no_real_redis(monkeypatch):
    """Most tests do not need Redis. Real rate limit tests live in test_rate_limit.py."""
    monkeypatch.setattr("app.core.rate_limit.limiter", AllowAllLimiter())


@pytest.fixture(autouse=True)
def traces(monkeypatch):
    """Collect traces in a list instead of writing them to Postgres."""
    collected = []
    monkeypatch.setattr("app.api.chat.schedule_trace", collected.append)
    return collected


@pytest.fixture
def api_key(tmp_path, monkeypatch):
    """Give each test its own temporary key store and one valid key."""
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("test-key")
    monkeypatch.setattr("app.core.auth.key_store", store)
    return key