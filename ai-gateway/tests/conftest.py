import pytest

from app.core.auth import KeyStore
from app.core.rate_limit import RateLimitResult


class AllowAllLimiter:
    async def check(self, key_id: str, limit: int) -> RateLimitResult:
        return RateLimitResult(True, limit, limit, 0)


@pytest.fixture(autouse=True)
def no_real_redis(monkeypatch):
    """Purane tests ko Redis ki zarurat nahi. Rate limit ke asli tests 3.5 mein."""
    monkeypatch.setattr("app.core.rate_limit.limiter", AllowAllLimiter())


@pytest.fixture
def api_key(tmp_path, monkeypatch):
    """Har test ko ek alag, temporary key store aur us ki ek sahi key milti hai."""
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("test-key")
    monkeypatch.setattr("app.core.auth.key_store", store)
    return key