from fastapi.testclient import TestClient

from app.main import app


class StubRedis:
    async def ping(self):
        return True


class StubLimiter:
    client = StubRedis()


class StubSettings:
    def __init__(self, enabled):
        self.enabled = enabled


class StubCacheService:
    def __init__(self, enabled=True, fail=False):
        self.settings = StubSettings(enabled)
        self.fail = fail
        self.warm_ups = 0

    def warm_up(self):
        self.warm_ups += 1
        if self.fail:
            raise RuntimeError("model download failed")


def start_app(monkeypatch, service):
    monkeypatch.setattr("app.main.cache_service", service)
    monkeypatch.setattr("app.main.limiter", StubLimiter())
    with TestClient(app) as client:  # runs startup and shutdown
        return client.get("/health").status_code


def test_embedding_model_is_loaded_at_startup(monkeypatch):
    service = StubCacheService(enabled=True)
    assert start_app(monkeypatch, service) == 200
    assert service.warm_ups == 1


def test_disabled_cache_does_not_load_the_model(monkeypatch):
    service = StubCacheService(enabled=False)
    assert start_app(monkeypatch, service) == 200
    assert service.warm_ups == 0


def test_failed_warm_up_does_not_stop_the_server(monkeypatch):
    service = StubCacheService(enabled=True, fail=True)
    assert start_app(monkeypatch, service) == 200
    assert service.warm_ups == 1