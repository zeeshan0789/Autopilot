import asyncio
from decimal import Decimal

from fastapi.testclient import TestClient

from app.core.auth import hash_key
from app.core.tracing import TraceData, record_trace
from app.main import app

client = TestClient(app)


def post(key: str, model: str, **extra):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }
    return client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
    )


def test_success_is_traced(api_key, traces):
    r = post(api_key, "mock")
    assert r.status_code == 200
    assert len(traces) == 1
    t = traces[0]
    assert t.status == "success"
    assert t.http_status == 200
    assert t.requested_model == "mock"
    assert t.served_model == "mock"
    assert t.provider == "mock"
    assert t.fallback_used is False
    assert t.failed_attempts is None
    assert t.total_tokens == t.prompt_tokens + t.completion_tokens
    assert t.estimated_cost_usd == Decimal("0")
    assert t.api_key_id == hash_key(api_key)[:8]
    assert isinstance(t.latency_ms, int) and t.latency_ms >= 0


def test_fallback_is_traced(api_key, traces):
    r = post(api_key, "mock-down")
    assert r.status_code == 200
    t = traces[0]
    assert t.requested_model == "mock-down"
    assert t.served_model == "mock"
    assert t.fallback_used is True
    assert len(t.failed_attempts) == 1
    assert t.failed_attempts[0]["provider"] == "mock-down"


def test_total_failure_is_traced(api_key, traces):
    r = post(api_key, "mock-dead")
    assert r.status_code == 502
    t = traces[0]
    assert t.status == "error"
    assert t.http_status == 502
    assert t.provider is None
    assert t.total_tokens is None
    assert t.failed_attempts
    assert t.error


def test_unknown_model_is_traced(api_key, traces):
    r = post(api_key, "no-such-model")
    assert r.status_code == 404
    t = traces[0]
    assert t.status == "error"
    assert t.http_status == 404
    assert t.provider is None


def test_unsupported_stream_is_traced(api_key, traces):
    r = post(api_key, "mock", stream=True)
    assert r.status_code == 400
    assert traces[0].http_status == 400


def test_unauthenticated_request_is_not_traced(api_key, traces):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401
    assert traces == []


def test_request_id_header_matches_trace(api_key, traces):
    ok = post(api_key, "mock")
    bad = post(api_key, "no-such-model")
    assert ok.headers["x-request-id"] == traces[0].request_id
    assert bad.headers["x-request-id"] == traces[1].request_id
    assert traces[0].request_id != traces[1].request_id


def test_record_trace_swallows_database_errors(monkeypatch):
    class BrokenSession:
        async def __aenter__(self):
            raise OSError("database is down")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr("app.core.tracing.SessionLocal", lambda: BrokenSession())
    # Must not raise
    asyncio.run(record_trace(TraceData(request_id="r1", requested_model="m")))

def test_missing_failed_attempts_is_stored_as_sql_null():
    """Python None must become SQL NULL, not JSON null."""
    from app.db.models import Trace

    assert Trace.__table__.c.failed_attempts.type.none_as_null is True