from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post(model: str, messages=None):
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return client.post(
        "/v1/chat/completions", json={"model": model, "messages": messages}
    )


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_mock_model_works_without_fallback():
    r = post("mock")
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Mock reply: hello"
    assert r.headers["x-autopilot-provider"] == "mock"
    assert r.headers["x-autopilot-fallback"] == "false"


def test_primary_down_falls_back():
    r = post("mock-down")
    assert r.status_code == 200
    assert r.headers["x-autopilot-provider"] == "mock"
    assert r.headers["x-autopilot-fallback"] == "true"


def test_rate_limited_primary_falls_back():
    r = post("mock-429")
    assert r.status_code == 200
    assert r.headers["x-autopilot-fallback"] == "true"


def test_unknown_model_gives_404():
    r = post("model-jo-hai-hi-nahi")
    assert r.status_code == 404


def test_all_providers_failed_gives_502():
    r = post("mock-dead")
    assert r.status_code == 502


def test_empty_messages_gives_422():
    r = post("mock", messages=[])
    assert r.status_code == 422


def test_stream_not_supported_yet():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 400