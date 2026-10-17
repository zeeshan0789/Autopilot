from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post(model: str, key: str, messages=None):
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": messages},
        headers={"Authorization": f"Bearer {key}"},
    )


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_mock_model_works_without_fallback(api_key):
    r = post("mock", api_key)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Mock reply: hello"
    assert r.headers["x-autopilot-provider"] == "mock"
    assert r.headers["x-autopilot-fallback"] == "false"


def test_primary_down_falls_back(api_key):
    r = post("mock-down", api_key)
    assert r.status_code == 200
    assert r.headers["x-autopilot-provider"] == "mock"
    assert r.headers["x-autopilot-fallback"] == "true"


def test_rate_limited_primary_falls_back(api_key):
    r = post("mock-429", api_key)
    assert r.status_code == 200
    assert r.headers["x-autopilot-fallback"] == "true"


def test_unknown_model_gives_404(api_key):
    r = post("model-jo-hai-hi-nahi", api_key)
    assert r.status_code == 404


def test_all_providers_failed_gives_502(api_key):
    r = post("mock-dead", api_key)
    assert r.status_code == 502


def test_empty_messages_gives_422(api_key):
    r = post("mock", api_key, messages=[])
    assert r.status_code == 422


def test_stream_not_supported_yet(api_key):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r.status_code == 400