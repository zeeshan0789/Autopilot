from app.providers.registry import build_provider

CONFIG = {"name": "ollama-local", "type": "ollama", "base_url": "http://127.0.0.1:11434", "models": ["qwen2.5:0.5b"]}


def test_the_ollama_address_can_be_overridden_by_the_environment(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, name, base_url, timeout=120):
            seen["base_url"] = base_url

    monkeypatch.setattr("app.providers.registry.OllamaProvider", Recorder)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    build_provider(CONFIG)
    assert seen["base_url"] == "http://127.0.0.1:11434"

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    build_provider(CONFIG)
    assert seen["base_url"] == "http://host.docker.internal:11434"