import pytest

from app.core.auth import KeyStore


@pytest.fixture
def api_key(tmp_path, monkeypatch):
    """Har test ko ek alag, temporary key store aur us ki ek sahi key milti hai."""
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("test-key")
    monkeypatch.setattr("app.core.auth.key_store", store)
    return key