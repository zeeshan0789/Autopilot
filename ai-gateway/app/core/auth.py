import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.settings import BASE_DIR

KEYS_FILE = BASE_DIR / "data" / "api_keys.json"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class ApiKeyInfo:
    key_id: str  # hash ke pehle 8 harf, logs/traces mein pehchan ke liye
    name: str
    rate_limit_per_min: int


class KeyStore:
    def __init__(self, path: Path):
        self.path = path
        self._cache: dict = {}
        self._mtime: float | None = None

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        mtime = self.path.stat().st_mtime
        # File badli ho (nayi key bani) to dobara padho, warna cache
        if mtime != self._mtime:
            with open(self.path, encoding="utf-8") as f:
                self._cache = json.load(f)
            self._mtime = mtime
        return self._cache

    def create(self, name: str, rate_limit_per_min: int = 60) -> str:
        """Nayi key banata hai. Asli key sirf yahin wapas milti hai, store nahi hoti."""
        key = "ap-" + secrets.token_urlsafe(32)
        data = dict(self._load())
        data[hash_key(key)] = {
            "name": name,
            "rate_limit_per_min": rate_limit_per_min,
            "active": True,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._mtime = None  # agli baar zaroor dobara padho
        return key

    def verify(self, key: str) -> ApiKeyInfo | None:
        key_hash = hash_key(key)
        entry = self._load().get(key_hash)
        if not entry or not entry.get("active", True):
            return None
        return ApiKeyInfo(
            key_id=key_hash[:8],
            name=entry["name"],
            rate_limit_per_min=entry["rate_limit_per_min"],
        )


key_store = KeyStore(KEYS_FILE)

bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> ApiKeyInfo:
    """Guard: har protected endpoint pe yeh chalta hai."""
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="API key missing. Header bhejo: Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    info = key_store.verify(credentials.credentials)
    if info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return info