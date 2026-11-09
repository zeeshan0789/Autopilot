import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.db import session as db_session
from app.db.models import ApiKey
from app.settings import BASE_DIR

logger = logging.getLogger("autopilot.auth")

# The old file based key store. The gateway now uses Postgres (DbKeyStore). The file
# is only read by scripts/import_keys.py and by tests.
KEYS_FILE = BASE_DIR / "data" / "api_keys.json"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class ApiKeyInfo:
    key_id: str  # first 8 characters of the hash, used in traces and cache namespaces
    name: str
    rate_limit_per_min: int


class AuthUnavailable(Exception):
    """The key store could not be reached, so the key cannot be checked."""


class KeyStore:
    """Keys in a local JSON file (kept for tests and for importing old keys)."""

    def __init__(self, path: Path):
        self.path = path
        self._cache: dict = {}
        self._mtime: float | None = None

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            with open(self.path, encoding="utf-8") as f:
                self._cache = json.load(f)
            self._mtime = mtime
        return self._cache

    def create(self, name: str, rate_limit_per_min: int = 60) -> str:
        """Create a new key. The plain key is only returned here, never stored."""
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
        self._mtime = None
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

    async def authenticate(self, key: str) -> ApiKeyInfo | None:
        return self.verify(key)


class DbKeyStore:
    """API keys in Postgres.

    Answers are remembered for ttl_seconds, so a revoked key can still work for up to
    that long. If the database is down, a key that was seen recently keeps working and
    an unknown key gets AuthUnavailable (the gateway answers 503, it never lets it in).
    """

    def __init__(
        self,
        ttl_seconds: float = 10.0,
        max_cached: int = 1000,
        clock=time.monotonic,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_cached = max_cached
        self._clock = clock
        self._cache: dict[str, tuple[float, ApiKeyInfo | None]] = {}

    async def authenticate(self, key: str) -> ApiKeyInfo | None:
        key_hash = hash_key(key)
        now = self._clock()
        cached = self._cache.get(key_hash)
        if cached and now - cached[0] < self.ttl_seconds:
            return cached[1]

        try:
            info = await self._load(key_hash)
        except Exception as e:
            if cached:
                logger.warning("Key store unavailable (%s), using the last known answer", e)
                return cached[1]
            raise AuthUnavailable(str(e)) from e

        if len(self._cache) >= self.max_cached:
            self._cache.clear()  # bounded memory, even under a flood of random keys
        self._cache[key_hash] = (now, info)
        return info

    async def _load(self, key_hash: str) -> ApiKeyInfo | None:
        async with db_session.SessionLocal() as session:
            row = (
                await session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
            ).scalar_one_or_none()
        if row is None or not row.active:
            return None
        return ApiKeyInfo(
            key_id=row.key_id,
            name=row.name,
            rate_limit_per_min=row.rate_limit_per_min,
        )


key_store = DbKeyStore()

bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> ApiKeyInfo:
    """Guard: every protected endpoint runs this first."""
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="API key missing. Send: Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        info = await key_store.authenticate(credentials.credentials)
    except AuthUnavailable:
        raise HTTPException(status_code=503, detail="authentication service unavailable")

    if info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return info