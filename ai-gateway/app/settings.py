from pathlib import Path

import yaml
from dotenv import load_dotenv

# ai-gateway/ ka raasta (settings.py se do folder upar)
BASE_DIR = Path(__file__).resolve().parent.parent

# .env se keys environment mein load ho jati hain
load_dotenv(BASE_DIR / ".env")

DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "providers.yaml"


def _load_yaml(path: Path | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_providers_config(path: Path | None = None) -> list[dict]:
    return _load_yaml(path)["providers"]


def load_fallbacks_config(path: Path | None = None) -> dict[str, list[str]]:
    return _load_yaml(path).get("fallbacks") or {}

import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://autopilot:autopilot_dev@127.0.0.1:5432/autopilot",
)