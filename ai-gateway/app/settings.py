from pathlib import Path

import yaml
from dotenv import load_dotenv

# ai-gateway/ ka raasta (settings.py se do folder upar)
BASE_DIR = Path(__file__).resolve().parent.parent

# .env se keys environment mein load ho jati hain
load_dotenv(BASE_DIR / ".env")


def load_providers_config(path: Path | None = None) -> list[dict]:
    path = path or BASE_DIR / "config" / "providers.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["providers"]