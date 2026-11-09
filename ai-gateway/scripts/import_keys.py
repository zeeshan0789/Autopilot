"""Copy the keys from the old data/api_keys.json file into Postgres. Safe to run twice.

Usage (from ai-gateway/):  python -m scripts.import_keys
"""

import asyncio
import json

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from app.core.auth import KEYS_FILE
from app.db.models import ApiKey
from app.db.session import SessionLocal, engine


async def run() -> None:
    if not KEYS_FILE.exists():
        print(f"{KEYS_FILE} not found, nothing to import")
        return

    data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    imported = 0
    try:
        async with SessionLocal() as session:
            for key_hash, entry in data.items():
                active = entry.get("active", True)
                result = await session.execute(
                    insert(ApiKey)
                    .values(
                        key_hash=key_hash,
                        key_id=key_hash[:8],
                        name=entry["name"],
                        rate_limit_per_min=entry["rate_limit_per_min"],
                        active=active,
                        revoked_at=None if active else func.now(),
                    )
                    .on_conflict_do_nothing(index_elements=["key_hash"])
                )
                imported += result.rowcount
            await session.commit()
    finally:
        await engine.dispose()

    print(f"{imported} of {len(data)} keys imported ({len(data) - imported} were already there)")


if __name__ == "__main__":
    asyncio.run(run())