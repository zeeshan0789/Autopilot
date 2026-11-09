"""Create a gateway API key. Only its hash is stored in Postgres.

Usage (from ai-gateway/):  python -m scripts.create_key <name> [rate_limit_per_min]
"""

import asyncio
import secrets
import sys

from app.core.auth import hash_key
from app.db.models import ApiKey
from app.db.session import SessionLocal, engine


async def create(name: str, limit: int) -> str:
    key = "ap-" + secrets.token_urlsafe(32)
    key_hash = hash_key(key)
    try:
        async with SessionLocal() as session:
            session.add(
                ApiKey(
                    key_hash=key_hash,
                    key_id=key_hash[:8],
                    name=name,
                    rate_limit_per_min=limit,
                )
            )
            await session.commit()
    finally:
        await engine.dispose()
    return key


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.create_key <name> [rate_limit_per_min]")
        sys.exit(1)

    name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    key = asyncio.run(create(name, limit))

    print()
    print(f"Key '{name}' created (limit: {limit} requests/minute)")
    print()
    print(f"    {key}")
    print()
    print("Copy it now. It is not shown again (only its hash is stored).")


if __name__ == "__main__":
    main()