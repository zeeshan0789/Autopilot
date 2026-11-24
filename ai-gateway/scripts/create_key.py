"""Create a gateway API key. Only its hash is stored in Postgres.

Usage (from ai-gateway/):
    python -m scripts.create_key <name> [rate_limit_per_min] [--user someone@example.com]
Without --user the key has no owner (no daily quota). With --user it belongs to that account.
"""

import argparse
import asyncio
import secrets

from sqlalchemy import select

from app.core.auth import hash_key
from app.db.models import ApiKey, User
from app.db.session import SessionLocal, engine


async def create(name: str, limit: int, user_email: str | None) -> str:
    key = "ap-" + secrets.token_urlsafe(32)
    key_hash = hash_key(key)
    try:
        async with SessionLocal() as session:
            owner_id = None
            if user_email:
                owner = (
                    await session.execute(select(User).where(User.email == user_email.strip().lower()))
                ).scalar_one_or_none()
                if owner is None:
                    raise SystemExit(f"no user with the email {user_email}")
                owner_id = owner.id
            session.add(
                ApiKey(key_hash=key_hash, key_id=key_hash[:8], name=name, rate_limit_per_min=limit, user_id=owner_id)
            )
            await session.commit()
    finally:
        await engine.dispose()
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("limit", nargs="?", type=int, default=60)
    parser.add_argument("--user", help="email of the account that owns the key")
    args = parser.parse_args()

    key = asyncio.run(create(args.name, args.limit, args.user))
    owner = f", owner {args.user}" if args.user else ", no owner"
    print()
    print(f"Key '{args.name}' created (limit: {args.limit} requests/minute{owner})")
    print()
    print(f"    {key}")
    print()
    print("Copy it now. It is not shown again (only its hash is stored).")


if __name__ == "__main__":
    main()