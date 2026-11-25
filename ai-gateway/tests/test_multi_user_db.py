
import asyncio
import uuid

import pytest
from sqlalchemy import delete, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.auth import DbKeyStore, hash_key
from app.core.prompts import PromptStore
from app.db import session as db_session
from app.db.models import ApiKey, Prompt, PromptVersion, User
from app.settings import DATABASE_URL

pytestmark = pytest.mark.integration


def test_ownership_rules_with_a_real_database(monkeypatch):
    async def scenario():
        engine = create_async_engine(DATABASE_URL, connect_args={"timeout": 3})
        try:
            async with engine.connect() as connection:
                await connection.execute(text("select 1"))
        except Exception as e:
            await engine.dispose()
            pytest.skip(f"Postgres not reachable ({type(e).__name__}); run docker compose up -d")

        factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(db_session, "SessionLocal", factory)
        tag = uuid.uuid4().hex[:8]
        emails = [f"alice-{tag}@example.com", f"bob-{tag}@example.com"]
        key_text = f"ap-test-{tag}"
        try:
            async with factory() as session:
                alice = User(email=emails[0], password_hash="x", role="member")
                bob = User(email=emails[1], password_hash="x", role="member")
                session.add_all([alice, bob])
                await session.flush()
                session.add(ApiKey(
                    key_hash=hash_key(key_text),
                    key_id=hash_key(key_text)[:8],
                    name="k",
                    user_id=alice.id,
                    rate_limit_per_min=30
                ))
                for owner in (alice, bob):
                    prompt = Prompt(user_id=owner.id, name="qa")
                    session.add(prompt)
                    await session.flush()
                    session.add(PromptVersion(prompt_id=prompt.id, version=1, system_prompt=f"owned by {owner.id}", weight=1))
                await session.commit()
                alice_id, bob_id = alice.id, bob.id

            # The key knows its owner and the limits of the owner
            info = await DbKeyStore()._load(hash_key(key_text))
            assert info.user_id == alice_id
            assert info.rate_limit_per_min == 30  # the limit of the key itself, not the default
            assert (info.daily_request_limit, info.max_tokens_cap) == (200, 1024)

            # Each user only sees their own prompt, although both are called "qa"
            store = PromptStore()
            assert (await store._load("qa", alice_id))[0].system_prompt == f"owned by {alice_id}"
            assert (await store._load("qa", bob_id))[0].system_prompt == f"owned by {bob_id}"
            assert await store._load("qa", None) is None

            # A name is unique per owner
            async with factory() as session:
                session.add(Prompt(user_id=alice_id, name="qa"))
                with pytest.raises(IntegrityError):
                    await session.commit()

            # A suspended user's key stops working, and so does a revoked key
            async with factory() as session:
                await session.execute(update(User).where(User.id == alice_id).values(status="suspended"))
                await session.commit()
            assert await DbKeyStore()._load(hash_key(key_text)) is None
            async with factory() as session:
                await session.execute(update(User).where(User.id == alice_id).values(status="active"))
                await session.commit()
            assert (await DbKeyStore()._load(hash_key(key_text))) is not None
            async with factory() as session:
                await session.execute(update(ApiKey).where(ApiKey.key_hash == hash_key(key_text)).values(active=False))
                await session.commit()
            assert await DbKeyStore()._load(hash_key(key_text)) is None
        finally:
            async with factory() as session:
                await session.execute(delete(User).where(User.email.in_(emails)))  # keys and prompts go with them
                await session.commit()
            await engine.dispose()

    asyncio.run(scenario())
