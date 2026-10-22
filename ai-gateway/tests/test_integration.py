import asyncio
import time
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.rate_limit import RedisRateLimiter
from app.core.tracing import TraceData, record_trace
from app.db.models import Trace
from app.settings import DATABASE_URL, REDIS_URL


def test_default_urls_use_ipv4_loopback():
    """'localhost' can resolve to IPv6 first on Windows and cost a 1s timeout."""
    assert "localhost" not in REDIS_URL
    assert "localhost" not in DATABASE_URL


@pytest.mark.integration
def test_trace_roundtrip_with_real_postgres(monkeypatch):
    async def run():
        # Fresh engine inside this event loop (the app's global engine is bound to another one)
        engine = create_async_engine(DATABASE_URL, connect_args={"timeout": 3})
        Session = async_sessionmaker(engine, expire_on_commit=False)

        try:
            async with engine.connect() as conn:
                await conn.execute(text("select 1"))
        except Exception as e:
            await engine.dispose()
            pytest.skip(f"Postgres not reachable ({type(e).__name__}); run docker compose up -d")

        monkeypatch.setattr("app.core.tracing.SessionLocal", Session)
        request_id = "it-" + uuid.uuid4().hex

        try:
            await record_trace(
                TraceData(
                    request_id=request_id,
                    requested_model="integration-test",
                    served_model="mock",
                    provider="mock",
                    prompt_tokens=1,
                    completion_tokens=3,
                    total_tokens=4,
                    estimated_cost_usd=Decimal("0.0000086"),
                    latency_ms=5,
                )
            )

            async with Session() as session:
                row = (
                    await session.execute(
                        select(Trace).where(Trace.request_id == request_id)
                    )
                ).scalar_one()
                assert row.status == "success"
                assert row.estimated_cost_usd == Decimal("0.0000086")

                # The ORM hides the difference between JSON null and SQL NULL, so ask SQL directly
                is_sql_null = (
                    await session.execute(
                        text(
                            "select failed_attempts is null from traces "
                            "where request_id = :r"
                        ),
                        {"r": request_id},
                    )
                ).scalar()
                assert is_sql_null is True
        finally:
            async with Session() as session:
                await session.execute(
                    text("delete from traces where request_id = :r"),
                    {"r": request_id},
                )
                await session.commit()
            await engine.dispose()

    asyncio.run(run())


@pytest.mark.integration
def test_rate_limiter_with_real_redis():
    async def run():
        limiter = RedisRateLimiter(REDIS_URL)

        try:
            await limiter.client.ping()
        except Exception as e:
            await limiter.client.aclose()
            pytest.skip(f"Redis not reachable ({type(e).__name__}); run docker compose up -d")

        # Fixed window: avoid running across a minute boundary
        seconds_into_minute = time.time() % 60
        if seconds_into_minute > 55:
            await asyncio.sleep(61 - seconds_into_minute)

        key_id = "it-" + uuid.uuid4().hex[:8]
        try:
            results = [await limiter.check(key_id, 3) for _ in range(4)]
            assert [r.allowed for r in results] == [True, True, True, False]
            assert [r.remaining for r in results] == [2, 1, 0, 0]
        finally:
            keys = [k async for k in limiter.client.scan_iter(f"rl:{key_id}:*")]
            if keys:
                await limiter.client.delete(*keys)
            await limiter.client.aclose()

    asyncio.run(run())