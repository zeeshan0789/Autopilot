"""The daily request quota of a user: a Redis counter per user and UTC day.

Every request counts, cache hits included. If Redis is unreachable the request is allowed (the same
choice as for the per minute rate limit) and a warning is logged."""

import logging
import time
from dataclasses import dataclass

from redis import asyncio as aioredis
from redis.exceptions import RedisError

from app.settings import REDIS_URL

logger = logging.getLogger("autopilot.quota")

COUNTER_TTL_SECONDS = 2 * 24 * 3600


@dataclass
class QuotaResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds until the counter restarts (UTC midnight)


class DailyQuota:
    def __init__(self, url: str, clock=time.time):
        self.client = aioredis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        self._clock = clock

    async def consume(self, user_id: int, limit: int) -> QuotaResult:
        return await self._consume(f"quota:{user_id}", limit)

    async def consume_platform(self, limit: int) -> QuotaResult:
        """One counter that all members share, per UTC day."""
        return await self._consume("quota:platform", limit)

    async def _consume(self, prefix: str, limit: int) -> QuotaResult:
        now = self._clock()
        key = f"{prefix}:{time.strftime('%Y%m%d', time.gmtime(now))}"
        retry_after = 86400 - int(now % 86400)
        try:
            async with self.client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, COUNTER_TTL_SECONDS)
                count, _ = await pipe.execute()
        except (RedisError, OSError) as e:
            logger.warning("daily quota: Redis unavailable (%s), request allowed", e)
            return QuotaResult(True, limit, limit, 0)
        return QuotaResult(count <= limit, limit, max(0, limit - count), retry_after)


quota_limiter = DailyQuota(REDIS_URL)