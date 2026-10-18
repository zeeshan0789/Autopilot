import logging
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Response
from redis import asyncio as aioredis
from redis.exceptions import RedisError

from app.core.auth import ApiKeyInfo, require_api_key
from app.settings import REDIS_URL

logger = logging.getLogger("autopilot.rate_limit")


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds, agli window shuru hone tak


class RedisRateLimiter:
    def __init__(self, url: str, window_seconds: int = 60):
        # Timeout chhota: Redis band ho to request na atke
        self.client = aioredis.from_url(
            url, socket_connect_timeout=1, socket_timeout=1
        )
        self.window = window_seconds

    async def check(self, key_id: str, limit: int) -> RateLimitResult:
        now = int(time.time())
        window_start = now - (now % self.window)
        redis_key = f"rl:{key_id}:{window_start}"

        try:
            async with self.client.pipeline(transaction=True) as pipe:
                pipe.incr(redis_key)
                pipe.expire(redis_key, self.window * 2)  # purana counter khud saaf
                count, _ = await pipe.execute()
        except (RedisError, OSError) as e:
            # fail-open: Redis na chale to bhi gateway chalta rahe
            logger.warning("rate limiter: Redis nahi chala (%s), request guzar di", e)
            return RateLimitResult(True, limit, limit, 0)

        return RateLimitResult(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            retry_after=window_start + self.window - now,
        )


limiter = RedisRateLimiter(REDIS_URL)


async def enforce_rate_limit(
    response: Response,
    api_key: ApiKeyInfo = Depends(require_api_key),
) -> ApiKeyInfo:
    """Pehle auth (401), phir rate limit (429)."""
    result = await limiter.check(api_key.key_id, api_key.rate_limit_per_min)

    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
    }
    if not result.allowed:
        headers["Retry-After"] = str(result.retry_after)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({result.limit} requests/minute). "
            f"{result.retry_after} second baad dobara try karo.",
            headers=headers,
        )

    for name, value in headers.items():
        response.headers[name] = value
    return api_key