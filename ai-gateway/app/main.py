import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.cache.runtime import cache_service
from app.core.rate_limit import limiter

# uvicorn's own logger, so INFO lines show up in the terminal
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check at startup that Redis is reachable
    try:
        await limiter.client.ping()
        logger.info("Redis connected: rate limiting ON")
    except Exception as e:
        logger.warning(
            "Redis not reachable (%s). Rate limiting is OFF (fail open). "
            "Run docker compose up -d and check REDIS_URL.",
            e,
        )

    # Load the embedding model now so the first request does not pay for it
    if cache_service.settings.enabled:
        try:
            await asyncio.to_thread(cache_service.warm_up)
            logger.info("Semantic cache ready (embedding model loaded)")
        except Exception as e:
            logger.warning(
                "Semantic cache warm up failed (%s). Requests will keep failing open.", e
            )
    else:
        logger.info("Semantic cache is disabled in config/cache.yaml")
    yield


app = FastAPI(title="Autopilot Gateway", lifespan=lifespan)

app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}