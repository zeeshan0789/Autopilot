import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.rate_limit import limiter

# uvicorn ka apna logger, taake INFO bhi terminal mein dikhe
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server start hote hi check: Redis tak pohonch ho rahi hai?
    try:
        await limiter.client.ping()
        logger.info("Redis connected: rate limiting ON")
    except Exception as e:
        logger.warning(
            "Redis nahi mila (%s). Rate limiting BAND hai (fail-open). "
            "docker compose up -d chalao aur REDIS_URL check karo.",
            e,
        )
    yield


app = FastAPI(title="Autopilot Gateway", lifespan=lifespan)

app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}