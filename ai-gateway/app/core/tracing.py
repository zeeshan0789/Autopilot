import asyncio
import logging
from dataclasses import asdict, dataclass
from decimal import Decimal

from app.db.models import Trace
from app.db.session import SessionLocal

logger = logging.getLogger("autopilot.tracing")

# Keep references to running tasks so they are not garbage collected mid-flight
_pending: set[asyncio.Task] = set()


@dataclass
class TraceData:
    request_id: str
    requested_model: str
    api_key_id: str | None = None
    served_model: str | None = None
    provider: str | None = None
    fallback_used: bool = False
    failed_attempts: list[dict] | None = None
    status: str = "success"  # success or error
    http_status: int = 200
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    latency_ms: int | None = None

    # Semantic cache (None when the request never reached the cache step)
    cache_status: str | None = None  # hit, miss, blocked, bypass
    cache_reason: str | None = None
    cache_similarity: float | None = None
    cache_entry_id: str | None = None
    saved_tokens: int | None = None
    saved_cost_usd: Decimal | None = None


async def record_trace(data: TraceData) -> None:
    """Write one trace row. Never raises: tracing must not break requests."""
    try:
        async with SessionLocal() as session:
            session.add(Trace(**asdict(data)))
            await session.commit()
    except Exception as e:
        logger.warning("Failed to write trace %s: %s", data.request_id, e)


def schedule_trace(data: TraceData) -> None:
    """Fire and forget: the response does not wait for the database."""
    task = asyncio.create_task(record_trace(data))
    _pending.add(task)
    task.add_done_callback(_pending.discard)