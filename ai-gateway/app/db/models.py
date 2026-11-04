import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Float, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Trace(Base):
    """Har gateway request ki ek line."""

    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Kaun aur kya maanga
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    api_key_id: Mapped[str | None] = mapped_column(String(16), index=True)
    requested_model: Mapped[str] = mapped_column(String(128))

    # Kisne jawab diya
    served_model: Mapped[str | None] = mapped_column(String(128))
    provider: Mapped[str | None] = mapped_column(String(64))
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    failed_attempts: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))

    # Nateeja
    status: Mapped[str] = mapped_column(String(16), index=True)  # success / error
    http_status: Mapped[int] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)

    # Naap
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 8))
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    # Semantic cache (NULL on rows written before the cache existed)
    # cache_status: hit, miss, blocked (guard rejected a close match), bypass (not looked up)
    cache_status: Mapped[str | None] = mapped_column(String(16), index=True)
    cache_reason: Mapped[str | None] = mapped_column(String(32))
    cache_similarity: Mapped[float | None] = mapped_column(Float)
    cache_entry_id: Mapped[str | None] = mapped_column(String(32))
    # On a hit: what the original model call used, i.e. what the cache avoided
    saved_tokens: Mapped[int | None] = mapped_column(Integer)
    saved_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 8))
    # Routing policy
    routed_model: Mapped[str | None] = mapped_column(String(128))
    policy_rule: Mapped[str | None] = mapped_column(String(64))