import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text,
    UniqueConstraint, func, text,
)
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
    # Managed prompt (NULL when the request did not use one)
    prompt_name: Mapped[str | None] = mapped_column(String(64), index=True)
    prompt_version: Mapped[int | None] = mapped_column(Integer)
    # Learned router (NULL when it was not consulted)
    ml_pick: Mapped[str | None] = mapped_column(String(128))
    ml_status: Mapped[str | None] = mapped_column(String(16), index=True)
    ml_prob: Mapped[float | None] = mapped_column(Float)
    ml_similarity: Mapped[float | None] = mapped_column(Float)
    # Whose request it was (NULL for requests made with a key that has no owner)
    user_id: Mapped[int | None] = mapped_column(Integer, index=True)

class Prompt(Base):
    """A named prompt that clients can ask for. Its versions are in prompt_versions. A prompt belongs
    to the user who created it, and its name is unique per owner."""

    __tablename__ = "prompts"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="prompts_user_id_name_key"),
        # Prompts without an owner (created before accounts existed) still need unique names
        Index("uq_prompts_legacy_name", "name", unique=True, postgresql_where=text("user_id IS NULL")),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PromptVersion(Base):
    """One version of a prompt. The text of a version in use should never be edited."""

    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("prompt_id", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_id: Mapped[int] = mapped_column(
        ForeignKey("prompts.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    system_prompt: Mapped[str] = mapped_column(Text)
    user_template: Mapped[str | None] = mapped_column(Text)
    # Share of the traffic. 0 means the version is not served (it can still be forced)
    weight: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

class ApiKey(Base):
    """A gateway API key. Only the SHA 256 hash of the key is stored."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # First 8 characters of the hash: the short id used in traces and cache namespaces
    key_id: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(100))
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, server_default=text("60"))
    active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The user who owns the key. NULL for keys that were created before accounts existed.
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)


class User(Base):
    """A person with an account. role: admin (runs the platform) or member (a normal user with an own workspace)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(16), server_default=text("'member'"))
    status: Mapped[str] = mapped_column(String(16), server_default=text("'active'"))  # active or suspended
    daily_request_limit: Mapped[int] = mapped_column(Integer, server_default=text("200"))
    max_tokens_cap: Mapped[int] = mapped_column(Integer, server_default=text("1024"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

class Invite(Base):
    """A single use code that lets one person sign up while the signup mode is 'invite'.
    Only the hash of the code is stored."""

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True)
    note: Mapped[str | None] = mapped_column(String(200))
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    used_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))