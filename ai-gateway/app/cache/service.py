"""Cache service: decides whether a request may use the cache and what the result is.

It ties together the embedder, the semantic cache and the safety guard. It does not
call any model and knows nothing about HTTP. The gateway (step P2.4c) uses it like this:

    lookup = service.lookup(request, namespace)
    if lookup.status == "hit":
        return service.response_from(lookup.entry)
    ... call the model ...
    service.store(lookup, response, provider=..., estimated_cost_usd=..., fallback_used=...)

Embedding is CPU work and blocks, so an async caller should run lookup() and store()
in a thread (asyncio.to_thread).
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Protocol

import numpy as np

from app.cache.guard import decide_hit, is_time_sensitive
from app.cache.semantic_cache import CacheEntry, CacheScope, SemanticCache
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse, Message

logger = logging.getLogger("autopilot.cache")

# How many close entries to examine per lookup. The closest one can be blocked by the
# guard (the 2024 answer when asking about 2025) while the next one is fine.
CANDIDATES_TO_CHECK = 5

# A short answer that starts like this is a refusal or a "cannot help" reply. Models are
# not deterministic, so the same question may get a real answer next time. Caching the
# refusal would keep serving it until the entry expires.
_REFUSAL_OPENERS = (
    "i'm sorry", "i am sorry", "sorry", "i apologize", "apologies",
    "i can't", "i cannot", "i can not", "i won't", "i will not",
    "i'm unable", "i am unable", "i'm not able", "i am not able",
    "i'm afraid", "unfortunately, i", "as an ai",
    "i don't have", "i do not have", "i don't know", "i do not know",
    "i'm not sure", "i am not sure",
)
# A long answer that merely starts with "Sorry" is real content, so only short ones count
_MAX_REFUSAL_CHARS = 300


def looks_like_refusal(answer: str) -> bool:
    text = answer.strip()
    if len(text) > _MAX_REFUSAL_CHARS:
        return False
    # Curly apostrophes and leading markdown or quotes must not hide the opener
    text = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    text = text.lstrip(" \t\r\n*_\"'`>#")
    return text.startswith(_REFUSAL_OPENERS)


@dataclass(frozen=True)
class CacheSettings:
    enabled: bool = True
    # Starting value only. It must be tuned on a labeled sample (step P2.6).
    similarity_threshold: float = 0.90
    ttl_seconds: float = 3600
    max_entries_per_scope: int = 1000
    # The embedding model truncates long input, so two long prompts that differ only
    # after the cut-off would look identical. Long prompts skip the cache.
    max_prompt_chars: int = 800
    # Higher temperature means the caller wants varied answers, so skip the cache.
    max_temperature: float = 0.7

    def __post_init__(self):
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.max_entries_per_scope < 1:
            raise ValueError("max_entries_per_scope must be at least 1")
        if self.max_prompt_chars < 1:
            raise ValueError("max_prompt_chars must be at least 1")
        if self.max_temperature < 0:
            raise ValueError("max_temperature must not be negative")


class EmbedderLike(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, text: str) -> np.ndarray: ...


@dataclass
class CacheLookup:
    # hit: reuse entry. miss: nothing close enough. blocked: something close was found
    # but the guard rejected it. bypass: the cache was not consulted at all.
    status: str
    reason: str
    similarity: float | None = None
    entry: CacheEntry | None = None
    # Kept so store() can reuse them without embedding the prompt again
    scope: CacheScope | None = None
    embedding: np.ndarray | None = None
    prompt_text: str | None = None

    @property
    def can_store(self) -> bool:
        return self.status in ("miss", "blocked") and self.embedding is not None


# ---------- Building the cache key from a request ----------


def conversation_text(messages: list[Message]) -> str:
    """The text that is embedded and guarded: everything except system messages."""
    turns = [m for m in messages if m.role != "system"]
    if len(turns) == 1 and turns[0].role == "user":
        return turns[0].content.strip()
    return "\n".join(f"{m.role}: {m.content.strip()}" for m in turns)


def system_hash(messages: list[Message]) -> str:
    system = "\n".join(m.content.strip() for m in messages if m.role == "system")
    if not system:
        return "none"
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


def temperature_bucket(temperature: float | None) -> str:
    if temperature is None:
        return "default"  # the provider's own default
    if temperature <= 0.3:
        return "low"
    if temperature <= 0.7:
        return "mid"
    return "high"


# ---------- The service ----------


class CacheService:
    def __init__(
        self,
        settings: CacheSettings,
        embedder: EmbedderLike,
        cache: SemanticCache | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.settings = settings
        self.embedder = embedder
        self._clock = clock
        self._cache = cache
        self._lock = threading.Lock()  # FAISS and numpy arrays are not thread safe

    def warm_up(self) -> None:
        """Load the embedding model now instead of during the first request."""
        self.embedder.encode("warm up")
        with self._lock:
            self._get_cache()

    def size(self) -> int:
        with self._lock:
            return self._get_cache().size()

    def lookup(self, request: ChatCompletionRequest, namespace: str) -> CacheLookup:
        reason = self._bypass_reason(request)
        if reason:
            return CacheLookup("bypass", reason)

        text = conversation_text(request.messages)
        scope = CacheScope(
            namespace=namespace,
            model=request.model,
            system_hash=system_hash(request.messages),
            temperature_bucket=temperature_bucket(request.temperature),
        )

        try:
            embedding = self.embedder.encode(text)
        except Exception as e:
            # The cache must never break a request: fail open
            logger.warning("Embedding failed, skipping the cache: %s", e)
            return CacheLookup("bypass", "embedding_error")

        try:
            with self._lock:
                candidates = self._get_cache().lookup_many(
                    scope, embedding, CANDIDATES_TO_CHECK
                )
        except Exception as e:
            logger.warning("Cache lookup failed, skipping the cache: %s", e)
            return CacheLookup("bypass", "cache_error")

        keep = dict(scope=scope, embedding=embedding, prompt_text=text)
        if not candidates:
            return CacheLookup("miss", "no_candidate", **keep)

        first_blocked = None
        for candidate in candidates:  # best first
            decision = decide_hit(
                candidate.similarity,
                text,
                candidate.entry.prompt_text,
                self.settings.similarity_threshold,
            )
            if decision.hit:
                return CacheLookup(
                    "hit", "hit", candidate.similarity, candidate.entry, **keep
                )
            if decision.reason == "below_threshold":
                break  # sorted by similarity, every remaining one is lower
            if first_blocked is None:
                first_blocked = decision

        if first_blocked is not None:
            return CacheLookup(
                "blocked", first_blocked.reason, first_blocked.similarity, **keep
            )
        return CacheLookup("miss", "below_threshold", candidates[0].similarity, **keep)

    def store(
        self,
        lookup: CacheLookup,
        response: ChatCompletionResponse,
        *,
        provider: str,
        estimated_cost_usd: Decimal | None,
        fallback_used: bool,
    ) -> bool:
        """Remember a fresh model answer. Returns True when it was stored."""
        if not lookup.can_store:
            return False
        if fallback_used:
            # A fallback model's answer may be worse than the requested model's.
            # Caching it would keep serving it after the primary recovers.
            return False

        answer = response.choices[0].message.content if response.choices else ""
        if not answer.strip():
            return False  # never cache an empty answer
        if looks_like_refusal(answer):
            return False  # never cache "I'm sorry, but I can't help with that"

        entry = CacheEntry(
            prompt_text=lookup.prompt_text,
            response=response.model_dump(),
            served_model=response.model,
            provider=provider,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
            created_at=self._clock(),
            ttl_seconds=self.settings.ttl_seconds,
        )
        try:
            with self._lock:
                self._get_cache().add(lookup.scope, lookup.embedding, entry)
        except Exception as e:
            logger.warning("Could not store the answer in the cache: %s", e)
            return False
        return True

    def response_from(self, entry: CacheEntry) -> ChatCompletionResponse:
        """Rebuild a response from a cached entry, with a fresh id and timestamp."""
        data = {
            key: value
            for key, value in entry.response.items()
            if key not in ("id", "created")
        }
        return ChatCompletionResponse(**data)

    # ---------- internals ----------

    def _get_cache(self) -> SemanticCache:
        if self._cache is None:
            self._cache = SemanticCache(
                dimension=self.embedder.dimension,
                max_entries_per_scope=self.settings.max_entries_per_scope,
                clock=self._clock,
            )
        return self._cache

    def _bypass_reason(self, request: ChatCompletionRequest) -> str | None:
        if not self.settings.enabled:
            return "disabled"
        if request.autopilot is not None and not request.autopilot.cache:
            return "opt_out"
        if request.stream:
            return "stream"
        if request.max_tokens is not None:
            # The cache key does not include max_tokens, so a truncated answer could
            # be served to a caller who asked for a longer one, and the other way round.
            return "max_tokens_set"
        if (
            request.temperature is not None
            and request.temperature > self.settings.max_temperature
        ):
            return "high_temperature"

        text = conversation_text(request.messages)
        if not text:
            return "empty_prompt"
        if len(text) > self.settings.max_prompt_chars:
            return "prompt_too_long"
        if is_time_sensitive(text):
            return "time_sensitive"
        return None