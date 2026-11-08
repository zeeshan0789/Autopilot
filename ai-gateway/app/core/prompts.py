"""Managed prompts: versions, A/B traffic split and template rendering.

A request can name a prompt in its "autopilot" field. The gateway picks one version of
that prompt (a weighted random choice, or a stable choice per "user" so a person always
sees the same variant), fills in the template and puts the result in front of the messages.
"""

import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass

from sqlalchemy import select

from app.db import session as db_session
from app.db.models import Prompt, PromptVersion
from app.schemas.openai import ChatCompletionRequest, Message

logger = logging.getLogger("autopilot.prompts")

CACHE_TTL_SECONDS = 10


class PromptError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PromptVersionData:
    version: int
    system_prompt: str
    user_template: str | None
    weight: int


@dataclass(frozen=True)
class SelectedPrompt:
    name: str
    version: int
    messages: list


# ---------- Choosing a version (the A/B split) ----------


def choose_version(
    name: str,
    versions: list[PromptVersionData],
    forced: int | None = None,
    sticky_key: str | None = None,
    rng=random,
) -> PromptVersionData:
    """Pick the version to serve.

    forced: use exactly this version (even with weight 0).
    sticky_key: the same key always gets the same version while the weights stay the same.
    Otherwise a weighted random choice.
    """
    if forced is not None:
        for v in versions:
            if v.version == forced:
                return v
        raise PromptError(404, f"prompt '{name}' has no version {forced}")

    candidates = sorted((v for v in versions if v.weight > 0), key=lambda v: v.version)
    if not candidates:
        raise PromptError(409, f"prompt '{name}' has no active version (all weights are 0)")

    total = sum(v.weight for v in candidates)
    if sticky_key:
        digest = hashlib.sha256(f"{name}:{sticky_key}".encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % total
    else:
        bucket = rng.randrange(total)

    cumulative = 0
    for v in candidates:
        cumulative += v.weight
        if bucket < cumulative:
            return v
    return candidates[-1]


# ---------- Rendering ----------

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def fill(template: str, variables: dict[str, str]) -> str:
    """Replace {{name}} placeholders in a single pass (values are not expanded again)."""

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise PromptError(400, f"missing variable '{key}'")
        return variables[key]

    return _PLACEHOLDER.sub(replace, template)


def render_messages(
    version: PromptVersionData, messages: list[Message], variables: dict[str, str]
) -> list[Message]:
    """The prompt's system message goes first. A user template wraps the last user message
    ({{input}} is that message). The caller's own messages follow."""
    rendered = list(messages)
    if version.user_template:
        index = max((i for i, m in enumerate(rendered) if m.role == "user"), default=None)
        if index is None:
            raise PromptError(400, "this prompt needs a user message")
        content = fill(version.user_template, {**variables, "input": rendered[index].content})
        rendered[index] = Message(role="user", content=content)

    system = Message(role="system", content=fill(version.system_prompt, variables))
    return [system] + rendered


# ---------- The store ----------


class PromptStore:
    """Reads prompts from Postgres and remembers them for a few seconds."""

    def __init__(self, ttl_seconds: float = CACHE_TTL_SECONDS, clock=time.monotonic):
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._cache: dict[str, tuple[float, list[PromptVersionData] | None]] = {}

    async def get_versions(self, name: str) -> list[PromptVersionData] | None:
        """The versions of a prompt, or None when the prompt does not exist."""
        now = self._clock()
        cached = self._cache.get(name)
        if cached and now - cached[0] < self.ttl_seconds:
            return cached[1]
        try:
            versions = await self._load(name)
        except Exception as e:
            if cached:
                logger.warning("Prompt store unavailable (%s), using the last known '%s'", e, name)
                return cached[1]
            logger.warning("Prompt store unavailable (%s)", e)
            raise PromptError(503, "the prompt store is unavailable") from e
        self._cache[name] = (now, versions)
        return versions

    async def _load(self, name: str) -> list[PromptVersionData] | None:
        async with db_session.SessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        Prompt.id,
                        PromptVersion.version,
                        PromptVersion.system_prompt,
                        PromptVersion.user_template,
                        PromptVersion.weight,
                    )
                    .select_from(Prompt)
                    .join(PromptVersion, PromptVersion.prompt_id == Prompt.id, isouter=True)
                    .where(Prompt.name == name)
                    .order_by(PromptVersion.version)
                )
            ).all()
        if not rows:
            return None
        return [
            PromptVersionData(r.version, r.system_prompt, r.user_template, r.weight)
            for r in rows
            if r.version is not None
        ]

    async def resolve(self, request: ChatCompletionRequest) -> SelectedPrompt:
        options = request.autopilot
        name = options.prompt
        versions = await self.get_versions(name)
        if versions is None:
            raise PromptError(404, f"prompt '{name}' not found")
        chosen = choose_version(name, versions, options.version, request.user)
        messages = render_messages(chosen, request.messages, options.variables)
        return SelectedPrompt(name=name, version=chosen.version, messages=messages)


prompt_store = PromptStore()