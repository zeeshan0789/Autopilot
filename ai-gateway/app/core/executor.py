import asyncio
import logging
import random
from dataclasses import dataclass, field

from app.providers.base import BaseProvider, ProviderError, ProviderTransientError
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger("autopilot.executor")


@dataclass
class Attempt:
    provider: str
    model: str
    error: str


@dataclass
class ExecutionResult:
    response: ChatCompletionResponse
    provider: str
    model: str
    fallback_used: bool
    failed_attempts: list[Attempt] = field(default_factory=list)


class AllProvidersFailedError(Exception):
    def __init__(self, attempts: list[Attempt], errors: list[ProviderError]):
        self.attempts = attempts
        self.errors = errors
        super().__init__("sab providers fail ho gaye")


async def call_with_retry(
    provider: BaseProvider,
    request: ChatCompletionRequest,
    max_retries: int = 2,
    base_delay: float = 0.5,
) -> ChatCompletionResponse:
    """Waqti errors pe backoff ke saath dobara try karta hai.

    Total koshishein = 1 (pehli) + max_retries.
    Doosre errors (429, 401, 404...) foran upar chale jate hain.
    """
    attempt = 0
    while True:
        try:
            return await provider.chat(request)
        except ProviderTransientError as e:
            if attempt >= max_retries:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 0.1)
            logger.warning(
                "provider=%s attempt=%d failed (%s), %.2fs baad dobara",
                provider.name,
                attempt + 1,
                e,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1


async def execute_with_fallback(
    chain: list[tuple[BaseProvider, str]],
    request: ChatCompletionRequest,
    max_retries: int = 2,
) -> ExecutionResult:
    """Chain ke har (provider, model) ko order mein try karta hai."""
    failures: list[Attempt] = []
    errors: list[ProviderError] = []

    for index, (provider, model) in enumerate(chain):
        # Har provider ko apna model naam chahiye, isliye request ki copy
        req = request.model_copy(update={"model": model})
        try:
            response = await call_with_retry(provider, req, max_retries=max_retries)
        except ProviderError as e:
            failures.append(Attempt(provider.name, model, str(e)))
            errors.append(e)
            logger.warning(
                "provider=%s model=%s fail (%s), agle provider pe ja rahe hain",
                provider.name,
                model,
                e,
            )
            continue

        return ExecutionResult(
            response=response,
            provider=provider.name,
            model=model,
            fallback_used=index > 0,
            failed_attempts=failures,
        )

    raise AllProvidersFailedError(failures, errors)