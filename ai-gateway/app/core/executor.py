import asyncio
import logging
import random

from app.providers.base import BaseProvider, ProviderTransientError
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger("autopilot.executor")


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
            # 0.5s, 1s, 2s... aur thora sa random jitter
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