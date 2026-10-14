import asyncio

from app.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)


def count_words(text: str) -> int:
    # Asli tokenizer nahi, bas ek mota andaza
    return len(text.split())


class MockProvider(BaseProvider):
    def __init__(self, name: str = "mock", mode: str = "ok", delay: float = 0.0):
        super().__init__(name)
        self.mode = mode    # ok, error, rate_limit, timeout
        self.delay = delay  # slow response simulate karne ke liye (seconds)

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self.delay:
            await asyncio.sleep(self.delay)

        if self.mode == "error":
            raise ProviderError(f"{self.name}: simulated error")
        if self.mode == "rate_limit":
            raise ProviderRateLimitError(f"{self.name}: simulated 429")
        if self.mode == "timeout":
            raise ProviderTimeoutError(f"{self.name}: simulated timeout")

        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        reply = f"Mock reply: {last_user}"

        prompt_tokens = sum(count_words(m.content) for m in request.messages)
        completion_tokens = count_words(reply)

        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content=reply))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )