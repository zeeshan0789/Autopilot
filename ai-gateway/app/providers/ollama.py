import httpx

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


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        name: str = "ollama",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        super().__init__(name)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
        }

        options = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"{self.name}: timeout") from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"{self.name}: Ollama se connect nahi hua ({type(e).__name__})"
            ) from e

        if resp.status_code == 429:
            raise ProviderRateLimitError(f"{self.name}: 429")
        if resp.status_code != 200:
            raise ProviderError(f"{self.name}: {resp.status_code} {resp.text[:200]}")

        data = resp.json()
        content = data["message"]["content"]
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content=content))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )