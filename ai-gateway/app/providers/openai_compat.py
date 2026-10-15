import os

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


class OpenAICompatProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        timeout: float = 30.0,
    ):
        super().__init__(name)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(f"{self.name}: {self.api_key_env} .env mein nahi mili")

        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"{self.name}: timeout") from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"{self.name}: connect nahi hua ({type(e).__name__})"
            ) from e

        if resp.status_code == 429:
            raise ProviderRateLimitError(f"{self.name}: 429 limit/quota khatam")
        if resp.status_code != 200:
            raise ProviderError(f"{self.name}: {resp.status_code} {resp.text[:200]}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content=content))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )