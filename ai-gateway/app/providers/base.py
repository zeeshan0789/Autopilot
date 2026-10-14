from abc import ABC, abstractmethod

from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(Exception):
    """Provider ne koi bhi error diya."""


class ProviderRateLimitError(ProviderError):
    """Provider ne 429 diya (limit khatam)."""


class ProviderTimeoutError(ProviderError):
    """Provider ne waqt par jawab nahi diya."""


class BaseProvider(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Har provider ko yeh method dena hoga."""