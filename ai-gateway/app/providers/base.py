from abc import ABC, abstractmethod

from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(Exception):
    """Provider ne koi bhi error diya (retry ka faida nahi)."""


class ProviderTransientError(ProviderError):
    """Waqti masla (connect fail, 5xx). Retry karne se theek ho sakta hai."""


class ProviderTimeoutError(ProviderTransientError):
    """Provider ne waqt par jawab nahi diya."""


class ProviderRateLimitError(ProviderError):
    """Provider ne 429 diya. Retry nahi, fallback."""


class BaseProvider(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Har provider ko yeh method dena hoga."""