from fastapi import APIRouter, HTTPException

from app.providers.base import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()

# Abhi seedha code mein. Step 2 mein yeh config file se aayega.
mock_provider = MockProvider()
ollama_provider = OllamaProvider(base_url="http://localhost:11434")


def pick_provider(model: str):
    if model == "mock":
        return mock_provider
    return ollama_provider


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true abhi supported nahi")

    provider = pick_provider(request.model)

    try:
        return await provider.chat(request)
    except ProviderRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ProviderTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))