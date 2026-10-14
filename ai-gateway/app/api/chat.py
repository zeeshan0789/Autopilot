from fastapi import APIRouter, HTTPException

from app.providers.base import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.mock import MockProvider
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()

# Abhi sirf mock provider. Step 1.5 mein Ollama, Step 2 mein config file se.
provider = MockProvider()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true abhi supported nahi")

    try:
        return await provider.chat(request)
    except ProviderRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ProviderTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))