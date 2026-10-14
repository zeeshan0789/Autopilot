from fastapi import APIRouter, HTTPException

from app.providers.base import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.registry import registry
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true abhi supported nahi")

    providers = registry.for_model(request.model)
    if not providers:
        raise HTTPException(
            status_code=404,
            detail=f"model '{request.model}' kisi provider mein configured nahi",
        )

    provider = providers[0]  # Step 2.4 mein yahan fallback aayega

    try:
        return await provider.chat(request)
    except ProviderRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ProviderTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))