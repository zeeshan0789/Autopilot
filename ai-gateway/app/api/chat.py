from fastapi import APIRouter, HTTPException, Response

from app.core.executor import AllProvidersFailedError, execute_with_fallback
from app.providers.base import ProviderRateLimitError
from app.providers.registry import registry
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, response: Response):
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true abhi supported nahi")

    chain = registry.chain_for(request.model)
    if not chain:
        raise HTTPException(
            status_code=404,
            detail=f"model '{request.model}' kisi provider mein configured nahi",
        )

    try:
        result = await execute_with_fallback(chain, request)
    except AllProvidersFailedError as e:
        detail = "; ".join(f"{a.provider}({a.model}): {a.error}" for a in e.attempts)
        if all(isinstance(err, ProviderRateLimitError) for err in e.errors):
            raise HTTPException(status_code=429, detail=detail)
        raise HTTPException(status_code=502, detail=detail)

    response.headers["X-Autopilot-Provider"] = result.provider
    response.headers["X-Autopilot-Model"] = result.model
    response.headers["X-Autopilot-Fallback"] = str(result.fallback_used).lower()
    return result.response