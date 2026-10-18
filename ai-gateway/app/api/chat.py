import time
import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Response

from app.core.auth import ApiKeyInfo
from app.core.cost import estimate_cost
from app.core.executor import AllProvidersFailedError, execute_with_fallback
from app.core.rate_limit import enforce_rate_limit
from app.core.tracing import TraceData, schedule_trace
from app.providers.base import ProviderRateLimitError
from app.providers.registry import registry
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    api_key: ApiKeyInfo = Depends(enforce_rate_limit),
):
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    trace = TraceData(
        request_id=request_id,
        requested_model=request.model,
        api_key_id=api_key.key_id,
    )

    try:
        if request.stream:
            raise HTTPException(
                status_code=400, detail="stream=true is not supported yet"
            )

        chain = registry.chain_for(request.model)
        if not chain:
            raise HTTPException(
                status_code=404,
                detail=f"model '{request.model}' is not configured in any provider",
            )

        try:
            result = await execute_with_fallback(chain, request)
        except AllProvidersFailedError as e:
            trace.failed_attempts = [asdict(a) for a in e.attempts]
            detail = "; ".join(
                f"{a.provider}({a.model}): {a.error}" for a in e.attempts
            )
            if all(isinstance(err, ProviderRateLimitError) for err in e.errors):
                raise HTTPException(status_code=429, detail=detail)
            raise HTTPException(status_code=502, detail=detail)

    except HTTPException as exc:
        # Every error path ends up here, so one place records it
        trace.status = "error"
        trace.http_status = exc.status_code
        trace.error = str(exc.detail)[:1000]
        trace.latency_ms = elapsed_ms(started)
        schedule_trace(trace)
        exc.headers = {**(exc.headers or {}), "X-Request-ID": request_id}
        raise

    usage = result.response.usage
    trace.served_model = result.model
    trace.provider = result.provider
    trace.fallback_used = result.fallback_used
    trace.failed_attempts = [asdict(a) for a in result.failed_attempts] or None
    trace.prompt_tokens = usage.prompt_tokens
    trace.completion_tokens = usage.completion_tokens
    trace.total_tokens = usage.total_tokens
    trace.estimated_cost_usd = estimate_cost(
        result.model, usage.prompt_tokens, usage.completion_tokens
    )
    trace.latency_ms = elapsed_ms(started)
    schedule_trace(trace)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Autopilot-Provider"] = result.provider
    response.headers["X-Autopilot-Model"] = result.model
    response.headers["X-Autopilot-Fallback"] = str(result.fallback_used).lower()
    return result.response