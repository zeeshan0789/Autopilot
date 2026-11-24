
import asyncio
import logging
import time
from urllib import request
import uuid
from dataclasses import asdict
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Response

from app.cache.runtime import cache_service
from app.cache.service import CacheLookup
from app.core.auth import ApiKeyInfo
from app.core.cost import estimate_cost
from app.core.executor import AllProvidersFailedError, execute_with_fallback
from app.core.policy import RouteDecision, policy_router
from app.core.prompts import PromptError, prompt_store
from app.mlrouter.runtime import MlDecision, ml_runtime
from app.core.rate_limit import enforce_rate_limit
from app.core.tracing import TraceData, schedule_trace
from app.providers.base import ProviderRateLimitError
from app.providers.registry import registry
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger("autopilot.chat")

router = APIRouter()


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

def add_route_headers(response: Response, decision: RouteDecision) -> None:
    if decision.routed:
        response.headers["X-Autopilot-Route"] = decision.rule
        response.headers["X-Autopilot-Routed-Model"] = decision.model

def add_prompt_header(response: Response, trace: TraceData) -> None:
    if trace.prompt_name is not None:
        response.headers["X-Autopilot-Prompt"] = f"{trace.prompt_name}:{trace.prompt_version}"


async def decide_ml(request: ChatCompletionRequest) -> MlDecision:
    """Embedding and prediction are CPU work, so they run in a thread. The router must never break a request."""
    try:
        messages = [m.model_dump() for m in request.messages]
        return await asyncio.to_thread(ml_runtime.decide, messages)
    except Exception as e:
        logger.warning("Learned router crashed, using the rule policy: %s", e)
        return MlDecision(None, "error")


def ml_route(request: ChatCompletionRequest, ml: MlDecision) -> RouteDecision:
    if ml.model is not None and registry.for_model(ml.model):
        return RouteDecision(ml.model, "ml_router", None, True)
    # The router declined (an unfamiliar prompt) or failed: the rule policy decides, like for "auto"
    fallback = policy_router.decide(request.model_copy(update={"model": policy_router.settings.alias}))
    if fallback.routed:
        model = fallback.model
    else:
        model = policy_router.settings.tiers[policy_router.settings.default_tier]
    return RouteDecision(model, f"ml_{ml.status}>{fallback.rule}", fallback.tier, True)

async def lookup_cache(request: ChatCompletionRequest, namespace: str) -> CacheLookup:
    """Embedding is CPU work, so it runs in a thread. The cache must never break a request."""
    try:
        return await asyncio.to_thread(cache_service.lookup, request, namespace)
    except Exception as e:
        logger.warning("Cache lookup crashed, skipping the cache: %s", e)
        return CacheLookup("bypass", "cache_error")


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
        user_id=api_key.user_id,
    )


    try:
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported yet"
            )

        if api_key.max_tokens_cap and request.max_tokens and request.max_tokens > api_key.max_tokens_cap:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens is above the limit of this account ({api_key.max_tokens_cap})",
            )


        # A request can ask for a managed prompt (versioned, with an A/B traffic split)
        cache_namespace = api_key.key_id
        if request.autopilot and request.autopilot.prompt:
            try:
                selected = await prompt_store.resolve(request, api_key.user_id)
            except PromptError as e:
                raise HTTPException(status_code=e.status_code, detail=str(e))
            trace.prompt_name = selected.name
            trace.prompt_version = selected.version
            request = request.model_copy(update={"messages": selected.messages})
            # Answers from different prompt versions must never be mixed by the cache
            cache_namespace = f"{api_key.key_id}|{selected.name}:{selected.version}"
        # Model "smart" is routed by the learned router, model "auto" by the rule policy (and scored
        # silently by the learned router in shadow mode). Any other model is used as requested.
        ml = None
        ml_settings = ml_runtime.settings
        if ml_settings.enabled and (
            request.model == ml_settings.alias
            or (ml_settings.shadow and request.model == policy_router.settings.alias)
        ):
            ml = await decide_ml(request)
            trace.ml_pick, trace.ml_status = ml.model, ml.status
            trace.ml_prob, trace.ml_similarity = ml.probability, ml.similarity

        if ml is not None and request.model == ml_settings.alias:
            decision = ml_route(request, ml)
        else:
            decision = policy_router.decide(request)
        trace.routed_model = decision.model
        trace.policy_rule = decision.rule
        if decision.routed:
            # From here on the request carries the concrete model, so the fallback
            # chain and the cache scope both use the model that will really answer
            request = request.model_copy(update={"model": decision.model})

        chain = registry.chain_for(request.model)
        if not chain:
            raise HTTPException(
                status_code=404,
                detail=f"model '{request.model}' is not configured in any provider",
            )

        # Each API key has its own cache, so one caller never sees another's answers
        lookup = await lookup_cache(request, cache_namespace)
        trace.cache_status = lookup.status
        trace.cache_reason = lookup.reason
        trace.cache_similarity = lookup.similarity

        if lookup.status == "hit":
            entry = lookup.entry
            # tokens and cost describe what this request actually used (nothing);
            # saved_* describe what the original model call used and this hit avoided
            trace.served_model = entry.served_model
            trace.provider = "cache"
            trace.prompt_tokens = 0
            trace.completion_tokens = 0
            trace.total_tokens = 0
            trace.estimated_cost_usd = Decimal(0)
            trace.cache_entry_id = entry.entry_id
            trace.saved_tokens = entry.prompt_tokens + entry.completion_tokens
            trace.saved_cost_usd = entry.estimated_cost_usd
            trace.latency_ms = elapsed_ms(started)
            schedule_trace(trace)

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Autopilot-Cache"] = "hit"
            response.headers["X-Autopilot-Provider"] = "cache"
            response.headers["X-Autopilot-Model"] = entry.served_model
            response.headers["X-Autopilot-Fallback"] = "false"
            add_route_headers(response, decision)
            add_prompt_header(response, trace)
            return cache_service.response_from(entry)

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
    cost = estimate_cost(result.model, usage.prompt_tokens, usage.completion_tokens)
    trace.served_model = result.model
    trace.provider = result.provider
    trace.fallback_used = result.fallback_used
    trace.failed_attempts = [asdict(a) for a in result.failed_attempts] or None
    trace.prompt_tokens = usage.prompt_tokens
    trace.completion_tokens = usage.completion_tokens
    trace.total_tokens = usage.total_tokens
    trace.estimated_cost_usd = cost

    # Remember the fresh answer. store() refuses fallback answers, empty answers and
    # requests the cache did not look at. A failure here must not fail the request.
    try:
        await asyncio.to_thread(
            cache_service.store,
            lookup,
            result.response,
            provider=result.provider,
            estimated_cost_usd=cost,
            fallback_used=result.fallback_used,
        )
    except Exception as e:
        logger.warning("Could not store the answer in the cache: %s", e)

    trace.latency_ms = elapsed_ms(started)
    schedule_trace(trace)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Autopilot-Cache"] = lookup.status
    response.headers["X-Autopilot-Provider"] = result.provider
    response.headers["X-Autopilot-Model"] = result.model
    response.headers["X-Autopilot-Fallback"] = str(result.fallback_used).lower()
    add_route_headers(response, decision)
    add_prompt_header(response, trace)
    return result.response
