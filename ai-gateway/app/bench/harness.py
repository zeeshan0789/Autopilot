"""Runs a workload against the provider directly and against the Autopilot gateway.

The gateway runs in this process through an ASGI transport, with the real providers. Traces are
kept in memory, the rate limiter and the key store are replaced by trivial ones. The primary
provider can be made to fail during a window of requests, the same way for every mode.
"""

import asyncio
import tempfile
import time
import types
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import httpx

import app.api.chat as chat_module
import app.core.auth as auth_module
import app.core.rate_limit as rate_limit_module
from app.bench.results import ItemResult
from app.bench.workload import WorkItem
from app.cache.service import CacheService, CacheSettings
from app.core.auth import KeyStore
from app.core.cost import estimate_cost
from app.core.rate_limit import RateLimitResult
from app.evals.checks import run_checks
from app.main import app as gateway_app
from app.providers.base import ProviderError
from app.providers.registry import registry
from app.schemas.openai import ChatCompletionRequest
from app.settings import load_cache_config

MODES = ("direct", "autopilot", "autopilot_no_guard", "autopilot_strict")


@dataclass
class BenchConfig:
    model: str = "openai/gpt-oss-20b"
    delay_seconds: float = 2.0  # pause after every real provider call (free tier limits)
    outage_from: float = 0.3  # the outage covers this share of the workload...
    outage_to: float = 0.6  # ...up to this one

    def outage_window(self, total: int) -> tuple:
        return int(total * self.outage_from), int(total * self.outage_to)


class _AllowAllLimiter:
    async def check(self, key_id, limit):
        return RateLimitResult(True, limit, limit, 0)


class _DisabledMl:
    settings = types.SimpleNamespace(enabled=False, alias="smart", shadow=False)

    def decide(self, messages):
        raise AssertionError("the learned router is not part of this benchmark")


class Outage:
    """Makes the given providers fail while `active` is True."""

    def __init__(self):
        self.active = False
        self._patched = []

    def install(self, providers):
        for provider in providers:
            original = provider.chat

            async def chat(request, _original=original):
                if self.active:
                    raise ProviderError("injected outage")
                return await _original(request)

            provider.chat = chat
            self._patched.append(provider)

    def remove(self):
        for provider in self._patched:
            if "chat" in vars(provider):
                del provider.chat
        self._patched = []


def _passed(item: WorkItem, answer: str) -> bool:
    return all(r.passed for r in run_checks(item.checks, answer))


async def _direct_request(provider, item: WorkItem, model: str, in_outage: bool) -> ItemResult:
    request = ChatCompletionRequest(model=model, messages=item.messages, temperature=0.0)
    started = time.perf_counter()
    try:
        response = await provider.chat(request)
    except Exception as e:
        latency = (time.perf_counter() - started) * 1000
        return ItemResult(item.id, item.kind, item.base_id, in_outage, "error", False, latency, error=str(e)[:200])
    latency = (time.perf_counter() - started) * 1000
    answer = response.choices[0].message.content or ""
    cost = estimate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return ItemResult(
        item.id, item.kind, item.base_id, in_outage, "ok", _passed(item, answer), latency,
        cost_usd=float(cost or 0), served_model=model, answer=answer,
    )


async def _gateway_request(client, item: WorkItem, model: str, in_outage: bool, traces: list) -> ItemResult:
    before = len(traces)
    started = time.perf_counter()
    response = await client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": item.messages, "temperature": 0.0},
    )
    latency = (time.perf_counter() - started) * 1000
    trace = traces[before] if len(traces) > before else None

    if response.status_code != 200:
        try:
            detail = str(response.json().get("detail", ""))[:200]
        except ValueError:
            detail = response.text[:200]
        return ItemResult(
            item.id, item.kind, item.base_id, in_outage, "error", False, latency,
            cache_status=trace.cache_status if trace else None, error=f"http {response.status_code}: {detail}",
        )

    answer = response.json()["choices"][0]["message"]["content"] or ""
    return ItemResult(
        item.id, item.kind, item.base_id, in_outage, "ok", _passed(item, answer), latency,
        cost_usd=float(trace.estimated_cost_usd or 0) if trace else 0.0,
        cache_status=response.headers.get("x-autopilot-cache"),
        fallback_used=response.headers.get("x-autopilot-fallback") == "true",
        served_model=response.headers.get("x-autopilot-model"),
        answer=answer,
    )


async def run_mode(mode: str, workload: list[WorkItem], config: BenchConfig, embedder, progress=None, sleep=asyncio.sleep) -> list[ItemResult]:
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'")
    start, end = config.outage_window(len(workload))
    providers = registry.for_model(config.model)
    if not providers:
        raise ValueError(f"no provider serves {config.model}")

    outage = Outage()
    outage.install(providers)
    results: list[ItemResult] = []
    try:
        with ExitStack() as stack:
            traces: list = []
            client = None
            if mode != "direct":
                settings = CacheSettings(
                    **{
                        **load_cache_config(),
                        "guard_enabled": mode != "autopilot_no_guard",
                        "strict_words": mode == "autopilot_strict",
                    }
                )
                store = KeyStore(Path(tempfile.mkdtemp()) / "keys.json")
                key = store.create("benchmark", 1_000_000)
                stack.enter_context(patch.object(chat_module, "cache_service", CacheService(settings, embedder)))
                stack.enter_context(patch.object(chat_module, "schedule_trace", traces.append))
                stack.enter_context(patch.object(chat_module, "ml_runtime", _DisabledMl()))
                stack.enter_context(patch.object(rate_limit_module, "limiter", _AllowAllLimiter()))
                stack.enter_context(patch.object(auth_module, "key_store", store))
                client = httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=gateway_app),
                    base_url="http://benchmark",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=180.0,
                )

            try:
                for index, item in enumerate(workload):
                    in_outage = start <= index < end
                    outage.active = in_outage
                    if mode == "direct":
                        result = await _direct_request(providers[0], item, config.model, in_outage)
                        called_provider = True
                    else:
                        result = await _gateway_request(client, item, config.model, in_outage, traces)
                        called_provider = result.cache_status != "hit"
                    results.append(result)
                    if progress:
                        progress(index + 1, len(workload), item, result)
                    if called_provider and index < len(workload) - 1 and config.delay_seconds > 0:
                        await sleep(config.delay_seconds)
            finally:
                if client is not None:
                    await client.aclose()
    finally:
        outage.remove()
    return results