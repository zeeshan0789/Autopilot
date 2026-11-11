"""Runs eval cases through the gateway and records the answers.

The cache is switched off for these requests (autopilot.cache = false), otherwise the
run would measure the cache and not the model. An answer from a fallback model is not an
answer from the model under test, so it is retried and, if it keeps happening, marked
invalid instead of being scored.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.evals.cases import EvalCase


class EvalAbort(Exception):
    """The run cannot continue (for example the gateway rejects the API key)."""


@dataclass
class RunConfig:
    name: str = "run"
    base_url: str = "http://127.0.0.1:8000"
    model: str = "openai/gpt-oss-20b"
    prompt: str | None = None  # a managed prompt to evaluate (see phase 3)
    version: int | None = None
    variables: dict = field(default_factory=dict)
    temperature: float = 0.0
    delay_seconds: float = 2.0  # pause between cases, free tiers have rate limits
    retries: int = 3  # extra attempts per case
    allow_fallback: bool = False
    backoff_base: float = 2.0


@dataclass
class CaseResult:
    id: str
    status: str  # ok, invalid (no usable answer after the retries), error (the gateway refused the request)
    answer: str | None = None
    http_status: int | None = None
    served_model: str | None = None
    fallback_used: bool = False
    latency_ms: int | None = None
    total_tokens: int | None = None
    attempts: int = 0
    error: str | None = None


@dataclass
class EvalRun:
    name: str
    created_at: str
    config: dict
    results: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "config": self.config,
            "results": [asdict(r) for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalRun":
        return cls(
            name=data["name"],
            created_at=data["created_at"],
            config=data["config"],
            results=[CaseResult(**r) for r in data["results"]],
        )


def save_run(run: EvalRun, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_run(path) -> EvalRun:
    return EvalRun.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_payload(case: EvalCase, config: RunConfig) -> dict:
    autopilot: dict = {"cache": False}
    if config.prompt:
        autopilot.update(prompt=config.prompt, version=config.version, variables=config.variables)
    return {
        "model": config.model,
        "messages": case.messages,
        "temperature": config.temperature,
        "autopilot": autopilot,
    }


def _backoff(config: RunConfig, attempt: int) -> float:
    return min(60.0, config.backoff_base * (2 ** (attempt - 1)))


def _retry_after(response: httpx.Response, default: float) -> float:
    try:
        return min(120.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return default


def _error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))[:300]
    except ValueError:
        return response.text[:300]


def _parse_success(response: httpx.Response) -> dict:
    body = response.json()
    return {
        "answer": body["choices"][0]["message"]["content"] or "",
        "http_status": 200,
        "served_model": response.headers.get("x-autopilot-model"),
        "fallback_used": response.headers.get("x-autopilot-fallback") == "true",
        "total_tokens": (body.get("usage") or {}).get("total_tokens"),
    }


async def run_case(client, case: EvalCase, config: RunConfig, sleep=asyncio.sleep) -> CaseResult:
    payload = build_payload(case, config)
    attempts = config.retries + 1
    reason = "no attempt was made"

    for attempt in range(1, attempts + 1):
        wait = _backoff(config, attempt)
        started = time.perf_counter()
        try:
            response = await client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as e:
            reason = f"network error ({type(e).__name__})"
        else:
            latency = int((time.perf_counter() - started) * 1000)
            status = response.status_code
            if status == 200:
                try:
                    parsed = _parse_success(response)
                except (ValueError, KeyError, IndexError, TypeError):
                    parsed = None
                    reason = "the gateway returned an unexpected response"
                if parsed is not None:
                    if not parsed["fallback_used"] or config.allow_fallback:
                        return CaseResult(id=case.id, status="ok", attempts=attempt, latency_ms=latency, **parsed)
                    reason = "answered by a fallback model, not the requested one"
            elif status in (401, 403):
                raise EvalAbort(f"the gateway rejected the API key (http {status})")
            elif status == 429 or status >= 500:
                reason = f"the gateway answered http {status}"
                wait = _retry_after(response, wait)
            else:
                # The request itself is wrong, retrying cannot help
                return CaseResult(
                    id=case.id,
                    status="error",
                    http_status=status,
                    attempts=attempt,
                    error=_error_detail(response),
                )
        if attempt < attempts:
            await sleep(wait)

    return CaseResult(id=case.id, status="invalid", attempts=attempts, error=reason)


async def run_cases(
    cases: list[EvalCase],
    config: RunConfig,
    api_key: str,
    *,
    client=None,
    sleep=asyncio.sleep,
    progress=None,
) -> EvalRun:
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0,
        )

    run = EvalRun(
        name=config.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=asdict(config),
    )
    try:
        for index, case in enumerate(cases, 1):
            result = await run_case(client, case, config, sleep)
            run.results.append(result)
            if progress:
                progress(index, len(cases), case, result)
            if index < len(cases) and config.delay_seconds > 0:
                await sleep(config.delay_seconds)
    finally:
        if owns_client:
            await client.aclose()
    return run