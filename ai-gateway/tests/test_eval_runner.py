import asyncio
import json

import httpx
import pytest

from app.evals.cases import EvalCase
from app.evals.runner import EvalAbort, RunConfig, load_run, run_cases, save_run


def case(case_id="c1", text="hi"):
    return EvalCase(
        id=case_id,
        category="x",
        messages=[{"role": "user", "content": text}],
        checks=[{"type": "contains", "value": "ok"}],
    )


def ok(text="ok", fallback=False, model="openai/gpt-oss-20b", tokens=12):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {"total_tokens": tokens}},
        headers={"x-autopilot-fallback": str(fallback).lower(), "x-autopilot-model": model},
    )


class Gateway:
    """Answers with the given response makers in order, then keeps repeating the last one."""

    def __init__(self, *makers):
        self.makers = list(makers)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        maker = self.makers.pop(0) if len(self.makers) > 1 else self.makers[0]
        return maker()


def run(cases, gateway, **config):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(gateway), base_url="http://gateway")
        try:
            return await run_cases(cases, RunConfig(**config), "ap-test", client=client, sleep=fake_sleep)
        finally:
            await client.aclose()

    return asyncio.run(go()), sleeps


def test_a_good_answer_is_recorded():
    result = run([case()], Gateway(ok))[0].results[0]
    assert (result.status, result.answer, result.attempts) == ("ok", "ok", 1)
    assert result.served_model == "openai/gpt-oss-20b"
    assert result.total_tokens == 12
    assert result.fallback_used is False
    assert result.latency_ms is not None


def test_the_cache_is_switched_off_for_eval_requests():
    gateway = Gateway(ok)
    run([case(text="What is 2 + 2?")], gateway, model="auto", temperature=0.0)
    payload = gateway.requests[0]
    assert payload["autopilot"] == {"cache": False}
    assert payload["model"] == "auto"
    assert payload["temperature"] == 0.0
    assert payload["messages"] == [{"role": "user", "content": "What is 2 + 2?"}]


def test_a_managed_prompt_is_passed_on():
    gateway = Gateway(ok)
    run([case()], gateway, prompt="support", version=2, variables={"name": "Sara"})
    assert gateway.requests[0]["autopilot"] == {
        "cache": False,
        "prompt": "support",
        "version": 2,
        "variables": {"name": "Sara"},
    }


def test_a_fallback_answer_is_retried():
    gateway = Gateway(lambda: ok(fallback=True, model="qwen2.5:0.5b"), ok)
    run_result, sleeps = run([case()], gateway)
    result = run_result.results[0]
    assert (result.status, result.attempts, result.fallback_used) == ("ok", 2, False)
    assert sleeps == [2.0]


def test_a_fallback_that_never_clears_is_invalid_not_scored():
    gateway = Gateway(lambda: ok(fallback=True, model="qwen2.5:0.5b"))
    run_result, sleeps = run([case()], gateway, retries=2)
    result = run_result.results[0]
    assert (result.status, result.attempts) == ("invalid", 3)
    assert "fallback" in result.error
    assert len(sleeps) == 2  # no pointless wait after the last attempt


def test_fallback_answers_can_be_allowed():
    gateway = Gateway(lambda: ok(fallback=True, model="qwen2.5:0.5b"))
    result = run([case()], gateway, allow_fallback=True)[0].results[0]
    assert (result.status, result.fallback_used) == ("ok", True)


def test_rate_limit_waits_for_retry_after():
    gateway = Gateway(lambda: httpx.Response(429, headers={"retry-after": "7"}, json={}), ok)
    run_result, sleeps = run([case()], gateway)
    assert run_result.results[0].status == "ok"
    assert sleeps == [7.0]


def test_server_errors_are_retried_then_marked_invalid():
    gateway = Gateway(lambda: httpx.Response(502, json={"detail": "providers failed"}))
    result = run([case()], gateway, retries=1)[0].results[0]
    assert (result.status, result.attempts) == ("invalid", 2)
    assert "502" in result.error


def test_network_errors_are_retried():
    def refuse():
        raise httpx.ConnectError("connection refused")

    result = run([case()], Gateway(refuse, ok))[0].results[0]
    assert (result.status, result.attempts) == ("ok", 2)


def test_a_rejected_key_stops_the_run():
    gateway = Gateway(lambda: httpx.Response(401, json={"detail": "Invalid API key"}))
    with pytest.raises(EvalAbort):
        run([case()], gateway)


def test_a_bad_request_is_an_error_without_retries():
    gateway = Gateway(lambda: httpx.Response(400, json={"detail": "missing variable 'name'"}))
    result = run([case()], gateway)[0].results[0]
    assert (result.status, result.attempts) == ("error", 1)
    assert "missing variable" in result.error
    assert len(gateway.requests) == 1


def test_cases_are_paced():
    run_result, sleeps = run([case("a"), case("b"), case("c")], Gateway(ok), delay_seconds=1.5)
    assert len(run_result.results) == 3
    assert sleeps == [1.5, 1.5]  # between cases, not after the last one


def test_progress_is_reported():
    seen = []
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(Gateway(ok)), base_url="http://gateway")
        try:
            await run_cases(
                [case("a"), case("b")],
                RunConfig(delay_seconds=0),
                "ap-test",
                client=client,
                sleep=fake_sleep,
                progress=lambda i, n, c, r: seen.append((i, n, c.id, r.status)),
            )
        finally:
            await client.aclose()

    asyncio.run(go())
    assert seen == [(1, 2, "a", "ok"), (2, 2, "b", "ok")]
    assert sleeps == []


def test_a_run_survives_saving_and_loading(tmp_path):
    run_result, _ = run([case("a")], Gateway(ok), name="demo", model="openai/gpt-oss-20b")
    path = tmp_path / "runs" / "demo.json"
    save_run(run_result, path)
    loaded = load_run(path)
    assert loaded.name == "demo"
    assert loaded.config["model"] == "openai/gpt-oss-20b"
    assert loaded.results[0].answer == "ok"
    assert "api" not in json.dumps(loaded.config).lower() or "ap-test" not in json.dumps(loaded.config)