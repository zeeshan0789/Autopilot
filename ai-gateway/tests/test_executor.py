import asyncio

import pytest

from app.core.executor import (
    AllProvidersFailedError,
    call_with_retry,
    execute_with_fallback,
)
from app.providers.base import ProviderRateLimitError, ProviderTransientError
from app.providers.mock import MockProvider
from app.schemas.openai import ChatCompletionRequest


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Backoff mein asli intezar nahi, taake tests tez chalein."""

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("app.core.executor.asyncio.sleep", fake_sleep)


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "hello"}]
    )


# ---------- Retry ----------


def test_retry_succeeds_after_transient_failures():
    provider = MockProvider(fail_first=2)
    response = asyncio.run(call_with_retry(provider, make_request()))
    assert response.choices[0].message.content == "Mock reply: hello"
    assert provider._calls == 3  # 2 fail + 1 kamyab


def test_retry_gives_up_after_max_retries():
    provider = MockProvider(fail_first=5)
    with pytest.raises(ProviderTransientError):
        asyncio.run(call_with_retry(provider, make_request()))
    assert provider._calls == 3  # 1 pehli + 2 retry


def test_rate_limit_is_not_retried():
    provider = MockProvider(mode="rate_limit")
    with pytest.raises(ProviderRateLimitError):
        asyncio.run(call_with_retry(provider, make_request()))
    assert provider._calls == 1


# ---------- Fallback ----------


def test_no_fallback_when_primary_works():
    chain = [
        (MockProvider(name="primary"), "m1"),
        (MockProvider(name="backup"), "m2"),
    ]
    result = asyncio.run(execute_with_fallback(chain, make_request()))
    assert result.provider == "primary"
    assert result.model == "m1"
    assert result.fallback_used is False
    assert result.failed_attempts == []


def test_fallback_used_when_primary_fails():
    chain = [
        (MockProvider(name="primary", mode="error"), "m1"),
        (MockProvider(name="backup"), "m2"),
    ]
    result = asyncio.run(execute_with_fallback(chain, make_request()))
    assert result.provider == "backup"
    assert result.model == "m2"
    assert result.fallback_used is True
    assert len(result.failed_attempts) == 1
    assert result.failed_attempts[0].provider == "primary"


def test_fallback_after_rate_limit_without_retry():
    primary = MockProvider(name="primary", mode="rate_limit")
    chain = [(primary, "m1"), (MockProvider(name="backup"), "m2")]
    result = asyncio.run(execute_with_fallback(chain, make_request()))
    assert result.fallback_used is True
    assert primary._calls == 1  # 429 pe retry nahi hua


def test_fallback_after_retries_are_exhausted():
    primary = MockProvider(name="primary", fail_first=10)
    chain = [(primary, "m1"), (MockProvider(name="backup"), "m2")]
    result = asyncio.run(execute_with_fallback(chain, make_request()))
    assert result.provider == "backup"
    assert primary._calls == 3  # pehle 3 koshishein, phir fallback


def test_all_providers_fail():
    chain = [
        (MockProvider(name="a", mode="error"), "m1"),
        (MockProvider(name="b", mode="rate_limit"), "m2"),
    ]
    with pytest.raises(AllProvidersFailedError) as exc:
        asyncio.run(execute_with_fallback(chain, make_request()))
    assert len(exc.value.attempts) == 2