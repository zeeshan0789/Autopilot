from decimal import Decimal

import pytest

from app.cache.embedder import Embedder
from app.cache.service import CacheService, CacheSettings
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def service():
    embedder = Embedder()
    try:
        embedder.encode("warm up")
    except Exception as e:
        pytest.skip(f"Embedding model not available ({type(e).__name__})")
    return CacheService(CacheSettings(), embedder)


def request_for(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": text}])


def learn(service, text: str, namespace: str) -> None:
    lookup = service.lookup(request_for(text), namespace)
    response = ChatCompletionResponse(
        model="m",
        choices=[Choice(message=Message(role="assistant", content="some answer"))],
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )
    assert service.store(
        lookup,
        response,
        provider="mock",
        estimated_cost_usd=Decimal("0"),
        fallback_used=False,
    )


def test_repeating_the_same_prompt_hits(service):
    learn(service, "What is the capital of France?", namespace="real-repeat")
    lookup = service.lookup(request_for("What is the capital of France?"), "real-repeat")
    assert lookup.status == "hit"
    assert lookup.similarity > 0.99


def test_changed_year_is_never_a_hit(service):
    learn(service, "What was Apple revenue in 2024?", namespace="real-year")
    lookup = service.lookup(request_for("What was Apple revenue in 2025?"), "real-year")
    assert lookup.status in ("miss", "blocked")


def test_negated_question_is_never_a_hit(service):
    learn(service, "Is it safe to eat raw eggs?", namespace="real-negation")
    lookup = service.lookup(request_for("Is it not safe to eat raw eggs?"), "real-negation")
    assert lookup.status in ("miss", "blocked")