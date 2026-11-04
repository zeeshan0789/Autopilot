import pytest

from app.core.policy import (
    PolicyRouter,
    PolicySettings,
    Rule,
    extract_features,
    policy_router,
)
from app.providers.registry import registry
from app.schemas.openai import ChatCompletionRequest
from app.settings import load_policy_config, load_pricing_config

TIERS = {"simple": "small-model", "complex": "big-model"}


def make_request(text="hello", model="auto", **fields) -> ChatCompletionRequest:
    messages = fields.pop("messages", None) or [{"role": "user", "content": text}]
    return ChatCompletionRequest(model=model, messages=messages, **fields)


def make_router(*rules, enabled=True, default_tier="simple") -> PolicyRouter:
    return PolicyRouter(
        PolicySettings(
            tiers=TIERS, default_tier=default_tier, rules=tuple(rules), enabled=enabled
        )
    )


def decide(router, text="hello", **fields):
    return router.decide(make_request(text, **fields))


# ---------- The shipped config ----------


def test_shipped_config_is_valid():
    settings = PolicySettings.from_config(load_policy_config())
    assert settings.alias
    assert settings.rules


def test_every_tier_model_is_configured_and_priced():
    """A tier pointing at a model nobody serves would turn every routed request into a 404."""
    settings = PolicySettings.from_config(load_policy_config())
    priced = set(load_pricing_config()["models"])
    for tier, model in settings.tiers.items():
        assert registry.for_model(model), f"tier '{tier}': {model} is not in providers.yaml"
        assert model in priced, f"tier '{tier}': {model} has no price in pricing.yaml"


def test_shipped_policy_routes_the_obvious_cases():
    assert policy_router.decide(make_request("What is the capital of France?")).rule == "default"
    assert policy_router.decide(make_request("Compare Postgres and MySQL")).tier == "complex"
    assert policy_router.decide(make_request("hello", model="mock")).rule == "explicit"


# ---------- Settings validation ----------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tiers={}, default_tier="simple"),
        dict(tiers=TIERS, default_tier="missing"),
        dict(tiers={"simple": ""}, default_tier="simple"),
        dict(tiers=TIERS, default_tier="simple", alias=" "),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "nope", {"min_chars": 1}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {"min_chars": -1}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {"min_chars": True}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {"min_chars": "5"}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {"typo_condition": 1}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {"any_keywords": []}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {"any_keywords": "prove"}),)),
        dict(tiers=TIERS, default_tier="simple", rules=(Rule("r", "complex", {"contains_code": "yes"}),)),
        dict(
            tiers=TIERS,
            default_tier="simple",
            rules=(
                Rule("same", "complex", {"min_chars": 1}),
                Rule("same", "complex", {"min_chars": 2}),
            ),
        ),
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PolicySettings(**kwargs)


# ---------- Which requests are routed ----------


def test_explicit_model_is_never_routed():
    router = make_router(Rule("always", "complex", {"min_chars": 0}))
    d = decide(router, "anything", model="openai/gpt-oss-20b")
    assert (d.model, d.rule, d.routed, d.tier) == ("openai/gpt-oss-20b", "explicit", False, None)


def test_the_alias_is_routed():
    d = decide(make_router(), "hello")
    assert (d.model, d.rule, d.routed, d.tier) == ("small-model", "default", True, "simple")


def test_disabled_policy_treats_the_alias_as_an_ordinary_model_name():
    router = make_router(Rule("always", "complex", {"min_chars": 0}), enabled=False)
    d = decide(router, "hello")
    assert (d.model, d.rule, d.routed) == ("auto", "explicit", False)


# ---------- The rules ----------


def test_first_matching_rule_wins():
    router = make_router(
        Rule("first", "complex", {"min_chars": 1}),
        Rule("second", "simple", {"min_chars": 1}),
    )
    assert decide(router, "x").rule == "first"


def test_all_conditions_of_a_rule_must_hold():
    router = make_router(Rule("both", "complex", {"min_chars": 20, "any_keywords": ["analyze"]}))
    assert decide(router, "analyze").rule == "default"  # keyword but too short
    assert decide(router, "x" * 30).rule == "default"  # long but no keyword
    assert decide(router, "please analyze this " + "x" * 30).rule == "both"


def test_length_rules():
    router = make_router(Rule("long", "complex", {"min_chars": 50}), Rule("tiny", "simple", {"max_chars": 5}))
    assert decide(router, "x" * 50).rule == "long"
    assert decide(router, "x" * 49).rule == "default"
    assert decide(router, "hi").rule == "tiny"


def test_system_prompt_does_not_count_towards_length():
    router = make_router(Rule("long", "complex", {"min_chars": 50}))
    messages = [
        {"role": "system", "content": "You are a very helpful assistant. " * 20},
        {"role": "user", "content": "hi"},
    ]
    assert router.decide(make_request(messages=messages)).rule == "default"


def test_history_length_rule():
    router = make_router(Rule("history", "complex", {"min_messages": 4}))
    turns = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(4)
    ]
    assert router.decide(make_request(messages=turns)).rule == "history"
    assert router.decide(make_request(messages=turns[:3])).rule == "default"


def test_question_count_rule():
    router = make_router(Rule("multi", "complex", {"min_questions": 3}))
    assert decide(router, "A? B? C?").rule == "multi"
    assert decide(router, "A? B?").rule == "default"


def test_large_output_rule():
    router = make_router(Rule("long_answer", "complex", {"min_max_tokens": 800}))
    assert decide(router, "x", max_tokens=800).rule == "long_answer"
    assert decide(router, "x", max_tokens=799).rule == "default"
    assert decide(router, "x").rule == "default"  # no max_tokens at all


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Explain this step by step", True),
        ("PLEASE ANALYZE the data", True),
        ("We analyzed it yesterday", False),  # whole words only
        ("Is this comparable?", False),
        ("Compare A and B", False),  # "compare" is not in this test rule
    ],
)
def test_keywords_match_whole_words_in_any_case(text, expected):
    router = make_router(Rule("kw", "complex", {"any_keywords": ["step by step", "analyze"]}))
    assert (decide(router, text).rule == "kw") is expected


@pytest.mark.parametrize(
    "text,is_code",
    [
        ("Fix this:\n```python\nprint(1)\n```", True),
        ("def add(a, b): return a + b", True),
        ("const f = (x) => x + 1", True),
        ("#include <stdio.h>", True),
        ("import os\nprint(os.name)", True),
        ("class Foo(Base):", True),
        ("function add(a, b) { return a + b }", True),
        ("What is a class action lawsuit?", False),
        ("What are the import duties on cars?", False),
        ("Tell me about first class travel", False),
        ("What is the capital of France?", False),
    ],
)
def test_code_detection(text, is_code):
    assert extract_features(make_request(text)).has_code is is_code


def test_code_rule_routes_to_the_complex_tier():
    router = make_router(Rule("code", "complex", {"contains_code": True}))
    assert decide(router, "```x```").tier == "complex"
    assert decide(router, "hello").tier == "simple"


# ---------- Failing safe ----------


def test_a_policy_bug_falls_back_to_the_default_tier(monkeypatch):
    def boom(request):
        raise RuntimeError("bug in feature extraction")

    monkeypatch.setattr("app.core.policy.extract_features", boom)
    d = decide(make_router(Rule("r", "complex", {"min_chars": 0})), "hello")
    assert (d.model, d.rule, d.routed) == ("small-model", "policy_error", True)