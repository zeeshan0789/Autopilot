import asyncio
import random
from collections import Counter

import pytest

from app.core.prompts import (
    PromptError,
    PromptStore,
    PromptVersionData,
    choose_version,
    fill,
    render_messages,
)
from app.schemas.openai import ChatCompletionRequest, Message


def ver(number, weight=1, system=None, template=None):
    return PromptVersionData(number, system or f"You are version {number}.", template, weight)


def user(text):
    return Message(role="user", content=text)


# ---------- Choosing a version ----------


def test_weights_decide_the_share_of_traffic():
    versions = [ver(1, 90), ver(2, 10)]
    rng = random.Random(1)
    counts = Counter(choose_version("p", versions, rng=rng).version for _ in range(10000))
    assert 8800 < counts[1] < 9200
    assert counts[1] + counts[2] == 10000


def test_a_version_with_weight_zero_is_never_served():
    versions = [ver(1, 0), ver(2, 5)]
    rng = random.Random(2)
    assert {choose_version("p", versions, rng=rng).version for _ in range(200)} == {2}


def test_all_weights_zero_is_an_error():
    with pytest.raises(PromptError) as exc:
        choose_version("p", [ver(1, 0), ver(2, 0)])
    assert exc.value.status_code == 409


def test_a_forced_version_wins_even_with_weight_zero():
    assert choose_version("p", [ver(1, 100), ver(2, 0)], forced=2).version == 2


def test_forcing_a_missing_version_is_a_404():
    with pytest.raises(PromptError) as exc:
        choose_version("p", [ver(1)], forced=9)
    assert exc.value.status_code == 404


def test_the_same_user_always_gets_the_same_version():
    versions = [ver(1, 50), ver(2, 50)]
    for name in ("sara", "omar", "li"):
        first = choose_version("p", versions, sticky_key=name).version
        assert all(choose_version("p", versions, sticky_key=name).version == first for _ in range(20))


def test_different_users_are_spread_over_the_versions():
    versions = [ver(1, 50), ver(2, 50)]
    served = {choose_version("p", versions, sticky_key=f"user{i}").version for i in range(60)}
    assert served == {1, 2}


# ---------- Rendering ----------


def test_system_prompt_comes_first_and_variables_are_filled():
    messages = render_messages(
        ver(1, system="You greet {{ name }} warmly."), [user("hi")], {"name": "Sara"}
    )
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == "You greet Sara warmly."
    assert messages[1].content == "hi"


def test_caller_system_messages_stay_after_the_prompt():
    caller = [Message(role="system", content="Be brief."), user("hi")]
    messages = render_messages(ver(1), caller, {})
    assert [m.content for m in messages] == ["You are version 1.", "Be brief.", "hi"]


def test_user_template_wraps_the_last_user_message():
    caller = [user("first"), Message(role="assistant", content="ok"), user("second")]
    messages = render_messages(ver(1, template="Question: {{input}} ({{tone}})"), caller, {"tone": "kind"})
    assert messages[-1].content == "Question: second (kind)"
    assert messages[1].content == "first"  # earlier turns are untouched


def test_missing_variable_is_a_400():
    with pytest.raises(PromptError) as exc:
        render_messages(ver(1, system="Hello {{name}}"), [user("hi")], {})
    assert exc.value.status_code == 400
    assert "name" in str(exc.value)


def test_user_template_needs_a_user_message():
    with pytest.raises(PromptError) as exc:
        render_messages(ver(1, template="{{input}}"), [Message(role="assistant", content="x")], {})
    assert exc.value.status_code == 400


def test_values_are_not_expanded_a_second_time():
    assert fill("A {{x}} B", {"x": "{{y}}", "y": "boom"}) == "A {{y}} B"


# ---------- The store ----------


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeStore(PromptStore):
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.loads = 0
        self.fail = False

    async def _load(self, name):
        self.loads += 1
        if self.fail:
            raise OSError("database is down")
        return self.data.get(name)


def test_versions_are_remembered_for_the_ttl():
    clock = Clock()
    store = FakeStore({"p": [ver(1)]}, ttl_seconds=10, clock=clock)
    asyncio.run(store.get_versions("p"))
    asyncio.run(store.get_versions("p"))
    assert store.loads == 1
    clock.now += 11
    asyncio.run(store.get_versions("p"))
    assert store.loads == 2


def test_last_known_versions_are_used_when_the_database_is_down():
    clock = Clock()
    store = FakeStore({"p": [ver(1)]}, ttl_seconds=10, clock=clock)
    asyncio.run(store.get_versions("p"))
    clock.now += 11
    store.fail = True
    assert asyncio.run(store.get_versions("p"))[0].version == 1


def test_database_down_and_nothing_remembered_is_a_503():
    store = FakeStore({}, clock=Clock())
    store.fail = True
    with pytest.raises(PromptError) as exc:
        asyncio.run(store.get_versions("p"))
    assert exc.value.status_code == 503


def test_unknown_prompt_is_a_404():
    store = FakeStore({}, clock=Clock())
    request = ChatCompletionRequest(
        model="mock",
        messages=[{"role": "user", "content": "hi"}],
        autopilot={"prompt": "missing"},
    )
    with pytest.raises(PromptError) as exc:
        asyncio.run(store.resolve(request))
    assert exc.value.status_code == 404