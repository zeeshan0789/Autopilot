from collections import Counter

import pytest

from app.bench.workload import build_workload
from app.mlrouter.generate import generate_cases


@pytest.fixture(scope="module")
def cases():
    return generate_cases()


def test_the_workload_has_the_expected_mix(cases):
    work = build_workload(cases)
    assert len(work) == 150
    assert Counter(w.kind for w in work) == {"unique": 60, "repeat": 50, "paraphrase": 40}
    assert [w.id for w in work] == [f"w{i:03d}" for i in range(150)]


def test_every_family_has_five_unique_prompts(cases):
    family = {c.id: c.category for c in cases}
    unique = [w for w in build_workload(cases) if w.kind == "unique"]
    assert len(set(family[w.base_id] for w in unique)) == 12
    assert set(Counter(family[w.base_id] for w in unique).values()) == {5}


def test_a_prompt_is_asked_for_the_first_time_before_it_is_repeated(cases):
    seen = set()
    for item in build_workload(cases):
        if item.kind == "unique":
            seen.add(item.base_id)
        else:
            assert item.base_id in seen


def test_repeats_are_identical_and_rewordings_keep_the_original_text(cases):
    work = build_workload(cases)
    first = {w.base_id: w for w in work if w.kind == "unique"}
    for item in work:
        base = first[item.base_id]
        assert item.checks == base.checks
        if item.kind == "repeat":
            assert item.messages == base.messages
        if item.kind == "paraphrase":
            assert item.messages != base.messages
            assert base.messages[-1]["content"] in item.messages[-1]["content"]
            assert item.messages[:-1] == base.messages[:-1]  # a system prompt is untouched


def test_all_three_rewordings_are_used(cases):
    first = {w.base_id: w for w in build_workload(cases) if w.kind == "unique"}
    forms = set()
    for item in build_workload(cases):
        if item.kind == "paraphrase":
            text, original = item.messages[-1]["content"], first[item.base_id].messages[-1]["content"]
            forms.add("prefix" if text.endswith(original) else "suffix")
    assert forms == {"prefix", "suffix"}


def test_the_workload_is_deterministic_and_depends_on_the_seed(cases):
    assert build_workload(cases) == build_workload(cases)
    assert build_workload(cases, seed=1) != build_workload(cases, seed=2)


def test_the_requested_sizes_are_respected(cases):
    work = build_workload(cases, unique_per_family=2, repeats=3, paraphrases=1)
    assert Counter(w.kind for w in work) == {"unique": 24, "repeat": 3, "paraphrase": 1}