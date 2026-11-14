from collections import Counter

import pytest

from app.evals.cases import load_cases
from app.evals.checks import run_checks, validate_check
from app.mlrouter.data import parse_case_id, split_of
from app.mlrouter.generate import generate_cases, write_cases

EXPECTED = {
    "arith": 60, "wordproblem": 36, "sequence": 30, "format_list": 30, "format_words": 30,
    "format_json": 24, "extract": 36, "sort_numbers": 24, "count_letters": 24,
    "system_rule": 30, "factual": 55, "reverse_text": 21,
}


@pytest.fixture(scope="module")
def cases():
    return generate_cases()


def test_the_pool_has_the_expected_size_and_families(cases):
    assert len(cases) == 400
    assert Counter(c.category for c in cases) == EXPECTED


def test_all_three_difficulty_levels_are_present(cases):
    levels = Counter(parse_case_id(c.id)[1] for c in cases)
    assert set(levels) == {1, 2, 3}
    assert min(levels.values()) >= 100


def test_ids_and_prompts_are_unique(cases):
    assert len({c.id for c in cases}) == len(cases)
    prompts = {(c.messages[0]["content"] if c.messages[0]["role"] == "system" else "", c.messages[-1]["content"]) for c in cases}
    assert len(prompts) == len(cases)


def test_every_check_is_valid(cases):
    for case in cases:
        for check in case.checks:
            validate_check(check)


def test_every_reference_answer_passes_its_own_checks(cases):
    """The checks and the expected answers must agree, or the labels of the router would be wrong."""
    for case in cases:
        failed = [r for r in run_checks(case.checks, case.reference) if not r.passed]
        assert not failed, (case.id, case.reference, failed)


def test_generation_is_deterministic():
    assert generate_cases(7) == generate_cases(7)
    assert generate_cases(7) != generate_cases(8)


def test_the_split_is_stable_and_roughly_a_quarter(cases):
    shares = Counter(split_of(c.id) for c in cases)
    assert shares["train"] + shares["test"] == 400
    assert 0.15 <= shares["test"] / 400 <= 0.35
    assert split_of(cases[0].id) == split_of(cases[0].id)


def test_handwritten_cases_are_a_separate_set():
    assert split_of("fact_01") == "handwritten"
    assert parse_case_id("fact_01") == ("handwritten", None)


def test_case_ids_are_parsed():
    assert parse_case_id("gen_format_list_d3_007") == ("format_list", 3)
    assert parse_case_id("gen_arith_d1_000") == ("arith", 1)


def test_cases_survive_writing_and_loading(cases, tmp_path):
    path = tmp_path / "cases.jsonl"
    write_cases(path, cases)
    loaded = load_cases(path)
    assert len(loaded) == 400
    assert loaded[0].id == cases[0].id and loaded[0].reference == cases[0].reference
    assert [c.checks for c in loaded] == [c.checks for c in cases]